"""Prepare a training dataset from a folder of raw photos.

What it does:
  * finds images (jpg/png/webp/heic-if-supported) in --src
  * auto-orients via EXIF, converts to RGB PNG, flattens transparency on white
  * resizes so the long edge == --size (default 1024), preserving aspect ratio
  * writes <name>.png + <name>.txt (caption) pairs into --out
  * prints a QC report: count, resolution spread, tiny/huge images, failures

Usage (from repo root):
    python -m src.prepare_dataset --src "C:/photos/raw" --name person_v1
    python -m src.prepare_dataset --src "C:/photos/raw" --name person_v1 --vision
    python -m src.prepare_dataset --src "C:/photos/raw" --name person_v1 --trigger "sks person"

Dataset quality drives ~80% of LoRA quality. Aim for 15-40 sharp, varied photos
of ONE person: different angles, expressions, lighting, and backgrounds. Avoid
near-duplicates, heavy filters, other people, and motion blur.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TypedDict, cast

from PIL import Image, ImageOps

from . import config
from .caption import BLIP_MODEL_ID, BLIP_MODEL_REVISION, make_caption, source_caption
from .comfy_activity import check_orphaned_comfy_prompts
from .dataset_pairs import scan_dataset_pairs
from .io_utils import atomic_write_bytes, atomic_write_text
from .run_guardian import active_run_guard_covers, pipeline_run_guard

VALID_EXT = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}
MANIFEST_NAME = ".source-manifest.json"
MANIFEST_SCHEMA_VERSION = 3


class _Report(TypedDict):
    ok: int
    skipped: int
    small: int
    failed: int
    widths: list[int]
    heights: list[int]
    caption_lengths: list[int]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_images(src: Path) -> Iterator[Path]:
    for p in sorted(src.rglob("*")):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            yield p


def _deduplicate_images(
    images: list[Path],
) -> tuple[list[Path], list[tuple[Path, Path]]]:
    """Keep one exact file while preserving every source photo untouched."""

    unique: list[Path] = []
    duplicates: list[tuple[Path, Path]] = []
    digests: dict[str, Path] = {}
    for image in images:
        fingerprint = _sha256_file(image)
        original = digests.get(fingerprint)
        if original is None:
            digests[fingerprint] = image
            unique.append(image)
        else:
            duplicates.append((image, original))
    return unique, duplicates


def _is_generated_pair(path: Path) -> bool:
    return path.is_file() and path.stem.isdigit() and path.suffix in {".png", ".txt"}


def _cleanup_staging_dir(staging: Path) -> None:
    for path in staging.iterdir():
        if path.is_file():
            path.unlink()
    staging.rmdir()


def _commit_staged_dataset(staging: Path, out: Path) -> None:
    generated = [path for path in staging.iterdir() if _is_generated_pair(path)]
    expected_names = {path.name for path in generated}

    for path in generated:
        os.replace(path, out / path.name)
    for old_path in out.iterdir():
        if _is_generated_pair(old_path) and old_path.name not in expected_names:
            old_path.unlink()
    os.replace(staging / MANIFEST_NAME, out / MANIFEST_NAME)


def captioner_identity(caption_mode: str) -> str:
    """Return the exact caption implementation identity recorded for reuse."""

    if caption_mode == "blip":
        return f"{BLIP_MODEL_ID}@{BLIP_MODEL_REVISION}"
    if caption_mode == "template":
        return "template-v1"
    raise ValueError(f"Unsupported caption mode: {caption_mode}")


def _source_manifest(
    src: Path,
    images: list[Path],
    *,
    size: int,
    trigger: str,
    caption_mode: str,
) -> dict[str, object]:
    sources: list[dict[str, int | str]] = []
    digests: dict[str, Path] = {}
    for image in images:
        sha256 = _sha256_file(image)
        if sha256 in digests:
            raise ValueError(
                f"Duplicate source images have identical content: "
                f"{digests[sha256]} and {image}."
            )
        digests[sha256] = image
        sidecar = image.with_suffix(".txt")
        sources.append(
            {
                "path": image.relative_to(src).as_posix(),
                "size": image.stat().st_size,
                "mtime_ns": image.stat().st_mtime_ns,
                "sha256": sha256,
                **(
                    {"caption_sha256": _sha256_file(sidecar)}
                    if sidecar.is_file()
                    else {}
                ),
            }
        )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "settings": {
            "size": size,
            "trigger": trigger,
            "caption_mode": caption_mode,
            "captioner": captioner_identity(caption_mode),
        },
        "sources": sources,
    }


def _prepared_output_manifest(folder: Path) -> list[dict[str, str]]:
    return [
        {"image": path.name, "sha256": _sha256_file(path)}
        for path in sorted(folder.iterdir())
        if path.is_file() and path.stem.isdigit() and path.suffix == ".png"
    ]


def prepared_output_manifest_issue(folder: Path, manifest: object) -> str | None:
    """Describe the first generated-image integrity problem, if any."""

    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or not isinstance(manifest.get("outputs"), list)
        or not folder.is_dir()
    ):
        return "The preparation record has no valid prepared-image inventory."

    expected: dict[str, str] = {}
    for entry in cast(list[object], manifest["outputs"]):
        if not isinstance(entry, dict):
            return "The preparation record has a damaged prepared-image entry."
        name = entry.get("image")
        sha256 = entry.get("sha256")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not Path(name).stem.isdigit()
            or Path(name).suffix != ".png"
            or name in expected
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            return "The preparation record has a damaged prepared-image entry."
        expected[name] = sha256
    if not expected:
        return "The preparation record does not list any prepared images."

    try:
        actual = {path.name: path for path in scan_dataset_pairs(folder).image_files}
        missing = sorted(expected.keys() - actual.keys())
        if missing:
            return f"The prepared image {missing[0]} is missing."
        unexpected = sorted(actual.keys() - expected.keys())
        if unexpected:
            return (
                f"The prepared image {unexpected[0]} was not created by this "
                "preparation run."
            )
        for name, path in actual.items():
            if _sha256_file(path) != expected[name]:
                return f"The prepared image {name} changed after preparation."
    except OSError as exc:
        return f"The prepared images could not be verified: {exc}"
    return None


def prepared_outputs_match_manifest(folder: Path, manifest: object) -> bool:
    """Return whether generated images still match their preparation record.

    Captions are deliberately excluded because the desktop exposes them for user
    editing. Their presence, readability, and identity phrase are checked by the
    dataset-pair review before reuse.
    """

    return prepared_output_manifest_issue(folder, manifest) is None


def source_files_match_manifest(src: Path, manifest: object) -> bool:
    """Return whether the current unique source set exactly matches a manifest.

    The desktop uses this before reusing already-prepared pairs.  Rebuilding the
    source entries through the same path as preparation catches added/removed
    unique content and changes even when a file's size and timestamp were
    preserved. Source caption edits also require preparation again; edits to
    the prepared captions remain available without rebuilding.
    """

    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or not isinstance(manifest.get("sources"), list)
        or not isinstance(manifest.get("settings"), dict)
        or not src.is_dir()
    ):
        return False
    settings = manifest["settings"]
    size = settings.get("size")
    trigger = settings.get("trigger")
    caption_mode = settings.get("caption_mode")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or not isinstance(trigger, str)
        or not trigger.strip()
        or caption_mode not in {"template", "blip"}
    ):
        return False

    try:
        candidates = list(_iter_images(src))
        images, _duplicates = _deduplicate_images(candidates)
        current = _source_manifest(
            src,
            images,
            size=size,
            trigger=trigger.strip(),
            caption_mode=caption_mode,
        )
    except (OSError, ValueError):
        return False
    previous_sources = cast(list[object], manifest["sources"])
    current_sources = cast(list[object], current["sources"])
    if not all(
        isinstance(entry, dict) and isinstance(entry.get("sha256"), str)
        for entry in previous_sources
    ):
        return False
    return {
        (entry["sha256"], entry.get("caption_sha256", ""))
        for entry in current_sources
        if isinstance(entry, dict)
    } == {
        (entry["sha256"], entry.get("caption_sha256", ""))
        for entry in previous_sources
        if isinstance(entry, dict)
    }


def _validate_source_manifest(
    out: Path,
    current: dict[str, object],
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    manifest_path = out / MANIFEST_NAME
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"Dataset source manifest is invalid; use --overwrite: {exc}"
            ) from exc
        if (
            not isinstance(previous, dict)
            or previous.get("schema_version") != MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError(
                "Dataset source manifest uses an older format; use --overwrite "
                "to rebuild it safely."
            )
        if previous.get("sources") != current["sources"]:
            raise ValueError(
                "Source images changed or captions were edited since preparation; "
                "use --overwrite to rebuild it safely."
            )
        if previous.get("settings") != current["settings"]:
            raise ValueError(
                "Dataset preparation settings changed since this dataset was "
                "prepared; use --overwrite to rebuild it with the new settings."
            )
        if not prepared_outputs_match_manifest(out, previous):
            raise ValueError(
                "Prepared images changed since this dataset was created; use "
                "--overwrite to rebuild and verify them safely."
            )
        return
    has_generated_pairs = any(
        path.is_file() and path.stem.isdigit() and path.suffix in {".png", ".txt"}
        for path in out.iterdir()
    )
    if has_generated_pairs:
        raise ValueError(
            "Existing dataset has no source manifest; use --overwrite once "
            "to establish a safe incremental baseline."
        )


def _process_one(path: Path, size: int) -> Image.Image:
    """Open, auto-orient, flatten to RGB, resize long edge to `size`."""
    with Image.open(path) as source:
        img: Image.Image = ImageOps.exif_transpose(source)  # honor camera rotation
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")

    w, h = img.size
    long_edge = max(w, h)
    if long_edge != size:
        scale = size / long_edge
        new = (max(1, round(w * scale)), max(1, round(h * scale)))
        resample = Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.BICUBIC
        img = img.resize(new, resample)
    return img


def _prepare_images(
    images: list[Path],
    target: Path,
    *,
    size: int,
    trigger: str,
    caption_mode: str,
    overwrite: bool,
) -> _Report:
    report: _Report = {
        "ok": 0,
        "skipped": 0,
        "small": 0,
        "failed": 0,
        "widths": [],
        "heights": [],
        "caption_lengths": [],
    }

    for i, path in enumerate(images, start=1):
        stem = f"{i:03d}"
        img_out = target / f"{stem}.png"
        txt_out = target / f"{stem}.txt"
        if img_out.exists() and txt_out.exists() and not overwrite:
            print(f"  [skip] {img_out.name} exists (use --overwrite)")
            try:
                with Image.open(img_out) as existing:
                    report["widths"].append(existing.size[0])
                    report["heights"].append(existing.size[1])
                caption = txt_out.read_text(encoding="utf-8").strip()
                if not caption:
                    raise ValueError("caption is empty")
                report["caption_lengths"].append(len(caption.split()))
                report["skipped"] += 1
                continue
            except Exception as e:  # noqa: BLE001 - invalid cached pairs are rebuilt
                print(f"  [warn] Existing pair is invalid ({e}); rebuilding")
        elif (img_out.exists() or txt_out.exists()) and not overwrite:
            print(f"  [warn] Incomplete pair for {stem}; rebuilding both files")
        try:
            with Image.open(path) as orig:
                if max(orig.size) < size:
                    report["small"] += 1
                    print(
                        f"  [warn] {path.name} is small ({orig.size}); will be upscaled"
                    )
            img = _process_one(path, size)
            encoded = io.BytesIO()
            img.save(encoded, "PNG")
            atomic_write_bytes(img_out, encoded.getvalue())
            imported_caption = source_caption(path, trigger)
            if imported_caption is None:
                caption = make_caption(img_out, trigger=trigger, mode=caption_mode)
            else:
                caption = imported_caption
                print(f"  [caption] Imported {path.with_suffix('.txt').name}")
            atomic_write_text(txt_out, caption)
            report["ok"] += 1
            report["caption_lengths"].append(len(caption.split()))
            report["widths"].append(img.size[0])
            report["heights"].append(img.size[1])
        except Exception as e:  # noqa: BLE001 - report corrupt files and continue
            img_out.unlink(missing_ok=True)
            txt_out.unlink(missing_ok=True)
            report["failed"] += 1
            print(f"  [fail] {path.name}: {e}")
    return report


def prepare(
    src: Path,
    name: str,
    size: int = 1024,
    trigger: str = config.DEFAULT_TRIGGER,
    caption_mode: str = "template",
    overwrite: bool = False,
) -> Path:
    name = config.validate_project_name(name)
    output = (config.DATASETS_DIR / name).resolve()
    workspace = config.RUN_CONFIGS_DIR.resolve().parent
    resources = (output,)
    if not active_run_guard_covers(
        project=name,
        workspace=workspace,
        use_accelerator=caption_mode == "blip",
        resource_paths=resources,
    ):
        with pipeline_run_guard(
            project=name,
            workspace=workspace,
            mode="prepare",
            use_accelerator=caption_mode == "blip",
            resource_paths=resources,
        ):
            if caption_mode == "blip":
                check_orphaned_comfy_prompts()
            return _prepare_with_guard(
                src=src,
                name=name,
                size=size,
                trigger=trigger,
                caption_mode=caption_mode,
                overwrite=overwrite,
            )
    if caption_mode == "blip":
        check_orphaned_comfy_prompts()
    return _prepare_with_guard(
        src=src,
        name=name,
        size=size,
        trigger=trigger,
        caption_mode=caption_mode,
        overwrite=overwrite,
    )


def _prepare_with_guard(
    src: Path,
    name: str,
    size: int = 1024,
    trigger: str = config.DEFAULT_TRIGGER,
    caption_mode: str = "template",
    overwrite: bool = False,
) -> Path:
    name = config.validate_project_name(name)
    if not src.is_dir():
        raise FileNotFoundError(f"Source image directory not found: {src}")
    if size <= 0:
        raise ValueError("--size must be greater than zero.")
    if not trigger.strip():
        raise ValueError("Caption trigger cannot be empty.")
    trigger = trigger.strip()
    if caption_mode not in {"template", "blip"}:
        raise ValueError("Caption mode must be 'template' or 'blip'.")
    out = config.DATASETS_DIR / name
    source_path = src.resolve()
    output_path = out.resolve()
    if (
        source_path == output_path
        or source_path.is_relative_to(output_path)
        or output_path.is_relative_to(source_path)
    ):
        raise ValueError(
            "Source and prepared dataset directories cannot contain each other."
        )
    out.mkdir(parents=True, exist_ok=True)

    # Rebuild replaces only our numbered pairs. Preserve other photos, but do
    # not report a ready dataset while leaving extra trainer inputs in place.
    inventory = scan_dataset_pairs(out)
    unmanaged_images = [
        path
        for path in (*inventory.image_files, *inventory.unsupported_image_files)
        if not _is_generated_pair(path)
    ]
    if unmanaged_images:
        names = ", ".join(path.name for path in unmanaged_images)
        raise ValueError(
            f"The prepared folder contains photos this preparation cannot replace: "
            f"{names}. Move those photos out of {out} or use a new project name. "
            "To include them in training, add them to the source photo folder "
            "and prepare again. No existing prepared files were changed."
        )

    candidates = list(_iter_images(src))
    if not candidates:
        raise ValueError(f"No supported images found under {src}")
    images, duplicates = _deduplicate_images(candidates)
    for duplicate, original in duplicates:
        print(
            "  [ignore] Exact duplicate "
            f"{duplicate.relative_to(src).as_posix()} matches "
            f"{original.relative_to(src).as_posix()}"
        )
    manifest = _source_manifest(
        src,
        images,
        size=size,
        trigger=trigger,
        caption_mode=caption_mode,
    )
    _validate_source_manifest(out, manifest, overwrite=overwrite)

    print(
        f"[prepare] {len(images)} unique image(s) -> {out}"
        + (f" ({len(duplicates)} exact duplicate(s) ignored)" if duplicates else "")
    )
    staging: Path | None = None
    if overwrite:
        staging = Path(tempfile.mkdtemp(prefix=f".{name}.prepare-", dir=out.parent))
    target = staging or out
    try:
        report = _prepare_images(
            images,
            target,
            size=size,
            trigger=trigger,
            caption_mode=caption_mode,
            overwrite=overwrite,
        )
        _print_report(name, out, report, trigger)
        if report["ok"] + report["skipped"] == 0:
            raise RuntimeError(f"Dataset preparation produced no usable pairs in {out}")
        manifest["outputs"] = _prepared_output_manifest(target)
        atomic_write_text(
            target / MANIFEST_NAME,
            json.dumps(manifest, indent=2) + "\n",
        )
        if staging is not None:
            _commit_staged_dataset(staging, out)
        return out
    finally:
        if staging is not None and staging.exists():
            _cleanup_staging_dir(staging)


def _print_report(name: str, out: Path, report: _Report, trigger: str) -> None:
    ws, hs = report["widths"], report["heights"]
    print("\n" + "=" * 60)
    print(f"Dataset '{name}' prepared at: {out}")
    print(f"  processed OK : {report['ok']}")
    print(f"  already valid: {report['skipped']}")
    print(f"  upscaled/small: {report['small']}")
    print(f"  failed        : {report['failed']}")
    if ws:
        print(f"  width  range  : {min(ws)}-{max(ws)}")
        print(f"  height range  : {min(hs)}-{max(hs)}")
    lengths = report["caption_lengths"]
    if lengths:
        average = sum(lengths) / len(lengths)
        print(
            f"  caption words : {min(lengths)}-{max(lengths)} (average {average:.1f})"
        )
    print(f"  trigger token : '{trigger}'")
    if report["ok"] + report["skipped"] < 10:
        print("  NOTE: <10 images is thin for a person LoRA; 15-40 is the sweet spot.")
    print("  Next: eyeball the .txt captions, then train:")
    print("        python -m src.train --config configs/person_sdxl.yaml")
    print("=" * 60)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Prepare a LoRA training dataset from raw photos."
    )
    ap.add_argument(
        "--src", required=True, help="Folder of raw photos (searched recursively)."
    )
    ap.add_argument("--name", required=True, help="Dataset name -> datasets/<name>/")
    ap.add_argument(
        "--size", type=int, default=1024, help="Long-edge size in px (default 1024)."
    )
    ap.add_argument(
        "--trigger", default=config.DEFAULT_TRIGGER, help="Trigger token for captions."
    )
    ap.add_argument(
        "--vision",
        action="store_true",
        help="Use BLIP auto-captioning (needs transformers+torch).",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Rewrite existing outputs."
    )
    args = ap.parse_args(argv)

    try:
        project = config.validate_project_name(args.name)
        workspace = config.RUN_CONFIGS_DIR.resolve().parent
        with pipeline_run_guard(
            project=project,
            workspace=workspace,
            mode="prepare",
            use_accelerator=args.vision,
            project_scopes=(config.DATASETS_DIR.resolve().parent,),
            resource_paths=((config.DATASETS_DIR / project).resolve(),),
        ):
            if args.vision:
                check_orphaned_comfy_prompts()
            prepare(
                src=Path(args.src),
                name=project,
                size=args.size,
                trigger=args.trigger,
                caption_mode="blip" if args.vision else "template",
                overwrite=args.overwrite,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
