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
from typing import TypedDict

from PIL import Image, ImageOps

from . import config
from .caption import BLIP_MODEL_ID, BLIP_MODEL_REVISION, make_caption
from .io_utils import atomic_write_bytes, atomic_write_text

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


class _Report(TypedDict):
    ok: int
    skipped: int
    small: int
    failed: int
    widths: list[int]
    heights: list[int]
    caption_lengths: list[int]


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
        digest = hashlib.sha256()
        with image.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        fingerprint = digest.hexdigest()
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
        digest = hashlib.sha256()
        with image.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        sha256 = digest.hexdigest()
        if sha256 in digests:
            raise ValueError(
                f"Duplicate source images have identical content: "
                f"{digests[sha256]} and {image}."
            )
        digests[sha256] = image
        sources.append(
            {
                "path": image.relative_to(src).as_posix(),
                "size": image.stat().st_size,
                "mtime_ns": image.stat().st_mtime_ns,
                "sha256": sha256,
            }
        )

    captioner = (
        f"{BLIP_MODEL_ID}@{BLIP_MODEL_REVISION}"
        if caption_mode == "blip"
        else "template-v1"
    )
    return {
        "schema_version": 2,
        "settings": {
            "size": size,
            "trigger": trigger,
            "caption_mode": caption_mode,
            "captioner": captioner,
        },
        "sources": sources,
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
        if previous == current:
            return
        if not isinstance(previous, dict) or previous.get("schema_version") != 2:
            raise ValueError(
                "Dataset source manifest uses an older format; use --overwrite "
                "to rebuild it safely."
            )
        if previous.get("sources") != current["sources"]:
            raise ValueError(
                "Source images changed since this dataset was prepared; "
                "use --overwrite to rebuild it safely."
            )
        raise ValueError(
            "Dataset preparation settings changed since this dataset was prepared; "
            "use --overwrite to rebuild it with the new settings."
        )
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
            caption = make_caption(img_out, trigger=trigger, mode=caption_mode)
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
        prepare(
            src=Path(args.src),
            name=args.name,
            size=args.size,
            trigger=args.trigger,
            caption_mode="blip" if args.vision else "template",
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
