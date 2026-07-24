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
import sys
from pathlib import Path

from PIL import Image, ImageOps

from . import config
from .caption import make_caption

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MIN_RECOMMENDED = 768  # warn if an image's long edge is smaller than this before upscaling


def _iter_images(src: Path):
    for p in sorted(src.rglob("*")):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            yield p


def _process_one(path: Path, size: int) -> Image.Image:
    """Open, auto-orient, flatten to RGB, resize long edge to `size`."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # honor camera rotation
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
        resample = Image.LANCZOS if scale < 1 else Image.BICUBIC
        img = img.resize(new, resample)
    return img


def prepare(
    src: Path,
    name: str,
    size: int = 1024,
    trigger: str = "ohwx person",
    caption_mode: str = "template",
    overwrite: bool = False,
) -> Path:
    out = config.DATASETS_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    images = list(_iter_images(src))
    if not images:
        print(f"[prepare] No images found under {src}")
        sys.exit(1)

    print(f"[prepare] {len(images)} candidate image(s) -> {out}")
    report = {"ok": 0, "small": 0, "failed": 0, "widths": [], "heights": []}

    for i, path in enumerate(images, start=1):
        stem = f"{i:03d}"
        img_out = out / f"{stem}.png"
        txt_out = out / f"{stem}.txt"
        if img_out.exists() and not overwrite:
            print(f"  [skip] {img_out.name} exists (use --overwrite)")
            continue
        try:
            orig = Image.open(path)
            if max(orig.size) < MIN_RECOMMENDED:
                report["small"] += 1
                print(f"  [warn] {path.name} is small ({orig.size}); will be upscaled")
            img = _process_one(path, size)
            img.save(img_out, "PNG")
            caption = make_caption(img_out, trigger=trigger, mode=caption_mode)
            txt_out.write_text(caption, encoding="utf-8")
            report["ok"] += 1
            report["widths"].append(img.size[0])
            report["heights"].append(img.size[1])
        except Exception as e:
            report["failed"] += 1
            print(f"  [fail] {path.name}: {e}")

    _print_report(name, out, report, trigger)
    return out


def _print_report(name, out, report, trigger):
    ws, hs = report["widths"], report["heights"]
    print("\n" + "=" * 60)
    print(f"Dataset '{name}' prepared at: {out}")
    print(f"  processed OK : {report['ok']}")
    print(f"  upscaled/small: {report['small']}")
    print(f"  failed        : {report['failed']}")
    if ws:
        print(f"  width  range  : {min(ws)}-{max(ws)}")
        print(f"  height range  : {min(hs)}-{max(hs)}")
    print(f"  trigger token : '{trigger}'")
    if report["ok"] < 10:
        print("  NOTE: <10 images is thin for a person LoRA; 15-40 is the sweet spot.")
    print("  Next: eyeball the .txt captions, then train:")
    print(f"        python -m src.train --config configs/person_sdxl.yaml")
    print("=" * 60)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prepare a LoRA training dataset from raw photos.")
    ap.add_argument("--src", required=True, help="Folder of raw photos (searched recursively).")
    ap.add_argument("--name", required=True, help="Dataset name -> datasets/<name>/")
    ap.add_argument("--size", type=int, default=1024, help="Long-edge size in px (default 1024).")
    ap.add_argument("--trigger", default="ohwx person", help="Trigger token for captions.")
    ap.add_argument("--vision", action="store_true", help="Use BLIP auto-captioning (needs transformers+torch).")
    ap.add_argument("--overwrite", action="store_true", help="Rewrite existing outputs.")
    args = ap.parse_args(argv)

    prepare(
        src=Path(args.src),
        name=args.name,
        size=args.size,
        trigger=args.trigger,
        caption_mode="blip" if args.vision else "template",
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
