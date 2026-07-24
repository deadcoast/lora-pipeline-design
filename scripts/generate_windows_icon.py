"""Generate the deterministic MIVIM Windows application icon."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

from PIL import Image, ImageDraw, ImageFont

_CANVAS_SIZE: Final = 1024
_ICON_SIZES: Final = (16, 24, 32, 48, 64, 128, 256)
_WINDOWS_FONTS: Final = (
    Path("C:/Windows/Fonts/segoeuib.ttf"),
    Path("C:/Windows/Fonts/seguisb.ttf"),
    Path("C:/Windows/Fonts/arialbd.ttf"),
)


def create_icon(destination: Path) -> Path:
    """Write a multi-resolution ICO matching the in-app MIVIM mark."""

    destination = destination.resolve()
    canvas = Image.new(
        "RGBA",
        (_CANVAS_SIZE, _CANVAS_SIZE),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (64, 64, 960, 960),
        radius=230,
        fill="#2f80d1",
        outline="#70b9ff",
        width=24,
    )
    font_path = next((path for path in _WINDOWS_FONTS if path.is_file()), None)
    if font_path is None:
        raise FileNotFoundError("Windows did not provide a supported bold UI font.")
    font = ImageFont.truetype(str(font_path), 560)
    bounds = draw.textbbox((0, 0), "M", font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (
            (_CANVAS_SIZE - width) / 2 - bounds[0],
            (_CANVAS_SIZE - height) / 2 - bounds[1] - 12,
        ),
        "M",
        font=font,
        fill="#ffffff",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="ICO", sizes=[(size, size) for size in _ICON_SIZES])
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(create_icon(args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
