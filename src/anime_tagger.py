"""Local WD ViT v3 inference for editable anime training captions."""

from __future__ import annotations

import csv
import os
import tempfile
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

MODEL_ID = "SmilingWolf/wd-vit-tagger-v3"
MODEL_REVISION = "790b0e92cefd2a0221451604e7831fe643ab7c4f"  # Publisher's v2.0 release
GENERAL_THRESHOLD = 0.35
CAPTIONER_ID = f"{MODEL_ID}@{MODEL_REVISION}:general-{GENERAL_THRESHOLD}:v1"
MODEL_BYTES = 378536310
_FILES = {"model.onnx": MODEL_BYTES, "selected_tags.csv": 308468}


def model_cache_dir() -> Path:
    override = os.environ.get("MIVIM_TAGGER_CACHE", "").strip()
    if override:
        return Path(override).expanduser()
    # Lazy import keeps caption selection independent of settings persistence.
    from .app_settings import user_app_data_dir

    return user_app_data_dir() / "models" / "wd-vit-tagger-v3" / MODEL_REVISION


def model_is_cached() -> bool:
    folder = model_cache_dir()
    return all(
        (folder / name).is_file() and (folder / name).stat().st_size == size
        for name, size in _FILES.items()
    )


def _download_file(folder: Path, name: str, size: int) -> Path:
    destination = folder / name
    if destination.is_file() and destination.stat().st_size == size:
        return destination
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
    request = urllib.request.Request(url, headers={"User-Agent": "MIVIM-LoRA-Studio"})
    temporary: Path | None = None
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            tempfile.NamedTemporaryFile(
                dir=folder, suffix=".part", delete=False
            ) as stream,
        ):
            temporary = Path(stream.name)
            downloaded = 0
            reported_at = 0.0
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                downloaded += len(chunk)
                if name == "model.onnx" and time.monotonic() - reported_at >= 1:
                    print(
                        f"[tagger] Downloading model: {downloaded / 1024**2:.0f} "
                        f"/ {size / 1024**2:.0f} MB",
                        flush=True,
                    )
                    reported_at = time.monotonic()
        if downloaded != size:
            raise OSError(
                f"Incomplete download of {name}: {downloaded} of {size} bytes"
            )
        os.replace(temporary, destination)
        return destination
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            "The anime tagger could not finish its first model download. "
            "Connect to the internet and retry preparation. "
            f"Completed downloads are reused. {exc}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class AnimeTagger:
    def __init__(self) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]

        folder = model_cache_dir()
        folder.mkdir(parents=True, exist_ok=True)
        print("[tagger] Loading WD ViT v3 for local anime tagging…", flush=True)
        paths = {
            name: _download_file(folder, name, size) for name, size in _FILES.items()
        }
        with paths["selected_tags.csv"].open(encoding="utf-8", newline="") as stream:
            self.labels = tuple(csv.DictReader(stream))
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        self.session: Any = ort.InferenceSession(
            str(paths["model.onnx"]),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.size = int(self.session.get_inputs()[0].shape[1])
        print("[tagger] Anime tagger ready (local CPU).", flush=True)

    def tags(self, path: Path) -> list[str]:
        import numpy as np

        with Image.open(path) as source:
            rgba = ImageOps.exif_transpose(source).convert("RGBA")
        white = Image.new("RGBA", rgba.size, "white")
        white.alpha_composite(rgba)
        rgb = white.convert("RGB")
        edge = max(rgb.size)
        square = Image.new("RGB", (edge, edge), "white")
        square.paste(rgb, ((edge - rgb.width) // 2, (edge - rgb.height) // 2))
        square = square.resize((self.size, self.size), Image.Resampling.BICUBIC)
        # WD's ONNX export consumes NHWC float32 BGR in the 0..255 range.
        tensor = np.asarray(square, dtype=np.float32)[None, :, :, ::-1].copy()
        scores = self.session.run(None, {self.input_name: tensor})[0][0]
        selected = [
            (row["name"], float(score))
            for row, score in zip(self.labels, scores, strict=True)
            if row["category"] == "0" and float(score) >= GENERAL_THRESHOLD
        ]
        # General tags describe the image; the user's trigger supplies identity.
        # Do not inject a guessed franchise character or a content-rating label.
        selected.sort(key=lambda item: item[1], reverse=True)
        return [name for name, _score in selected]


@lru_cache(maxsize=1)
def load_tagger() -> AnimeTagger:
    return AnimeTagger()


def anime_caption(image_path: Path, trigger: str) -> str:
    from .caption import template_caption

    tags = load_tagger().tags(image_path)
    if not tags:
        print(
            f"[tagger] No confident tags for {image_path.name}; "
            "review its trigger-only caption.",
            flush=True,
        )
    return template_caption(trigger, ", ".join(tags))
