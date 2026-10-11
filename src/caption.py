"""Import existing captions, or create template/BLIP captions for untagged images.

UTF-8 .txt files beside source images take priority. This preserves manually
edited descriptions and tagger output, including native anime tags.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"
BLIP_MODEL_REVISION = "353689b859fcf0523410b1806dace5fb46ecdf41"
CAPTION_MODES = ("auto", "anime", "template", "blip")


def resolve_caption_mode(mode: str, model_profile: str, *, vision: bool = False) -> str:
    if mode != "auto":
        return mode
    if model_profile != "sdxl":
        return "anime"
    return "blip" if vision else "template"


def caption_contains_trigger(caption: str, trigger: str) -> bool:
    """Return whether a caption contains the required phrase as complete words."""
    normalized_caption = " ".join(caption.casefold().split())
    normalized_trigger = " ".join(trigger.casefold().split())
    if not normalized_caption or not normalized_trigger:
        return False
    start = 0
    while (position := normalized_caption.find(normalized_trigger, start)) >= 0:
        before = normalized_caption[position - 1] if position else ""
        end = position + len(normalized_trigger)
        after = normalized_caption[end] if end < len(normalized_caption) else ""
        if not (before.isalnum() or before == "_") and not (
            after.isalnum() or after == "_"
        ):
            return True
        start = position + 1
    return False


def source_caption(image_path: Path, trigger: str) -> str | None:
    """Read the original sidecar without changing its tags or their order."""
    sidecar = image_path.with_suffix(".txt")
    if not sidecar.is_file():
        return None
    caption = sidecar.read_text(encoding="utf-8-sig").strip()
    if not caption:
        return None
    if caption_contains_trigger(caption, trigger):
        return caption
    return template_caption(trigger, caption)


def template_caption(trigger: str, hint: str = "") -> str:
    """Build a simple caption from the trigger token and an optional hint."""
    hint = hint.strip().strip(",")
    return f"{trigger}, {hint}".strip().strip(",") if hint else trigger


# ---- Optional BLIP captioner (lazy — only imported if actually used) ----


@dataclass
class _BlipState:
    processor: Any | None = None
    model: Any | None = None
    device: str | None = None


_blip = _BlipState()


def _load_blip() -> None:
    if _blip.model is not None:
        return
    # Imported lazily so the rest of the pipeline works without transformers/torch.
    import torch  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForMultimodalLM,
        AutoProcessor,
    )

    _blip.processor = cast(
        Any,
        AutoProcessor.from_pretrained(
            BLIP_MODEL_ID,
            revision=BLIP_MODEL_REVISION,
        ),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cast(
        Any,
        AutoModelForMultimodalLM.from_pretrained(
            BLIP_MODEL_ID,
            revision=BLIP_MODEL_REVISION,
            use_safetensors=True,
        ),
    )
    _blip.model = model.to(device).eval()
    _blip.device = device


def blip_caption(image_path: Path, trigger: str) -> str:
    """Describe an image with BLIP and prepend the trigger token."""
    import torch  # type: ignore[import-not-found]
    from PIL import Image

    _load_blip()
    proc, model, device = _blip.processor, _blip.model, _blip.device
    if proc is None or model is None or device is None:
        raise RuntimeError("BLIP failed to initialize")
    with Image.open(image_path) as source:
        img = source.convert("RGB")
    inputs = proc(img, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=40)
    desc = proc.decode(out[0], skip_special_tokens=True).strip()
    # BLIP often starts with "a "/"an " or a generic subject; keep it, just tag it.
    return f"{trigger}, {desc}".strip().strip(",")


def make_caption(
    image_path: Path,
    trigger: str,
    mode: str = "template",
    hint: str = "",
) -> str:
    if mode == "anime":
        from .anime_tagger import anime_caption

        return anime_caption(image_path, trigger)
    if mode == "blip":
        return blip_caption(image_path, trigger)
    return template_caption(trigger, hint)
