"""Captioning helpers.

Two strategies:
  1. template   (default) — writes "<trigger>, <class>" plus any per-image hint.
                 Fast, dependency-free, and for a single-person identity LoRA it's
                 often ALL you need: the identity is carried by the trigger token,
                 and over-describing the face can actually hurt.
  2. blip       (--vision) — uses a local BLIP model to describe each image, then
                 prepends the trigger token. Requires transformers + torch.

For a person LoRA the recommended default is:  trigger="ohwx person".
Use a RARE token ("ohwx", "sks", a made-up string) so it doesn't collide with
concepts the base model already knows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"
BLIP_MODEL_REVISION = "353689b859fcf0523410b1806dace5fb46ecdf41"


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
    if mode == "blip":
        return blip_caption(image_path, trigger)
    return template_caption(trigger, hint)
