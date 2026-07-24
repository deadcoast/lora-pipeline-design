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

from pathlib import Path
from typing import Optional


def template_caption(trigger: str, hint: str = "") -> str:
    """Build a simple caption from the trigger token and an optional hint."""
    hint = hint.strip().strip(",")
    return f"{trigger}, {hint}".strip().strip(",") if hint else trigger


# ---- Optional BLIP captioner (lazy — only imported if actually used) ----

_blip = {"processor": None, "model": None}


def _load_blip():
    if _blip["model"] is not None:
        return
    # Imported lazily so the rest of the pipeline works without transformers/torch.
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch

    name = "Salesforce/blip-image-captioning-large"
    _blip["processor"] = BlipProcessor.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _blip["model"] = BlipForConditionalGeneration.from_pretrained(name).to(device)
    _blip["device"] = device


def blip_caption(image_path: Path, trigger: str) -> str:
    """Describe an image with BLIP and prepend the trigger token."""
    from PIL import Image
    _load_blip()
    proc, model, device = _blip["processor"], _blip["model"], _blip["device"]
    img = Image.open(image_path).convert("RGB")
    inputs = proc(img, return_tensors="pt").to(device)
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
        try:
            return blip_caption(image_path, trigger)
        except Exception as e:  # graceful fallback so prep never hard-fails
            print(f"  [caption] BLIP failed ({e}); using template for {image_path.name}")
            return template_caption(trigger, hint)
    return template_caption(trigger, hint)
