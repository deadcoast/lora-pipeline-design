"""Central config: loads .env and resolves project paths.

Everything else imports from here so paths live in exactly one place.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # keep import-safe even before deps are installed
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

# Repo root = parent of this file's parent (src/ -> repo/)
ROOT = Path(__file__).resolve().parent.parent

# Load .env from the repo root if present.
load_dotenv(ROOT / ".env")


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _path(key: str, default: str) -> Path:
    """Resolve a path from env; relative values are anchored to the repo root."""
    raw = _get(key, default)
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / p)


# --- ComfyUI ---
COMFY_HOST: str = _get("COMFY_HOST", "127.0.0.1:8188")
COMFY_LORA_DIR: Path = Path(_get("COMFY_LORA_DIR", "")) if _get("COMFY_LORA_DIR") else None
COMFY_SDXL_CKPT: str = _get("COMFY_SDXL_CKPT", "sd_xl_base_1.0.safetensors")

# --- ai-toolkit ---
AI_TOOLKIT_DIR: str = _get("AI_TOOLKIT_DIR", "")
AI_TOOLKIT_PYTHON: str = _get("AI_TOOLKIT_PYTHON", "python")

# --- Project paths (relative to repo root unless absolute) ---
DATASETS_DIR: Path = _path("DATASETS_DIR", "datasets")
CHECKPOINTS_DIR: Path = _path("CHECKPOINTS_DIR", "outputs/checkpoints")
GRIDS_DIR: Path = _path("GRIDS_DIR", "outputs/grids")


def summary() -> str:
    return (
        f"ROOT              = {ROOT}\n"
        f"COMFY_HOST        = {COMFY_HOST}\n"
        f"COMFY_LORA_DIR    = {COMFY_LORA_DIR}\n"
        f"COMFY_SDXL_CKPT   = {COMFY_SDXL_CKPT}\n"
        f"AI_TOOLKIT_DIR    = {AI_TOOLKIT_DIR}\n"
        f"AI_TOOLKIT_PYTHON = {AI_TOOLKIT_PYTHON}\n"
        f"DATASETS_DIR      = {DATASETS_DIR}\n"
        f"CHECKPOINTS_DIR   = {CHECKPOINTS_DIR}\n"
        f"GRIDS_DIR         = {GRIDS_DIR}\n"
    )


if __name__ == "__main__":
    print(summary())
