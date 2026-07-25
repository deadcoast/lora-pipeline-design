"""Central config: loads .env and resolves project paths.

Everything else imports from here so paths live in exactly one place.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def _load_env(path: Path) -> bool:
    """Load a dotenv file when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # keep import-safe even before deps are installed
        return False
    return load_dotenv(path)


# Repo root = parent of this file's parent (src/ -> repo/)
ROOT = Path(__file__).resolve().parent.parent

# Load .env from the repo root if present.
_load_env(ROOT / ".env")


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _path(key: str, default: str) -> Path:
    """Resolve a path from env; relative values are anchored to the repo root."""
    raw = _get(key, default)
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / p)


def _optional_path(key: str) -> Path | None:
    raw = _get(key).strip()
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / p)


_PROJECT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MAX_PROJECT_NAME_LENGTH = 80
_WINDOWS_RESERVED_NAMES = {
    "AUX",
    "CON",
    "NUL",
    "PRN",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
DEFAULT_TRIGGER = "ohwx person"


def validate_project_name(name: str) -> str:
    """Validate names that are used as local dataset and output directories."""
    if len(name) > MAX_PROJECT_NAME_LENGTH:
        raise ValueError(
            f"Project name must be {MAX_PROJECT_NAME_LENGTH} characters or fewer."
        )
    if not _PROJECT_NAME.fullmatch(name):
        raise ValueError(
            "Project name must start with a letter or number and contain only "
            "letters, numbers, dots, underscores, or hyphens."
        )
    if name.endswith(".") or name.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"Project name is not valid as a Windows directory: {name!r}.")
    return name


# --- ComfyUI ---
COMFY_HOST: str = _get("COMFY_HOST", "127.0.0.1:8188")
COMFY_LORA_DIR: Path | None = _optional_path("COMFY_LORA_DIR")
COMFY_SDXL_CKPT: str = _get("COMFY_SDXL_CKPT", "sd_xl_base_1.0.safetensors")
COMFY_INSTALLATION_ID: str = _get("MIVIM_COMFY_INSTALLATION_ID")

# --- ai-toolkit ---
AI_TOOLKIT_DIR: str = _get("AI_TOOLKIT_DIR", "")
AI_TOOLKIT_PYTHON: str = _get("AI_TOOLKIT_PYTHON", "python")

# --- Project paths (relative to repo root unless absolute) ---
DATASETS_DIR: Path = _path("DATASETS_DIR", "datasets")
CHECKPOINTS_DIR: Path = _path("CHECKPOINTS_DIR", "outputs/checkpoints")
GRIDS_DIR: Path = _path("GRIDS_DIR", "outputs/grids")
RUN_CONFIGS_DIR: Path = _path("RUN_CONFIGS_DIR", "outputs/configs")


def summary() -> str:
    return (
        f"ROOT              = {ROOT}\n"
        f"COMFY_HOST        = {COMFY_HOST}\n"
        f"COMFY_LORA_DIR    = {COMFY_LORA_DIR}\n"
        f"COMFY_SDXL_CKPT   = {COMFY_SDXL_CKPT}\n"
        f"COMFY_INSTALLATION_ID = {COMFY_INSTALLATION_ID}\n"
        f"AI_TOOLKIT_DIR    = {AI_TOOLKIT_DIR}\n"
        f"AI_TOOLKIT_PYTHON = {AI_TOOLKIT_PYTHON}\n"
        f"DATASETS_DIR      = {DATASETS_DIR}\n"
        f"CHECKPOINTS_DIR   = {CHECKPOINTS_DIR}\n"
        f"GRIDS_DIR         = {GRIDS_DIR}\n"
        f"RUN_CONFIGS_DIR   = {RUN_CONFIGS_DIR}\n"
    )


if __name__ == "__main__":
    print(summary())
