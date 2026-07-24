"""LoRA training pipeline — orchestration package.

Modules:
    config           - loads .env and resolves paths
    prepare_dataset  - resize/convert/auto-orient images + write captions
    caption          - caption helpers (template trigger-token, optional BLIP)
    comfy_client     - minimal ComfyUI REST/WebSocket API client
    io_utils         - atomic filesystem writes and copies
    validate         - generate a validation grid from a trained LoRA
    train            - thin wrapper around ai-toolkit's run.py
    pipeline         - prep -> train -> validate, one command
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import (
        caption,
        comfy_client,
        config,
        io_utils,
        pipeline,
        prepare_dataset,
        train,
        validate,
    )

__all__ = [
    "caption",
    "comfy_client",
    "config",
    "io_utils",
    "pipeline",
    "prepare_dataset",
    "train",
    "validate",
]
