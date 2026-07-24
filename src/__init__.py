"""LoRA training pipeline — orchestration package.

Modules:
    desktop          - guided PySide6 desktop application
    app_models       - typed UI-independent project and event models
    app_service      - readiness checks and pipeline command construction
    app_settings     - versioned per-user settings persistence
    comfy_desktop    - read-only Comfy Desktop discovery and live model listing
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
        app_models,
        app_service,
        app_settings,
        caption,
        comfy_client,
        comfy_desktop,
        config,
        desktop,
        io_utils,
        pipeline,
        prepare_dataset,
        train,
        validate,
    )

__all__ = [
    "app_models",
    "app_service",
    "app_settings",
    "caption",
    "comfy_client",
    "comfy_desktop",
    "config",
    "desktop",
    "io_utils",
    "pipeline",
    "prepare_dataset",
    "train",
    "validate",
]
