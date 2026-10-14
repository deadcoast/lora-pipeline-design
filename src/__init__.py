"""LoRA training pipeline — orchestration package.

Modules:
    desktop          - guided PySide6 desktop application
    app_models       - typed UI-independent project, settings, and result models
    app_service      - readiness checks and pipeline command construction
    app_settings     - versioned per-user settings persistence
    dataset_review   - prepared-photo and caption review dialog
    photo_review     - source-photo issue review dialog
    comfy_desktop    - read-only Comfy Desktop discovery and live model listing
    config           - loads .env and resolves paths
    prepare_dataset  - resize/convert/auto-orient images + write captions
    caption          - caption helpers (template trigger-token, optional BLIP)
    comfy_activity   - durable ownership of submitted ComfyUI prompts
    comfy_client     - minimal ComfyUI REST/WebSocket API client
    io_utils         - atomic filesystem writes and copies
    process_control  - Studio/worker/trainer crash-lifetime ownership
    run_guardian     - cross-process project/resource/accelerator leases
    validate         - generate a validation grid from a trained LoRA
    train            - thin wrapper around ai-toolkit's run.py
    pipeline         - prep -> train -> validate, one command
"""

from typing import TYPE_CHECKING

__version__ = "0.9.0"

if TYPE_CHECKING:
    from . import (
        app_models,
        app_service,
        app_settings,
        caption,
        comfy_activity,
        comfy_client,
        comfy_desktop,
        config,
        dataset_review,
        desktop,
        io_utils,
        photo_review,
        pipeline,
        prepare_dataset,
        process_control,
        run_guardian,
        train,
        validate,
    )

__all__ = [
    "__version__",
    "app_models",
    "app_service",
    "app_settings",
    "caption",
    "comfy_activity",
    "comfy_client",
    "comfy_desktop",
    "config",
    "dataset_review",
    "desktop",
    "io_utils",
    "photo_review",
    "pipeline",
    "prepare_dataset",
    "process_control",
    "run_guardian",
    "train",
    "validate",
]
