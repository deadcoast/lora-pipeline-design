"""LoRA training pipeline — orchestration package.

Modules:
    config           - loads .env and resolves paths
    prepare_dataset  - resize/convert/auto-orient images + write captions
    caption          - caption helpers (template trigger-token, optional BLIP)
    comfy_client     - minimal ComfyUI REST/WebSocket API client
    validate         - generate a validation grid from a trained LoRA
    train            - thin wrapper around ai-toolkit's run.py
    pipeline         - prep -> train -> validate, one command
"""

__all__ = [
    "config",
    "prepare_dataset",
    "caption",
    "comfy_client",
    "validate",
    "train",
    "pipeline",
]
