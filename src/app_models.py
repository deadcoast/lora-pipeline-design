"""Typed, UI-independent state for the desktop companion application.

The desktop UI should remain a thin presentation layer.  These models describe
the user's project, persisted integration settings, diagnostics, and pipeline
events without importing Qt or any other GUI framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal

from . import config


class TrainingPreset(StrEnum):
    """Casual-user presets backed by options the shipped pipeline supports."""

    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"

    @property
    def display_name(self) -> str:
        return {
            self.FAST: "Fast preview",
            self.BALANCED: "Balanced",
            self.QUALITY: "Quality review",
        }[self]

    @property
    def description(self) -> str:
        return {
            self.FAST: "A 900-step, rank-8 draft with a baseline and one strength.",
            self.BALANCED: "Recommended 2,000-step, rank-16 identity training.",
            self.QUALITY: "A 3,000-step, rank-32 run with a wider validation sweep.",
        }[self]

    @property
    def image_size(self) -> int:
        return 768 if self is self.FAST else 1024

    @property
    def training_steps(self) -> int:
        return {self.FAST: 900, self.BALANCED: 2000, self.QUALITY: 3000}[self]

    @property
    def validation_strengths(self) -> tuple[float, ...]:
        return {
            self.FAST: (0.0, 0.8),
            self.BALANCED: (0.0, 0.6, 0.8, 1.0),
            self.QUALITY: (0.0, 0.5, 0.65, 0.8, 0.95, 1.1),
        }[self]


@dataclass(slots=True)
class ProjectDraft:
    """The small set of choices needed to run the guided pipeline."""

    project_name: str = ""
    source_dir: Path | None = None
    trigger: str = config.DEFAULT_TRIGGER
    preset: TrainingPreset = TrainingPreset.BALANCED
    use_vision_captions: bool = False
    overwrite_existing: bool = False
    training_steps: int | None = None

    @property
    def effective_steps(self) -> int:
        return self.training_steps or self.preset.training_steps

    def normalized(self) -> ProjectDraft:
        """Return a copy with user-entered text normalized for the pipeline."""
        return ProjectDraft(
            project_name=self.project_name.strip(),
            source_dir=self.source_dir,
            trigger=self.trigger.strip(),
            preset=self.preset,
            use_vision_captions=self.use_vision_captions,
            overwrite_existing=self.overwrite_existing,
            training_steps=self.training_steps,
        )

    def validation_errors(self, *, require_source: bool = True) -> tuple[str, ...]:
        """Return plain-language errors suitable for display next to the form."""
        draft = self.normalized()
        errors: list[str] = []
        try:
            config.validate_project_name(draft.project_name)
        except ValueError as exc:
            errors.append(str(exc))

        if not draft.trigger:
            errors.append("Choose a trigger phrase, such as 'ohwx person'.")
        if draft.training_steps is not None and draft.training_steps <= 0:
            errors.append("Training steps must be greater than zero.")

        if require_source:
            if draft.source_dir is None:
                errors.append("Choose the folder that contains the source photos.")
            elif not draft.source_dir.is_dir():
                errors.append(f"Source photo folder was not found: {draft.source_dir}")
        return tuple(errors)

    def validate(self, *, require_source: bool = True) -> ProjectDraft:
        """Return the normalized draft, or raise one actionable validation error."""
        errors = self.validation_errors(require_source=require_source)
        if errors:
            raise ValueError(errors[0])
        return self.normalized()


@dataclass(frozen=True, slots=True)
class RunResult:
    """One verified desktop result that is safe to restore on next launch."""

    project_name: str
    mode: Literal["prepare", "full"]
    result_path: Path
    completed_at: str
    trigger: str
    preset: TrainingPreset
    checkpoint: str = ""
    lora_name: str = ""
    result_fingerprint: str = ""
    workflow_path: Path | None = None
    workflow_fingerprint: str = ""
    comfy_installation_id: str = ""
    comfy_installation_name: str = ""
    comfy_host: str = ""
    installed_lora_path: Path | None = None
    lora_fingerprint: str = ""
    checkpoint_path: Path | None = None
    checkpoint_fingerprint: str = ""
    starting_strength: float = 0.8
    published_workflow_path: str = ""
    published_workflow_fingerprint: str = ""
    workflow_user_id: str = ""
    workflow_user_name: str = ""
    training_project_name: str = ""
    model_profile: str = "sdxl"


@dataclass(frozen=True, slots=True)
class ValidationRecovery:
    """Durable handoff state after training installed a LoRA but before validation."""

    project_name: str
    state: Literal["validation_pending", "complete"]
    lora_name: str
    installed_lora_path: Path
    expected_grid_path: Path
    trigger: str
    preset: TrainingPreset
    checkpoint: str
    updated_at: str
    attempt_id: str = ""
    comfy_installation_id: str = ""
    comfy_host: str = ""
    lora_fingerprint: str = ""
    checkpoint_path: Path | None = None
    checkpoint_fingerprint: str = ""
    validation_signature: str = ""
    strengths: tuple[float, ...] = ()
    seed: int = 42
    training_project_name: str = ""
    model_profile: str = "sdxl"


@dataclass(slots=True)
class AppSettings:
    """Non-secret application settings persisted between desktop sessions."""

    schema_version: int = 6
    comfy_host: str = "127.0.0.1:8188"
    comfy_installation_id: str = ""
    comfy_user_installation_id: str = ""
    comfy_user_id: str = ""
    comfy_lora_dir: Path | None = None
    comfy_checkpoint: str = "sd_xl_base_1.0.safetensors"
    confirmed_sdxl_checkpoint: str = ""
    model_profile: str = "sdxl"
    training_base_model: Path | None = None
    ai_toolkit_dir: Path | None = None
    ai_toolkit_python: Path | None = None
    datasets_dir: Path = Path("datasets")
    checkpoints_dir: Path = Path("outputs/checkpoints")
    grids_dir: Path = Path("outputs/grids")
    run_configs_dir: Path = Path("outputs/configs")
    last_project: ProjectDraft = field(default_factory=ProjectDraft)
    last_result: RunResult | None = None

    def environment_overrides(self) -> dict[str, str]:
        """Build the exact, non-secret environment consumed by ``src.config``."""
        comfy_host = (
            self.comfy_host.strip()
            .removeprefix("http://")
            .removeprefix("https://")
            .rstrip("/")
        )
        environment = {
            "COMFY_HOST": comfy_host,
            "COMFY_SDXL_CKPT": self.comfy_checkpoint.strip(),
            "MIVIM_MODEL_PROFILE": self.model_profile,
            "MIVIM_COMFY_INSTALLATION_ID": self.comfy_installation_id.strip(),
            "DATASETS_DIR": str(self.datasets_dir),
            "CHECKPOINTS_DIR": str(self.checkpoints_dir),
            "GRIDS_DIR": str(self.grids_dir),
            "RUN_CONFIGS_DIR": str(self.run_configs_dir),
        }
        optional_paths = {
            "COMFY_LORA_DIR": self.comfy_lora_dir,
            "AI_TOOLKIT_DIR": self.ai_toolkit_dir,
            "AI_TOOLKIT_PYTHON": self.ai_toolkit_python,
        }
        environment.update(
            {
                key: str(value)
                for key, value in optional_paths.items()
                if value is not None and str(value).strip()
            }
        )
        return environment
