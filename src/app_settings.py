"""Versioned, defensive persistence for the desktop companion application."""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from .app_models import AppSettings, ProjectDraft, RunResult, TrainingPreset
from .io_utils import atomic_write_text

SETTINGS_SCHEMA_VERSION = 6
DATA_DIR_OVERRIDE = "MIVIM_LORA_PIPELINE_DATA_DIR"
_APPLICATION_DIRECTORY = Path("MIVIM") / "LoRA Pipeline"


def user_app_data_dir(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: Path | None = None,
) -> Path:
    """Return a per-user application-data directory on major desktop platforms."""
    values = os.environ if environ is None else environ
    override = values.get(DATA_DIR_OVERRIDE, "").strip()
    if override:
        return Path(override).expanduser()

    platform_name = sys.platform if platform is None else platform
    user_home = Path.home() if home is None else home
    if platform_name.startswith("win"):
        base = values.get("LOCALAPPDATA") or values.get("APPDATA")
        root = Path(base) if base else user_home / "AppData" / "Local"
    elif platform_name == "darwin":
        root = user_home / "Library" / "Application Support"
    else:
        xdg_config = values.get("XDG_CONFIG_HOME", "").strip()
        root = Path(xdg_config).expanduser() if xdg_config else user_home / ".config"
    return root / _APPLICATION_DIRECTORY


def default_settings_path() -> Path:
    return user_app_data_dir() / "settings.json"


def with_user_storage_defaults(
    settings: AppSettings,
    settings_path: Path | None = None,
) -> AppSettings:
    """Move untouched GUI output defaults into a writable per-user workspace."""

    defaults = AppSettings()
    base = (settings_path or default_settings_path()).parent / "workspace"
    return replace(
        settings,
        datasets_dir=(
            base / "datasets"
            if settings.datasets_dir == defaults.datasets_dir
            else settings.datasets_dir
        ),
        checkpoints_dir=(
            base / "checkpoints"
            if settings.checkpoints_dir == defaults.checkpoints_dir
            else settings.checkpoints_dir
        ),
        grids_dir=(
            base / "grids"
            if settings.grids_dir == defaults.grids_dir
            else settings.grids_dir
        ),
        run_configs_dir=(
            base / "configs"
            if settings.run_configs_dir == defaults.run_configs_dir
            else settings.run_configs_dir
        ),
    )


def _text(value: object, default: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        return default
    cleaned = value.strip()
    return cleaned if cleaned or allow_empty else default


def _path(value: object, default: Path | None) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return default
    return Path(value.strip()).expanduser()


def _boolean(value: object, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _project_from_json(value: object, default: ProjectDraft) -> ProjectDraft:
    if not isinstance(value, dict):
        return default
    raw_preset = value.get("preset")
    try:
        preset = (
            TrainingPreset(raw_preset)
            if isinstance(raw_preset, str)
            else default.preset
        )
    except ValueError:
        preset = default.preset
    steps = value.get("training_steps")
    return ProjectDraft(
        project_name=_text(
            value.get("project_name"),
            default.project_name,
            allow_empty=True,
        ),
        source_dir=_path(value.get("source_dir"), default.source_dir),
        trigger=_text(value.get("trigger"), default.trigger),
        preset=preset,
        use_anime_tags=_boolean(value.get("use_anime_tags"), default.use_anime_tags),
        training_steps=(
            steps
            if isinstance(steps, int) and not isinstance(steps, bool) and steps > 0
            else None
        ),
        use_vision_captions=_boolean(
            value.get("use_vision_captions"),
            default.use_vision_captions,
        ),
        overwrite_existing=_boolean(
            value.get("overwrite_existing"),
            default.overwrite_existing,
        ),
    )


def _run_result_from_json(value: object) -> RunResult | None:
    if not isinstance(value, dict):
        return None
    project_name = _text(value.get("project_name"), "", allow_empty=True)
    result_path = _path(value.get("result_path"), None)
    completed_at = _text(value.get("completed_at"), "", allow_empty=True)
    mode = value.get("mode")
    raw_preset = value.get("preset")
    if not isinstance(raw_preset, str):
        return None
    try:
        preset = TrainingPreset(raw_preset)
    except ValueError:
        return None
    raw_starting_strength = value.get("starting_strength", 0.8)
    starting_strength = (
        float(raw_starting_strength)
        if isinstance(raw_starting_strength, (int, float))
        and not isinstance(raw_starting_strength, bool)
        and math.isfinite(raw_starting_strength)
        else 0.8
    )
    if (
        not project_name
        or result_path is None
        or not completed_at
        or mode not in {"prepare", "full"}
    ):
        return None
    return RunResult(
        model_profile=_text(value.get("model_profile"), "sdxl"),
        project_name=project_name,
        training_project_name=_text(
            value.get("training_project_name"), "", allow_empty=True
        ),
        mode=mode,
        result_path=result_path,
        completed_at=completed_at,
        trigger=_text(value.get("trigger"), "", allow_empty=True),
        preset=preset,
        checkpoint=_text(value.get("checkpoint"), "", allow_empty=True),
        lora_name=_text(value.get("lora_name"), "", allow_empty=True),
        result_fingerprint=_text(
            value.get("result_fingerprint"),
            "",
            allow_empty=True,
        ),
        workflow_path=_path(value.get("workflow_path"), None),
        workflow_fingerprint=_text(
            value.get("workflow_fingerprint"),
            "",
            allow_empty=True,
        ),
        comfy_installation_id=_text(
            value.get("comfy_installation_id"),
            "",
            allow_empty=True,
        ),
        comfy_installation_name=_text(
            value.get("comfy_installation_name"),
            "",
            allow_empty=True,
        ),
        comfy_host=_text(value.get("comfy_host"), "", allow_empty=True),
        installed_lora_path=_path(value.get("installed_lora_path"), None),
        lora_fingerprint=_text(
            value.get("lora_fingerprint"),
            "",
            allow_empty=True,
        ),
        checkpoint_path=_path(value.get("checkpoint_path"), None),
        checkpoint_fingerprint=_text(
            value.get("checkpoint_fingerprint"),
            "",
            allow_empty=True,
        ),
        starting_strength=starting_strength,
        published_workflow_path=_text(
            value.get("published_workflow_path"),
            "",
            allow_empty=True,
        ),
        published_workflow_fingerprint=_text(
            value.get("published_workflow_fingerprint"),
            "",
            allow_empty=True,
        ),
        workflow_user_id=_text(
            value.get("workflow_user_id"),
            "",
            allow_empty=True,
        ),
        workflow_user_name=_text(
            value.get("workflow_user_name"),
            "",
            allow_empty=True,
        ),
    )


def _settings_from_json(value: object) -> AppSettings:
    defaults = AppSettings()
    if not isinstance(value, dict):
        return defaults
    version = value.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in {1, 2, 3, 4, 5, SETTINGS_SCHEMA_VERSION}
    ):
        return defaults

    return AppSettings(
        model_profile=_text(value.get("model_profile"), "sdxl"),
        schema_version=SETTINGS_SCHEMA_VERSION,
        comfy_host=_text(value.get("comfy_host"), defaults.comfy_host),
        comfy_installation_id=_text(
            value.get("comfy_installation_id"),
            defaults.comfy_installation_id,
            allow_empty=True,
        ),
        comfy_user_installation_id=_text(
            value.get("comfy_user_installation_id"),
            defaults.comfy_user_installation_id,
            allow_empty=True,
        ),
        comfy_user_id=_text(
            value.get("comfy_user_id"),
            defaults.comfy_user_id,
            allow_empty=True,
        ),
        comfy_lora_dir=_path(
            value.get("comfy_lora_dir"),
            defaults.comfy_lora_dir,
        ),
        comfy_checkpoint=_text(
            value.get("comfy_checkpoint"),
            defaults.comfy_checkpoint,
        ),
        confirmed_sdxl_checkpoint=_text(
            value.get("confirmed_sdxl_checkpoint"),
            defaults.confirmed_sdxl_checkpoint,
            allow_empty=True,
        ),
        training_base_model=_path(
            value.get("training_base_model"),
            defaults.training_base_model,
        ),
        ai_toolkit_dir=_path(
            value.get("ai_toolkit_dir"),
            defaults.ai_toolkit_dir,
        ),
        ai_toolkit_python=_path(
            value.get("ai_toolkit_python"),
            defaults.ai_toolkit_python,
        ),
        datasets_dir=_path(value.get("datasets_dir"), defaults.datasets_dir)
        or defaults.datasets_dir,
        checkpoints_dir=_path(
            value.get("checkpoints_dir"),
            defaults.checkpoints_dir,
        )
        or defaults.checkpoints_dir,
        grids_dir=_path(value.get("grids_dir"), defaults.grids_dir)
        or defaults.grids_dir,
        run_configs_dir=_path(
            value.get("run_configs_dir"),
            defaults.run_configs_dir,
        )
        or defaults.run_configs_dir,
        last_project=_project_from_json(
            value.get("last_project"),
            defaults.last_project,
        ),
        last_result=_run_result_from_json(value.get("last_result")),
    )


def load_settings(path: Path | None = None) -> AppSettings:
    """Load valid known fields, falling back safely for absent or damaged files."""
    source = default_settings_path() if path is None else Path(path)
    try:
        parsed: Any = json.loads(source.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return AppSettings()
    return _settings_from_json(parsed)


def _path_for_json(value: Path | None) -> str | None:
    return None if value is None else value.as_posix()


def _project_to_json(project: ProjectDraft) -> dict[str, object]:
    return {
        "project_name": project.project_name,
        "source_dir": _path_for_json(project.source_dir),
        "trigger": project.trigger,
        "preset": project.preset.value,
        "training_steps": project.training_steps,
        "use_anime_tags": project.use_anime_tags,
        "use_vision_captions": project.use_vision_captions,
        "overwrite_existing": project.overwrite_existing,
    }


def _run_result_to_json(result: RunResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "project_name": result.project_name,
        "training_project_name": result.training_project_name,
        "mode": result.mode,
        "result_path": _path_for_json(result.result_path),
        "completed_at": result.completed_at,
        "trigger": result.trigger,
        "preset": result.preset.value,
        "checkpoint": result.checkpoint,
        "model_profile": result.model_profile,
        "lora_name": result.lora_name,
        "result_fingerprint": result.result_fingerprint,
        "workflow_path": _path_for_json(result.workflow_path),
        "workflow_fingerprint": result.workflow_fingerprint,
        "comfy_installation_id": result.comfy_installation_id,
        "comfy_installation_name": result.comfy_installation_name,
        "comfy_host": result.comfy_host,
        "installed_lora_path": _path_for_json(result.installed_lora_path),
        "lora_fingerprint": result.lora_fingerprint,
        "checkpoint_path": _path_for_json(result.checkpoint_path),
        "checkpoint_fingerprint": result.checkpoint_fingerprint,
        "starting_strength": result.starting_strength,
        "published_workflow_path": result.published_workflow_path,
        "published_workflow_fingerprint": result.published_workflow_fingerprint,
        "workflow_user_id": result.workflow_user_id,
        "workflow_user_name": result.workflow_user_name,
    }


def _settings_to_json(settings: AppSettings) -> dict[str, object]:
    """Serialize an explicit allowlist so credentials can never leak into JSON."""
    return {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "comfy_host": settings.comfy_host,
        "comfy_installation_id": settings.comfy_installation_id,
        "comfy_user_installation_id": settings.comfy_user_installation_id,
        "comfy_user_id": settings.comfy_user_id,
        "comfy_lora_dir": _path_for_json(settings.comfy_lora_dir),
        "comfy_checkpoint": settings.comfy_checkpoint,
        "model_profile": settings.model_profile,
        "confirmed_sdxl_checkpoint": settings.confirmed_sdxl_checkpoint,
        "training_base_model": _path_for_json(settings.training_base_model),
        "ai_toolkit_dir": _path_for_json(settings.ai_toolkit_dir),
        "ai_toolkit_python": _path_for_json(settings.ai_toolkit_python),
        "datasets_dir": _path_for_json(settings.datasets_dir),
        "checkpoints_dir": _path_for_json(settings.checkpoints_dir),
        "grids_dir": _path_for_json(settings.grids_dir),
        "run_configs_dir": _path_for_json(settings.run_configs_dir),
        "last_project": _project_to_json(settings.last_project),
        "last_result": _run_result_to_json(settings.last_result),
    }


def save_settings(settings: AppSettings, path: Path | None = None) -> Path:
    """Atomically save only the versioned, non-secret settings allowlist."""
    destination = default_settings_path() if path is None else Path(path)
    payload = json.dumps(
        _settings_to_json(settings),
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    atomic_write_text(destination, f"{payload}\n")
    return destination
