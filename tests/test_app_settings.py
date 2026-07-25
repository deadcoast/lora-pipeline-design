from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.app_models import (
    AppSettings,
    ProjectDraft,
    RunResult,
    TrainingPreset,
)
from src.app_settings import (
    SETTINGS_SCHEMA_VERSION,
    load_settings,
    save_settings,
    user_app_data_dir,
    with_user_storage_defaults,
)


def test_settings_round_trip_is_versioned_and_keeps_path_semantics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "photos"
    source.mkdir()
    settings = AppSettings(
        comfy_host="127.0.0.1:9000",
        comfy_installation_id="portrait-desktop",
        comfy_user_installation_id="portrait-desktop",
        comfy_user_id="profile-a",
        comfy_lora_dir=Path("D:/Comfy Shared/models/loras"),
        comfy_checkpoint="realistic.safetensors",
        confirmed_sdxl_checkpoint="realistic.safetensors",
        ai_toolkit_dir=Path("D:/tools/ai-toolkit"),
        ai_toolkit_python=Path("D:/tools/ai-toolkit/.venv/Scripts/python.exe"),
        last_result=RunResult(
            project_name="family_portrait",
            mode="full",
            result_path=Path("D:/MIVIM/grids/family_portrait_grid.png"),
            completed_at="2026-07-24T12:00:00+00:00",
            trigger="mivm person",
            preset=TrainingPreset.QUALITY,
            checkpoint="realistic.safetensors",
            lora_name="family_portrait.safetensors",
            result_fingerprint="sha256:verified-result",
            workflow_path=Path("D:/MIVIM/grids/family_portrait_use_in_comfyui.json"),
            workflow_fingerprint="sha256:verified-workflow",
            comfy_installation_id="portrait-desktop",
            comfy_installation_name="Portrait Desktop",
            comfy_host="127.0.0.1:9000",
            installed_lora_path=Path(
                "D:/Comfy Shared/models/loras/family_portrait.safetensors"
            ),
            lora_fingerprint="sha256:verified-lora",
            checkpoint_path=Path(
                "D:/Comfy Shared/models/checkpoints/realistic.safetensors"
            ),
            checkpoint_fingerprint="sha256:verified-checkpoint",
            starting_strength=0.8,
            published_workflow_path=(
                "workflows/MIVIM/family_portrait_use_in_comfyui.json"
            ),
            published_workflow_fingerprint="sha256:verified-workflow",
            workflow_user_id="profile-a",
            workflow_user_name="Portraits",
        ),
        last_project=ProjectDraft(
            project_name="family_portrait",
            source_dir=source,
            trigger="mivm person",
            preset=TrainingPreset.QUALITY,
            use_vision_captions=True,
        ),
    )
    destination = save_settings(settings, tmp_path / "state" / "settings.json")

    saved = json.loads(destination.read_text(encoding="utf-8"))
    loaded = load_settings(destination)

    assert saved["schema_version"] == SETTINGS_SCHEMA_VERSION
    assert "\\" not in saved["comfy_lora_dir"]
    assert loaded == settings


@pytest.mark.parametrize(
    "contents",
    [
        "{not json",
        "[]",
        '{"schema_version": 999, "comfy_host": "unsafe:9999"}',
        '{"schema_version": true, "comfy_host": "unsafe:9999"}',
    ],
)
def test_load_settings_falls_back_for_damaged_or_unknown_documents(
    tmp_path: Path,
    contents: str,
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(contents, encoding="utf-8")

    assert load_settings(path) == AppSettings()


def test_version_one_settings_migrate_without_losing_existing_paths(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "comfy_host": "127.0.0.1:9000",
                "datasets_dir": "D:/MIVIM/datasets",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_settings(path)

    assert loaded.schema_version == SETTINGS_SCHEMA_VERSION
    assert loaded.comfy_host == "127.0.0.1:9000"
    assert loaded.datasets_dir == Path("D:/MIVIM/datasets")
    assert loaded.training_base_model is None


def test_version_four_settings_migrate_to_profile_aware_handoff(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "comfy_host": "127.0.0.1:8299",
                "comfy_installation_id": "portrait-desktop",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_settings(path)

    assert loaded.schema_version == SETTINGS_SCHEMA_VERSION
    assert loaded.comfy_installation_id == "portrait-desktop"
    assert loaded.comfy_user_installation_id == ""
    assert loaded.comfy_user_id == ""


def test_load_settings_keeps_valid_fields_and_defaults_invalid_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": SETTINGS_SCHEMA_VERSION,
                "comfy_host": "127.0.0.1:8282",
                "comfy_lora_dir": 42,
                "datasets_dir": "",
                "last_project": {
                    "project_name": "portrait",
                    "trigger": ["not", "text"],
                    "preset": "unknown",
                    "use_vision_captions": "yes",
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_settings(path)

    assert loaded.comfy_host == "127.0.0.1:8282"
    assert loaded.comfy_lora_dir is None
    assert loaded.datasets_dir == Path("datasets")
    assert loaded.last_project.project_name == "portrait"
    assert loaded.last_project.trigger == "ohwx person"
    assert loaded.last_project.preset is TrainingPreset.BALANCED
    assert loaded.last_project.use_vision_captions is False


def test_settings_and_environment_use_explicit_non_secret_allowlists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMFY_API_TOKEN", "must-not-leak")
    monkeypatch.setenv("HF_TOKEN", "must-not-leak")
    settings = AppSettings(
        comfy_host="http://127.0.0.1:8188/",
        comfy_lora_dir=tmp_path / "loras",
        ai_toolkit_dir=tmp_path / "trainer",
        ai_toolkit_python=tmp_path / "trainer" / "python",
    )
    path = save_settings(settings, tmp_path / "settings.json")

    serialized = path.read_text(encoding="utf-8")
    environment = settings.environment_overrides()

    assert "must-not-leak" not in serialized
    assert "TOKEN" not in serialized.upper()
    assert set(environment) == {
        "COMFY_HOST",
        "COMFY_LORA_DIR",
        "COMFY_SDXL_CKPT",
        "MIVIM_COMFY_INSTALLATION_ID",
        "AI_TOOLKIT_DIR",
        "AI_TOOLKIT_PYTHON",
        "DATASETS_DIR",
        "CHECKPOINTS_DIR",
        "GRIDS_DIR",
        "RUN_CONFIGS_DIR",
    }
    assert all("TOKEN" not in key and "PASSWORD" not in key for key in environment)
    assert environment["COMFY_HOST"] == "127.0.0.1:8188"


def test_user_app_data_path_is_portable_and_overrideable(tmp_path: Path) -> None:
    windows = user_app_data_dir(
        environ={"LOCALAPPDATA": "C:/Users/test/AppData/Local"},
        platform="win32",
        home=Path("C:/Users/test"),
    )
    linux = user_app_data_dir(
        environ={"XDG_CONFIG_HOME": "/var/user-config"},
        platform="linux",
        home=Path("/home/test"),
    )
    overridden = user_app_data_dir(
        environ={"MIVIM_LORA_PIPELINE_DATA_DIR": str(tmp_path)},
        platform="linux",
        home=Path("/home/test"),
    )

    assert windows.as_posix().endswith("AppData/Local/MIVIM/LoRA Pipeline")
    assert linux == Path("/var/user-config/MIVIM/LoRA Pipeline")
    assert overridden == tmp_path


def test_gui_defaults_use_a_writable_user_workspace(tmp_path: Path) -> None:
    settings_path = tmp_path / "MIVIM" / "LoRA Pipeline" / "settings.json"

    settings = with_user_storage_defaults(AppSettings(), settings_path)

    workspace = settings_path.parent / "workspace"
    assert settings.datasets_dir == workspace / "datasets"
    assert settings.checkpoints_dir == workspace / "checkpoints"
    assert settings.grids_dir == workspace / "grids"
    assert settings.run_configs_dir == workspace / "configs"


def test_project_draft_reuses_pipeline_validation(tmp_path: Path) -> None:
    photos = tmp_path / "photos"
    photos.mkdir()

    valid = ProjectDraft(
        project_name=" subject_v1 ",
        source_dir=photos,
        trigger=" custom person ",
    ).validate()

    assert valid.project_name == "subject_v1"
    assert valid.trigger == "custom person"
    assert ProjectDraft(
        project_name="../outside",
        source_dir=photos,
    ).validation_errors()
    assert ProjectDraft(
        project_name="subject_v1",
        source_dir=tmp_path / "missing",
    ).validation_errors() == (
        f"Source photo folder was not found: {tmp_path / 'missing'}",
    )
