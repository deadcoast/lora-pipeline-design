from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src import app_service
from src.app_models import AppSettings, ProjectDraft, TrainingPreset
from src.app_service import ApplicationService, inspect_source, stage_from_log
from src.comfy_desktop import (
    ComfyApiStatus,
    ComfyDesktopDiscovery,
    ComfyLauncher,
    ComfyModelRoot,
)


def _write_images(folder: Path, count: int, *, size: int = 1200) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new(
            "RGB",
            (size + index, size - 100 + index),
            (index * 12, 30, 80),
        ).save(folder / f"{index:03d}.png")


def test_inspect_source_reports_real_image_quality_counts(tmp_path: Path) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 3, size=600)
    (photos / "broken.jpg").write_text("not an image", encoding="utf-8")
    (photos / "notes.txt").write_text("ignored", encoding="utf-8")

    result = inspect_source(photos, target_size=1024)

    assert result.image_count == 4
    assert result.usable_count == 3
    assert result.unreadable_count == 1
    assert result.small_count == 3
    assert result.duplicate_count == 0


def test_inspect_source_flags_exact_duplicates_before_pipeline_run(
    tmp_path: Path,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 2)
    (photos / "copy.png").write_bytes((photos / "000.png").read_bytes())

    result = inspect_source(photos)

    assert result.image_count == 3
    assert result.duplicate_count == 1
    assert result.unique_usable_count == 2


def test_readiness_keeps_photo_preparation_available_when_comfy_is_offline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 15)
    loras = tmp_path / "models" / "loras"
    loras.mkdir(parents=True)
    toolkit = tmp_path / "ai-toolkit"
    toolkit.mkdir()
    (toolkit / "run.py").write_text("", encoding="utf-8")
    toolkit_python = toolkit / "venv" / "Scripts" / "python.exe"
    toolkit_python.parent.mkdir(parents=True)
    toolkit_python.write_text("", encoding="utf-8")
    base_model = tmp_path / "models" / "checkpoints" / "sdxl.safetensors"
    base_model.parent.mkdir(parents=True)
    base_model.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(
            False,
            None,
            (),
            (),
            "not running",
        ),
    )
    settings = AppSettings(
        comfy_lora_dir=loras,
        comfy_checkpoint="sdxl.safetensors",
        training_base_model=base_model,
        ai_toolkit_dir=toolkit,
        ai_toolkit_python=toolkit_python,
        checkpoints_dir=tmp_path / "outputs" / "checkpoints",
    )
    draft = ProjectDraft(
        project_name="casual_user_v1",
        source_dir=photos,
        preset=TrainingPreset.BALANCED,
    )

    report = ApplicationService(root=tmp_path).readiness(settings, draft)

    assert report.can_prepare
    assert not report.can_run_full
    assert next(item for item in report.items if item.code == "comfy").state == "error"
    assert not list(loras.glob(".mivim-write-check-*"))


def test_readiness_allows_full_run_only_after_live_model_verification(
    tmp_path: Path,
    monkeypatch,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 15)
    loras = tmp_path / "models" / "loras"
    loras.mkdir(parents=True)
    toolkit = tmp_path / "ai-toolkit"
    toolkit.mkdir()
    (toolkit / "run.py").write_text("", encoding="utf-8")
    toolkit_python = toolkit / "venv" / "Scripts" / "python.exe"
    toolkit_python.parent.mkdir(parents=True)
    toolkit_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(
            True,
            "127.0.0.1:8188",
            ("sdxl.safetensors",),
            (),
            None,
        ),
    )
    monkeypatch.setattr(
        app_service,
        "_probe_trainer_runtime",
        lambda _executable: (True, "PyTorch and CUDA verified."),
    )
    base_model = tmp_path / "models" / "checkpoints" / "sdxl.safetensors"
    base_model.parent.mkdir(parents=True, exist_ok=True)
    base_model.write_bytes(b"checkpoint")
    settings = AppSettings(
        comfy_lora_dir=loras,
        comfy_checkpoint="sdxl.safetensors",
        training_base_model=base_model,
        ai_toolkit_dir=toolkit,
        ai_toolkit_python=toolkit_python,
        checkpoints_dir=tmp_path / "outputs" / "checkpoints",
    )
    draft = ProjectDraft(
        project_name="ready_v1",
        source_dir=photos,
        preset=TrainingPreset.QUALITY,
    )

    report = ApplicationService(root=tmp_path).readiness(settings, draft)

    assert report.can_prepare
    assert report.can_run_full
    assert all(item.state != "error" for item in report.items)


def test_desktop_command_uses_real_pipeline_entrypoint_and_preset(
    tmp_path: Path,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 1)
    settings = AppSettings(
        comfy_lora_dir=tmp_path / "loras",
        comfy_checkpoint="sdxl.safetensors",
        datasets_dir=Path("user-data/datasets"),
        checkpoints_dir=Path("user-data/checkpoints"),
        grids_dir=Path("user-data/grids"),
        run_configs_dir=Path("user-data/configs"),
    )
    draft = ProjectDraft(
        project_name="quick_v1",
        source_dir=photos,
        trigger="mivim person",
        preset=TrainingPreset.FAST,
    )

    command = ApplicationService(root=tmp_path).build_command(
        settings,
        draft,
        mode="prepare",
    )

    assert command.arguments[:2] == ("-m", "src.pipeline")
    assert ("--preset", "fast") == (
        command.arguments[
            command.arguments.index("--preset") : command.arguments.index("--preset")
            + 2
        ]
    )
    assert command.arguments[-2:] == ("--skip-train", "--skip-validate")
    assert command.environment["COMFY_SDXL_CKPT"] == "sdxl.safetensors"
    assert command.environment["MIVIM_DESKTOP_RUN"] == "1"
    assert command.expected_result == tmp_path / "user-data" / "datasets" / "quick_v1"


def test_packaged_desktop_reuses_its_executable_as_pipeline_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 1)
    monkeypatch.setenv("MIVIM_PACKAGED_APP", "1")
    draft = ProjectDraft(
        project_name="packaged_v1",
        source_dir=photos,
        preset=TrainingPreset.BALANCED,
    )

    command = ApplicationService(root=tmp_path).build_command(
        AppSettings(datasets_dir=tmp_path / "datasets"),
        draft,
        mode="prepare",
    )

    assert command.arguments[0] == "--pipeline-worker"
    assert "-m" not in command.arguments[:2]


def test_existing_project_requires_explicit_rebuild_before_preparation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 12)
    (tmp_path / "datasets" / "taken_v1").mkdir(parents=True)
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "offline"),
    )
    settings = AppSettings(
        datasets_dir=tmp_path / "datasets",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    draft = ProjectDraft(project_name="taken_v1", source_dir=photos)

    report = ApplicationService(root=tmp_path).readiness(settings, draft)

    collision = next(item for item in report.items if item.code == "collision")
    assert collision.state == "error"
    assert not report.can_prepare


def test_vision_captioning_is_blocked_when_optional_runtime_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 12)
    monkeypatch.setattr(app_service, "_module_available", lambda _name: False)
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "offline"),
    )
    draft = ProjectDraft(
        project_name="vision_v1",
        source_dir=photos,
        use_vision_captions=True,
    )

    report = ApplicationService(root=tmp_path).readiness(
        AppSettings(checkpoints_dir=tmp_path / "checkpoints"),
        draft,
    )

    captions = next(item for item in report.items if item.code == "captions")
    assert captions.state == "error"
    assert not report.can_prepare


def test_discovered_comfy_checkpoint_becomes_the_training_base(
    tmp_path: Path,
) -> None:
    checkpoints = tmp_path / "models" / "checkpoints"
    checkpoints.mkdir(parents=True)
    expected = checkpoints / "people" / "portraitXL.safetensors"
    expected.parent.mkdir()
    expected.write_bytes(b"model")
    model_root = ComfyModelRoot(
        path=tmp_path / "models",
        loras=tmp_path / "models" / "loras",
        checkpoints=checkpoints,
        source="test",
        exists=True,
    )
    discovery = ComfyDesktopDiscovery(
        config_dir=tmp_path,
        legacy_config_dir=None,
        installations=(),
        selected_installation=None,
        model_roots=(model_root,),
        input_dir=None,
        output_dir=None,
        host_candidates=(),
        launcher=ComfyLauncher("", "", "", (), False),
        warnings=(),
    )

    resolved = ApplicationService(root=tmp_path).resolve_checkpoint_path(
        discovery,
        "people/portraitXL.safetensors",
    )

    assert resolved == expected


def test_desktop_command_passes_local_comfy_model_to_trainer(tmp_path: Path) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 1)
    base_model = tmp_path / "sdxl.safetensors"
    base_model.write_bytes(b"model")

    command = ApplicationService(root=tmp_path).build_command(
        AppSettings(
            datasets_dir=tmp_path / "datasets",
            training_base_model=base_model,
        ),
        ProjectDraft(project_name="shared_model_v1", source_dir=photos),
        mode="full",
    )

    position = command.arguments.index("--base-model")
    assert command.arguments[position + 1] == str(base_model)


def test_validation_retry_skips_completed_stages_and_reuses_installed_lora(
    tmp_path: Path,
) -> None:
    settings = AppSettings(grids_dir=tmp_path / "grids")
    draft = ProjectDraft(project_name="retry_v1", trigger="mivim person")

    command = ApplicationService(root=tmp_path).build_validation_command(
        settings,
        draft,
        "retry_v1_000002000.safetensors",
    )

    assert command.mode == "validate"
    assert "--skip-prep" in command.arguments
    assert "--skip-train" in command.arguments
    assert "--src" not in command.arguments
    lora_position = command.arguments.index("--lora")
    assert command.arguments[lora_position + 1] == "retry_v1_000002000.safetensors"
    assert command.expected_result == tmp_path / "grids" / "retry_v1_grid.png"


def test_validation_retry_rejects_a_lora_path_instead_of_a_filename(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Invalid installed LoRA filename"):
        ApplicationService(root=tmp_path).build_validation_command(
            AppSettings(),
            ProjectDraft(project_name="retry_v1"),
            "../outside.safetensors",
        )


def test_log_progress_comes_from_real_pipeline_markers() -> None:
    assert stage_from_log("[prepare] 20 candidate image(s)") == (
        12,
        "Preparing photos",
    )
    assert stage_from_log("[train] Run config: run.yaml") == (
        34,
        "Preparing the trainer",
    )
    assert stage_from_log("[pipeline] Complete.") == (100, "Complete")
    assert stage_from_log("ordinary trainer output") is None
