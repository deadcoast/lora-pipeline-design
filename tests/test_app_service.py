from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from src import app_service, config, io_utils, prepare_dataset
from src.app_models import AppSettings, ProjectDraft, TrainingPreset
from src.app_service import ApplicationService, inspect_source, stage_from_log
from src.comfy_desktop import (
    ComfyApiStatus,
    ComfyDesktopDiscovery,
    ComfyInstallation,
    ComfyLauncher,
    ComfyModelRoot,
    ComfyRuntimeHint,
)


def _write_images(folder: Path, count: int, *, size: int = 1200) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new(
            "RGB",
            (size + index, size - 100 + index),
            (index * 12, 30, 80),
        ).save(folder / f"{index:03d}.png")


def test_trainer_probe_accepts_blackwell_after_a_real_cuda_kernel() -> None:
    payload = {
        "status": "ready",
        "torch_version": "2.8.0+cu128",
        "cuda_version": "12.8",
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "capability": [12, 0],
        "arch_list": ["sm_80", "sm_90", "sm_120"],
        "probe_value": 11440.0,
    }
    result = subprocess.CompletedProcess(
        ["python", "-c", "probe"],
        0,
        stdout=(
            "trainer startup warning\n"
            f"{app_service._TRAINER_PROBE_PREFIX}{json.dumps(payload)}\n"
        ),
        stderr="",
    )

    ready, detail = app_service._trainer_probe_result(result)

    assert ready
    assert "RTX 5090 (sm_120)" in detail
    assert "real CUDA test kernel" in detail
    assert "PyTorch 2.8.0+cu128 / CUDA 12.8" in detail


def test_trainer_probe_explains_an_unsupported_gpu_architecture() -> None:
    payload = {
        "status": "kernel_failed",
        "torch_version": "2.5.1+cu124",
        "cuda_version": "12.4",
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "capability": [12, 0],
        "arch_list": ["sm_80", "sm_86", "sm_89", "sm_90"],
        "error": "RuntimeError: no kernel image is available for execution",
    }
    result = subprocess.CompletedProcess(
        ["python", "-c", "probe"],
        0,
        stdout=f"{app_service._TRAINER_PROBE_PREFIX}{json.dumps(payload)}\n",
        stderr="",
    )

    ready, detail = app_service._trainer_probe_result(result)

    assert not ready
    assert "uses sm_120" in detail
    assert "sm_80, sm_86, sm_89, sm_90" in detail
    assert "no kernel image" in detail
    assert "compatible with sm_120" in detail


def test_trainer_probe_preserves_older_supported_gpu_architectures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trainer_python = tmp_path / "ai-toolkit" / "venv" / "Scripts" / "python.exe"
    payload = {
        "status": "ready",
        "torch_version": "2.4.0+cu121",
        "cuda_version": "12.1",
        "gpu_name": "NVIDIA GeForce RTX 3080",
        "capability": [8, 6],
        "arch_list": ["sm_80", "sm_86"],
    }
    called: list[list[str]] = []

    def run(arguments, **_kwargs):
        called.append(arguments)
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=f"{app_service._TRAINER_PROBE_PREFIX}{json.dumps(payload)}\n",
            stderr="",
        )

    monkeypatch.setattr(app_service.subprocess, "run", run)

    ready, detail = app_service.probe_trainer_runtime(trainer_python)

    assert ready
    assert called[0][0] == str(trainer_python)
    assert "torch.arange" in called[0][2]
    assert "RTX 3080 (sm_86)" in detail


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


def test_inspect_source_reports_heic_instead_of_silently_omitting_it(
    tmp_path: Path,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 1)
    heic = photos / "phone-photo.heic"
    heic.write_bytes(b"not a decodable HEIC image")

    result = inspect_source(photos)

    assert result.image_count == 2
    assert result.unreadable_count == 1
    issue = next(issue for issue in result.issues if issue.path == heic)
    assert issue.kind == "unreadable"


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
    duplicate = next(issue for issue in result.issues if issue.kind == "duplicate")
    assert duplicate.path == photos / "copy.png"
    assert duplicate.related_path == photos / "000.png"


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
    comfy_item = next(item for item in report.items if item.code == "comfy")
    assert comfy_item.state == "error"
    assert comfy_item.action is not None
    assert comfy_item.action.kind is app_service.ReadinessActionKind.OPEN_COMFY_DESKTOP
    assert comfy_item.action.label == "Open the Comfy Desktop install guide"
    assert not list(loras.glob(".mivim-write-check-*"))


def test_readiness_guides_first_desktop_instance_instead_of_reinstall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 15)
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
    discovery = ComfyDesktopDiscovery(
        config_dir=tmp_path / "Comfy Desktop",
        legacy_config_dir=None,
        installations=(),
        selected_installation=None,
        model_roots=(),
        input_dir=None,
        output_dir=None,
        host_candidates=(),
        launcher=ComfyLauncher(
            "Comfy Desktop",
            "test",
            "",
            (),
            True,
        ),
        warnings=(),
    )

    report = ApplicationService(root=tmp_path).readiness(
        AppSettings(),
        ProjectDraft(
            project_name="first_instance_v1",
            source_dir=photos,
            preset=TrainingPreset.BALANCED,
        ),
        discovery,
    )

    comfy_item = next(item for item in report.items if item.code == "comfy")
    assert "no local ComfyUI instance" in comfy_item.detail
    assert comfy_item.action is not None
    assert comfy_item.action.label == "Open Comfy Desktop to create an instance"


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
        "probe_trainer_runtime",
        lambda _executable: (True, "PyTorch and CUDA verified."),
    )
    base_model = tmp_path / "models" / "checkpoints" / "sdxl.safetensors"
    base_model.parent.mkdir(parents=True, exist_ok=True)
    base_model.write_bytes(b"checkpoint")
    shadowed_dir = tmp_path / "older-models" / "checkpoints"
    shadowed_dir.mkdir(parents=True)
    shadowed_model = shadowed_dir / base_model.name
    shadowed_model.write_bytes(b"different checkpoint with the same name")
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
    discovery = ComfyDesktopDiscovery(
        config_dir=tmp_path,
        legacy_config_dir=None,
        installations=(),
        selected_installation=None,
        model_roots=(
            ComfyModelRoot(
                path=tmp_path / "models",
                loras=loras,
                checkpoints=base_model.parent,
                source="test",
                exists=True,
            ),
        ),
        input_dir=None,
        output_dir=None,
        host_candidates=("127.0.0.1:8188",),
        launcher=ComfyLauncher("", "", "", (), False),
        warnings=(),
        checkpoint_dirs=(base_model.parent, shadowed_dir),
        lora_dirs=(loras,),
    )

    report = ApplicationService(root=tmp_path).readiness(
        settings,
        draft,
        discovery,
    )

    assert not report.can_run_full
    unconfirmed = next(item for item in report.items if item.code == "checkpoint")
    assert unconfirmed.state == "error"
    assert unconfirmed.action is not None
    assert unconfirmed.action.kind is app_service.ReadinessActionKind.CONFIRM_CHECKPOINT
    assert unconfirmed.action.label == "Confirm this is an SDXL base checkpoint"

    settings.confirmed_sdxl_checkpoint = settings.comfy_checkpoint
    report = ApplicationService(root=tmp_path).readiness(
        settings,
        draft,
        discovery,
    )

    assert report.can_prepare
    assert report.can_run_full
    assert all(item.state != "error" for item in report.items)
    model_binding = next(item for item in report.items if item.code == "base_model")
    assert model_binding.state == "warning"
    assert str(base_model) in model_binding.detail
    assert "lower-priority same-named model" in model_binding.detail


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
    strengths_position = command.arguments.index("--strengths")
    assert command.arguments[strengths_position + 1 : strengths_position + 3] == (
        "0.0",
        "0.8",
    )
    assert command.environment["COMFY_SDXL_CKPT"] == "sdxl.safetensors"
    assert command.environment["MIVIM_DESKTOP_RUN"] == "1"
    assert command.expected_result == tmp_path / "user-data" / "datasets" / "quick_v1"


@pytest.mark.parametrize("mode", ["prepare", "full", "validate"])
def test_packaged_desktop_reuses_its_executable_as_pipeline_worker(
    tmp_path: Path,
    monkeypatch,
    mode,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 1)
    monkeypatch.setenv("MIVIM_PACKAGED_APP", "1")
    studio = tmp_path / "Studio package" / "MIVIMLoRAStudio.exe"
    monkeypatch.setattr(app_service.sys, "argv", [str(studio)])
    monkeypatch.setattr(
        app_service.sys, "executable", str(studio.with_name("python.exe"))
    )
    draft = ProjectDraft(
        project_name="packaged_v1",
        source_dir=photos,
        preset=TrainingPreset.BALANCED,
    )

    service = ApplicationService(root=tmp_path)
    settings = AppSettings(datasets_dir=tmp_path / "datasets")
    command = (
        service.build_validation_command(settings, draft, "packaged_v1.safetensors")
        if mode == "validate"
        else service.build_command(settings, draft, mode=mode)
    )

    assert command.program == str(studio)
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


def test_project_collision_covers_every_output_namespace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 12)
    checkpoints = tmp_path / "checkpoints"
    checkpoint = checkpoints / "nested" / "taken_v2_000000250.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (checkpoint.parent / "taken_v20_000000250.safetensors").write_bytes(b"other")
    grids = tmp_path / "grids"
    grid = grids / "taken_v2_grid.png"
    grid.parent.mkdir()
    grid.write_bytes(b"grid")
    workflow = grids / "taken_v2_use_in_comfyui.json"
    workflow.write_text("{}\n", encoding="utf-8")
    work_dir = grids / "taken_v2"
    work_dir.mkdir()
    (work_dir / "cell.png").write_bytes(b"cell")
    configs = tmp_path / "configs"
    configs.mkdir()
    run_config = configs / "taken_v2.yaml"
    run_config.write_text("config: {}\n", encoding="utf-8")
    result_record = configs / "taken_v2-result.json"
    result_record.write_text("{}\n", encoding="utf-8")
    loras = tmp_path / "models" / "loras"
    loras.mkdir(parents=True)
    installed = loras / "taken_v2.safetensors"
    installed.write_bytes(b"lora")
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "offline"),
    )
    settings = AppSettings(
        comfy_lora_dir=loras,
        datasets_dir=tmp_path / "datasets",
        checkpoints_dir=checkpoints,
        grids_dir=grids,
        run_configs_dir=configs,
    )
    draft = ProjectDraft(project_name="taken_v2", source_dir=photos)
    service = ApplicationService(root=tmp_path)

    artifacts = service.project_artifacts(settings, draft.project_name)
    report = service.readiness(settings, draft)
    collision = next(item for item in report.items if item.code == "collision")

    assert artifacts.checkpoints == (checkpoint,)
    assert artifacts.validation_grid == grid
    assert artifacts.validation_workflow == workflow
    assert artifacts.validation_work_dir == work_dir
    assert artifacts.run_config == run_config
    assert artifacts.result_record == result_record
    assert artifacts.installed_loras == (installed,)
    assert collision.state == "error"
    assert "training checkpoint" in collision.detail
    assert "comparison grid" in collision.detail
    assert "ready ComfyUI workflow" in collision.detail
    assert "validation work files" in collision.detail
    assert "trainer run config" in collision.detail
    assert "completed-run record" in collision.detail
    assert "installed ComfyUI LoRA" in collision.detail
    assert not report.can_prepare
    with pytest.raises(ValueError, match="Existing project outputs"):
        service.build_command(settings, draft, mode="prepare")


def test_explicit_rebuild_acknowledges_all_project_output_collisions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 12)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "rebuild_v1.safetensors").write_bytes(b"old")
    grids = tmp_path / "grids"
    grids.mkdir()
    (grids / "rebuild_v1_grid.png").write_bytes(b"old grid")
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "offline"),
    )
    settings = AppSettings(
        datasets_dir=tmp_path / "datasets",
        checkpoints_dir=checkpoints,
        grids_dir=grids,
    )
    draft = ProjectDraft(
        project_name="rebuild_v1",
        source_dir=photos,
        overwrite_existing=True,
    )
    service = ApplicationService(root=tmp_path)

    report = service.readiness(settings, draft)
    command = service.build_command(settings, draft, mode="prepare")
    collision = next(item for item in report.items if item.code == "collision")

    assert collision.state == "warning"
    assert "explicitly enabled" in collision.detail
    assert report.can_prepare
    assert "--overwrite" in command.arguments


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


def test_setup_binds_duplicate_checkpoint_name_to_comfyuis_first_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preferred = tmp_path / "preferred"
    shadowed = tmp_path / "shadowed"
    for directory, content in ((preferred, b"preferred"), (shadowed, b"shadowed")):
        directory.mkdir()
        (directory / "portraitXL.safetensors").write_bytes(content)
    discovery = ComfyDesktopDiscovery(
        config_dir=tmp_path,
        legacy_config_dir=None,
        installations=(),
        selected_installation=None,
        model_roots=(),
        input_dir=None,
        output_dir=None,
        host_candidates=("127.0.0.1:8188",),
        launcher=ComfyLauncher("", "", "", (), False),
        warnings=(),
        checkpoint_dirs=(preferred, shadowed),
    )
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(
            True,
            "127.0.0.1:8188",
            ("portraitXL.safetensors",),
            (),
            None,
        ),
    )

    applied, _context, _status = ApplicationService(root=tmp_path).setup_snapshot(
        AppSettings(
            comfy_checkpoint="portraitXL.safetensors",
            training_base_model=shadowed / "portraitXL.safetensors",
        ),
        discovery,
    )

    assert applied.training_base_model == preferred / "portraitXL.safetensors"


def test_stale_checkpoint_with_empty_live_inventory_opens_comfyui(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(
            True,
            "127.0.0.1:8188",
            (),
            (),
            None,
        ),
    )

    report = ApplicationService(root=tmp_path).readiness(
        AppSettings(
            comfy_host="127.0.0.1:8188",
            comfy_checkpoint="removed.safetensors",
        ),
        ProjectDraft(
            project_name="empty_inventory_v1",
            source_dir=tmp_path / "photos",
        ),
    )
    checkpoint = next(item for item in report.items if item.code == "checkpoint")

    assert checkpoint.state == "error"
    assert checkpoint.action is not None
    assert checkpoint.action.kind is app_service.ReadinessActionKind.OPEN_COMFYUI
    assert checkpoint.action.label == "Open ComfyUI to add an SDXL model"
    assert "does not report any checkpoints" in checkpoint.detail


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


def test_validation_retry_uses_the_checkpoint_saved_with_the_recovery(
    tmp_path: Path,
) -> None:
    settings = AppSettings(comfy_checkpoint="new-selection.safetensors")
    command = ApplicationService(root=tmp_path).build_validation_command(
        settings,
        ProjectDraft(project_name="retry_v1"),
        "retry_v1.safetensors",
        checkpoint="trained-with-this.safetensors",
    )

    assert command.environment["COMFY_SDXL_CKPT"] == "trained-with-this.safetensors"


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
    assert stage_from_log("[validate] cell 3/8 reused strength=0.8 prompt#2") == (
        85,
        "Creating comparison images (3/8)",
    )
    assert stage_from_log(
        "[validate] Ready workflow written: portrait_use_in_comfyui.json"
    ) == (98, "Preparing the ComfyUI handoff")
    assert stage_from_log("ordinary trainer output") is None


def test_prepared_dataset_is_reused_without_reprocessing_or_source_folder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 12, size=1024)
    datasets = tmp_path / "datasets"
    monkeypatch.setattr(config, "DATASETS_DIR", datasets)
    prepare_dataset.prepare(photos, "resume_v1", trigger="mivim person")
    moved_photos = tmp_path / "photos-moved"
    photos.rename(moved_photos)
    draft = ProjectDraft(
        project_name="resume_v1",
        source_dir=photos,
        trigger="mivim person",
    )
    settings = AppSettings(
        datasets_dir=datasets,
        checkpoints_dir=tmp_path / "checkpoints",
    )
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "offline"),
    )
    service = ApplicationService(root=tmp_path)

    report = service.readiness(settings, draft)
    command = service.build_command(settings, draft, mode="full")

    assert report.prepared is not None and report.prepared.ready
    assert not report.can_prepare
    assert next(item for item in report.items if item.code == "collision").state == (
        "ready"
    )
    assert "--skip-prep" in command.arguments
    assert "--src" not in command.arguments


def test_prepared_dataset_reuse_rejects_a_new_unique_source_photo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 2, size=1024)
    datasets = tmp_path / "datasets"
    monkeypatch.setattr(config, "DATASETS_DIR", datasets)
    prepared = prepare_dataset.prepare(
        photos,
        "changed_sources_v1",
        trigger="mivim person",
    )
    Image.new("RGB", (1024, 1024), "purple").save(photos / "new-photo.png")
    draft = ProjectDraft(
        project_name="changed_sources_v1",
        source_dir=photos,
        trigger="mivim person",
    )

    summary = app_service.inspect_prepared_dataset(prepared, draft)

    assert not summary.ready
    assert summary.manifest_error is not None
    assert "source photo set changed" in summary.manifest_error.casefold()


@pytest.mark.parametrize(
    ("prepared_mode", "use_vision_captions"),
    (("template", True), ("blip", False)),
)
def test_prepared_dataset_reuse_rejects_caption_mode_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_mode: str,
    use_vision_captions: bool,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 2, size=1024)
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(
        prepare_dataset,
        "make_caption",
        lambda *_args, **_kwargs: "mivim person, portrait\n",
    )
    prepared = prepare_dataset.prepare(
        photos,
        f"caption_mode_{prepared_mode}",
        trigger="mivim person",
        caption_mode=prepared_mode,
    )

    summary = app_service.inspect_prepared_dataset(
        prepared,
        ProjectDraft(
            project_name=f"caption_mode_{prepared_mode}",
            source_dir=photos,
            trigger="mivim person",
            use_vision_captions=use_vision_captions,
        ),
    )

    assert not summary.ready
    assert summary.manifest_error is not None
    assert "photo-description method changed" in summary.manifest_error


def test_prepared_dataset_allows_caption_edits_but_rejects_image_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 2, size=1024)
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    prepared = prepare_dataset.prepare(
        photos,
        "output_attestation_v1",
        trigger="mivim person",
    )
    draft = ProjectDraft(
        project_name="output_attestation_v1",
        source_dir=photos,
        trigger="mivim person",
    )
    (prepared / "001.txt").write_text(
        "mivim person, edited by the user\n",
        encoding="utf-8",
    )

    assert app_service.inspect_prepared_dataset(prepared, draft).ready

    Image.new("RGB", (1024, 924), "black").save(prepared / "001.png")
    changed = app_service.inspect_prepared_dataset(prepared, draft)

    assert not changed.ready
    assert changed.manifest_error is not None
    assert "001.png changed after preparation" in changed.manifest_error


@pytest.mark.parametrize("filename", ["extra.png", "003.jpg", "holiday.WEBP"])
def test_prepared_dataset_rejects_an_added_complete_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 2, size=1024)
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    prepared = prepare_dataset.prepare(photos, "added_pair_v1", trigger="mivim person")
    draft = ProjectDraft(
        project_name="added_pair_v1", source_dir=photos, trigger="mivim person"
    )
    assert app_service.inspect_prepared_dataset(prepared, draft).ready
    extra = prepared / filename
    Image.new("RGB", (1024, 924), "blue").save(extra)
    extra.with_suffix(".txt").write_text(
        "mivim person, added photo\n", encoding="utf-8"
    )

    summary = app_service.inspect_prepared_dataset(prepared, draft)

    assert summary.inventory.structure_ready
    assert not summary.ready
    assert summary.manifest_error is not None
    assert filename in summary.manifest_error
    assert "not created by this preparation run" in summary.manifest_error


def test_prepared_dataset_rejects_a_deleted_complete_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 2, size=1024)
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    prepared = prepare_dataset.prepare(
        photos,
        "deleted_pair_v1",
        trigger="mivim person",
    )
    (prepared / "001.png").unlink()
    (prepared / "001.txt").unlink()

    summary = app_service.inspect_prepared_dataset(
        prepared,
        ProjectDraft(
            project_name="deleted_pair_v1",
            source_dir=photos,
            trigger="mivim person",
        ),
    )

    assert not summary.ready
    assert summary.manifest_error is not None
    assert "001.png is missing" in summary.manifest_error


def test_prepare_result_record_is_resume_metadata_not_an_output_collision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 12, size=1024)
    datasets = tmp_path / "datasets"
    configs = tmp_path / "configs"
    monkeypatch.setattr(config, "DATASETS_DIR", datasets)
    prepared = prepare_dataset.prepare(
        photos,
        "prepared_v2",
        trigger="mivim person",
    )
    configs.mkdir()
    (configs / "prepared_v2-result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_name": "prepared_v2",
                "mode": "prepare",
                "result_path": str(prepared.resolve()),
                "trigger": "mivim person",
                "preset": "balanced",
                "checkpoint": "",
                "lora_name": "",
                "completed_at": "2026-07-24T12:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    photos.rename(tmp_path / "photos-moved")
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "offline"),
    )
    settings = AppSettings(
        datasets_dir=datasets,
        checkpoints_dir=tmp_path / "checkpoints",
        run_configs_dir=configs,
    )
    draft = ProjectDraft(
        project_name="prepared_v2",
        source_dir=photos,
        trigger="mivim person",
    )
    service = ApplicationService(root=tmp_path)

    report = service.readiness(settings, draft)
    command = service.build_command(settings, draft, mode="full")
    collision = next(item for item in report.items if item.code == "collision")

    assert collision.state == "ready"
    assert "--skip-prep" in command.arguments
    assert "--src" not in command.arguments


def test_validation_recovery_requires_the_exact_installed_lora_and_grid(
    tmp_path: Path,
) -> None:
    lora_dir = tmp_path / "ComfyUI" / "models" / "loras"
    lora_dir.mkdir(parents=True)
    installed = lora_dir / "recovery_v1.safetensors"
    installed.write_bytes(b"trained weights")
    grids = tmp_path / "grids"
    configs = tmp_path / "configs"
    configs.mkdir()
    recovery_path = configs / "recovery_v1-recovery.json"
    recovery_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_name": "recovery_v1",
                "state": "validation_pending",
                "lora_name": installed.name,
                "installed_lora_path": str(installed.resolve()),
                "expected_grid_path": str((grids / "recovery_v1_grid.png").resolve()),
                "trigger": "ohwx person",
                "preset": "balanced",
                "checkpoint": "portraitXL.safetensors",
                "updated_at": "2026-07-24T12:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    settings = AppSettings(
        comfy_lora_dir=lora_dir,
        grids_dir=grids,
        run_configs_dir=configs,
    )
    service = ApplicationService(root=tmp_path)

    recovery = service.read_validation_recovery(settings, "recovery_v1")

    assert recovery is not None
    assert recovery.state == "validation_pending"
    assert recovery.lora_name == installed.name
    assert recovery.installed_lora_path == installed
    assert recovery.expected_grid_path == grids / "recovery_v1_grid.png"
    installed.unlink()
    stale = service.read_validation_recovery(settings, "recovery_v1")
    assert stale is not None
    assert "older version" in (service.validation_recovery_issue(settings, stale) or "")


def test_validation_recovery_v2_rejects_changed_installation_or_model_bytes(
    tmp_path: Path,
) -> None:
    lora_dir = tmp_path / "ComfyUI" / "models" / "loras"
    lora_dir.mkdir(parents=True)
    installed = lora_dir / "recovery_v2.safetensors"
    installed.write_bytes(b"trained weights")
    checkpoint = tmp_path / "ComfyUI" / "models" / "checkpoints" / "sdxl.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"original checkpoint")
    grids = tmp_path / "grids"
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "recovery_v2-recovery.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "project_name": "recovery_v2",
                "state": "validation_pending",
                "lora_name": installed.name,
                "installed_lora_path": str(installed.resolve()),
                "expected_grid_path": str((grids / "recovery_v2_grid.png").resolve()),
                "trigger": "ohwx person",
                "preset": "balanced",
                "checkpoint": "sdxl.safetensors",
                "updated_at": "2026-07-24T12:00:00+00:00",
                "attempt_id": "448be6d3-e39d-40a8-bd31-4767adaec0aa",
                "comfy_installation_id": "desktop-a",
                "comfy_host": "127.0.0.1:8188",
                "lora_fingerprint": io_utils.artifact_fingerprint(installed),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_fingerprint": io_utils.artifact_fingerprint(checkpoint),
                "validation_signature": "a" * 64,
                "strengths": [0.0, 0.8],
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )
    settings = AppSettings(
        comfy_host="127.0.0.1:8188",
        comfy_installation_id="desktop-a",
        comfy_lora_dir=lora_dir,
        comfy_checkpoint="sdxl.safetensors",
        training_base_model=checkpoint,
        grids_dir=grids,
        run_configs_dir=configs,
    )
    service = ApplicationService(root=tmp_path)

    recovery = service.read_validation_recovery(settings, "recovery_v2")

    assert recovery is not None
    assert service.validation_recovery_issue(settings, recovery) is None
    changed_port = replace(settings, comfy_host="127.0.0.1:8299")
    assert service.validation_recovery_issue(changed_port, recovery) is None
    changed_installation = replace(settings, comfy_installation_id="desktop-b")
    assert "different Comfy Desktop installation" in (
        service.validation_recovery_issue(changed_installation, recovery) or ""
    )
    installed.write_bytes(b"substituted weights")
    assert "changed after training" in (
        service.validation_recovery_issue(settings, recovery) or ""
    )


def test_prepared_dataset_requires_the_identity_phrase_in_every_caption(
    tmp_path: Path,
    monkeypatch,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 2, size=1024)
    datasets = tmp_path / "datasets"
    monkeypatch.setattr(config, "DATASETS_DIR", datasets)
    prepared_folder = prepare_dataset.prepare(
        photos,
        "caption_guard_v1",
        trigger="ohwx person",
    )
    (prepared_folder / "002.txt").write_text(
        "smiling in a garden\n",
        encoding="utf-8",
    )
    draft = ProjectDraft(
        project_name="caption_guard_v1",
        source_dir=photos,
        trigger="ohwx person",
    )
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "offline"),
    )
    service = ApplicationService(root=tmp_path)
    settings = AppSettings(
        datasets_dir=datasets,
        checkpoints_dir=tmp_path / "checkpoints",
    )

    summary = app_service.inspect_prepared_dataset(prepared_folder, draft)
    report = service.readiness(settings, draft)
    collision = next(item for item in report.items if item.code == "collision")

    assert summary.missing_trigger_count == 1
    assert not summary.ready
    assert collision.state == "error"
    assert "without the identity phrase" in collision.detail
    assert not report.can_run_full


def test_prepared_dataset_names_every_structural_pair_problem(
    tmp_path: Path,
    monkeypatch,
) -> None:
    photos = tmp_path / "photos"
    _write_images(photos, 2, size=1024)
    datasets = tmp_path / "datasets"
    monkeypatch.setattr(config, "DATASETS_DIR", datasets)
    prepared_folder = prepare_dataset.prepare(
        photos,
        "pair_guard_v1",
        trigger="ohwx person",
    )
    with Image.open(prepared_folder / "001.png") as image:
        image.save(prepared_folder / "001.jpg")
    Image.new("RGB", (64, 64), "blue").save(prepared_folder / "orphan.webp")
    (prepared_folder / "notes.txt").write_text(
        "ohwx person, unexpected caption\n",
        encoding="utf-8",
    )
    Image.new("RGB", (64, 64), "green").save(prepared_folder / "animated.gif")
    draft = ProjectDraft(
        project_name="pair_guard_v1",
        source_dir=photos,
        trigger="ohwx person",
    )
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "offline"),
    )
    settings = AppSettings(
        datasets_dir=datasets,
        checkpoints_dir=tmp_path / "checkpoints",
    )

    summary = app_service.inspect_prepared_dataset(prepared_folder, draft)
    report = ApplicationService(root=tmp_path).readiness(settings, draft)
    collision = next(item for item in report.items if item.code == "collision")

    assert not summary.ready
    for filename in ("001.jpg", "001.png", "orphan.webp", "notes.txt", "animated.gif"):
        assert filename in summary.problem_detail
        assert filename in collision.detail
    assert collision.action is not None
    assert collision.action.kind is app_service.ReadinessActionKind.REVIEW_PREPARED
    assert collision.action.label == "Review prepared files"
    assert not report.can_run_full


def test_caption_trigger_matching_is_case_insensitive_but_word_bounded() -> None:
    assert app_service.caption_contains_trigger(
        "OHWX   PERSON, smiling in a garden",
        "ohwx person",
    )
    assert not app_service.caption_contains_trigger(
        "an ohwx personification portrait",
        "ohwx person",
    )


def test_setup_uses_a_local_checkpoint_when_the_offline_default_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoints = tmp_path / "models" / "checkpoints"
    checkpoints.mkdir(parents=True)
    portrait = checkpoints / "portraitXL.safetensors"
    portrait.write_bytes(b"model")
    root = ComfyModelRoot(
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
        model_roots=(root,),
        input_dir=None,
        output_dir=None,
        host_candidates=("127.0.0.1:8188",),
        launcher=ComfyLauncher("", "", "", (), False),
        warnings=(),
    )
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "offline"),
    )

    settings, _context, _status = ApplicationService(root=tmp_path).setup_snapshot(
        AppSettings(),
        discovery,
    )

    assert settings.comfy_checkpoint == "portraitXL.safetensors"
    assert settings.training_base_model == portrait


def test_temporary_discovery_failure_preserves_known_installation_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "known-models" / "portraitXL.safetensors"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"model")
    discovery = ComfyDesktopDiscovery(
        config_dir=None,
        legacy_config_dir=None,
        installations=(),
        selected_installation=None,
        model_roots=(),
        input_dir=None,
        output_dir=None,
        host_candidates=(),
        launcher=ComfyLauncher("", "", "", (), False),
        warnings=("Desktop data was temporarily unavailable.",),
    )
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "offline"),
    )

    settings, _context, _status = ApplicationService(root=tmp_path).setup_snapshot(
        AppSettings(
            comfy_installation_id="known-desktop",
            training_base_model=checkpoint,
        ),
        discovery,
    )

    assert settings.comfy_installation_id == "known-desktop"
    assert settings.training_base_model == checkpoint


def test_setup_preserves_an_unavailable_instance_id_without_substitution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    current = ComfyInstallation(
        id="current",
        name="Current Desktop",
        source_id="standalone",
        install_path=tmp_path / "current",
        comfyui_path=tmp_path / "current" / "ComfyUI",
        version=None,
        status="installed",
        last_launched_at=1,
    )
    discovery = ComfyDesktopDiscovery(
        config_dir=tmp_path,
        legacy_config_dir=None,
        installations=(current,),
        selected_installation=current,
        model_roots=(),
        input_dir=None,
        output_dir=None,
        host_candidates=("127.0.0.1:8188",),
        launcher=ComfyLauncher("", "", "", (), False),
        warnings=(),
    )
    probed: list[tuple[str, ...]] = []

    def probe(hosts, **_kwargs):
        probed.append(tuple(hosts))
        return ComfyApiStatus(False, None, (), (), "offline")

    monkeypatch.setattr(app_service, "probe_comfy_api", probe)

    settings, context, status = ApplicationService(root=tmp_path).setup_snapshot(
        AppSettings(comfy_installation_id="original"),
        discovery,
    )

    assert settings.comfy_installation_id == "original"
    assert context.selected_installation is None
    assert context.host_candidates == ()
    assert probed == [()]
    assert not status.reachable


def test_readiness_never_probes_saved_host_for_duplicate_desktop_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installations = tuple(
        ComfyInstallation(
            id=installation_id,
            name="Same Desktop",
            source_id="standalone",
            install_path=tmp_path / installation_id,
            comfyui_path=tmp_path / installation_id / "ComfyUI",
            version=None,
            status="installed",
            last_launched_at=index,
        )
        for index, installation_id in enumerate(("a", "b"), start=1)
    )
    discovery = ComfyDesktopDiscovery(
        config_dir=tmp_path,
        legacy_config_dir=None,
        installations=installations,
        selected_installation=installations[0],
        model_roots=(),
        input_dir=None,
        output_dir=None,
        host_candidates=("127.0.0.1:8201", "127.0.0.1:8202"),
        launcher=ComfyLauncher("", "", "", (), False),
        warnings=(),
        runtime_hints=(
            ComfyRuntimeHint("127.0.0.1:8201", "Same Desktop", 101, 1000),
            ComfyRuntimeHint("127.0.0.1:8202", "Same Desktop", 102, 2000),
        ),
    )
    probed: list[tuple[str, ...]] = []

    def probe(hosts, **_kwargs):
        candidates = tuple(hosts)
        probed.append(candidates)
        return ComfyApiStatus(
            bool(candidates),
            candidates[0] if candidates else None,
            (),
            (),
            None if candidates else "ambiguous",
        )

    monkeypatch.setattr(app_service, "probe_comfy_api", probe)

    report = ApplicationService(root=tmp_path).readiness(
        AppSettings(
            comfy_installation_id="a",
            comfy_host="127.0.0.1:8202",
        ),
        ProjectDraft(
            project_name="duplicate_desktop_v1",
            source_dir=tmp_path / "photos",
        ),
        discovery,
    )
    connection = next(item for item in report.items if item.code == "connection")

    assert probed == [()]
    assert not report.comfy_status.reachable
    assert connection.state == "error"
    assert "unique name" in connection.detail
    assert connection.action is not None
    assert connection.action.kind is app_service.ReadinessActionKind.OPEN_COMFY_DESKTOP
    assert not report.can_run_full


def test_setup_never_borrows_python_from_a_different_selected_toolkit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    selected = tmp_path / "selected-toolkit"
    selected.mkdir()
    (selected / "run.py").write_text("", encoding="utf-8")
    other = tmp_path / "other-toolkit"
    other_python = other / "venv" / "Scripts" / "python.exe"
    other_python.parent.mkdir(parents=True)
    (other / "run.py").write_text("", encoding="utf-8")
    other_python.write_text("", encoding="utf-8")
    monkeypatch.setenv("AI_TOOLKIT_DIR", str(other))
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "offline"),
    )

    settings, _context, _status = ApplicationService(root=tmp_path).setup_snapshot(
        AppSettings(ai_toolkit_dir=selected),
    )

    assert settings.ai_toolkit_dir == selected
    assert settings.ai_toolkit_python is None


def test_setup_binds_selected_desktop_runtime_checkpoint_and_lora_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_a = tmp_path / "models-a"
    model_b = tmp_path / "models-b"
    for root in (model_a, model_b):
        (root / "checkpoints").mkdir(parents=True)
        (root / "loras").mkdir()
    checkpoint_b = model_b / "checkpoints" / "portraitBXL.safetensors"
    checkpoint_b.write_bytes(b"model")
    install_a = ComfyInstallation(
        id="a",
        name="Portrait A",
        source_id="standalone",
        install_path=tmp_path / "install-a",
        comfyui_path=tmp_path / "install-a" / "ComfyUI",
        version=None,
        status="installed",
        last_launched_at=1,
        use_shared_models=False,
        use_shared_input_output=False,
        model_dirs=(model_a,),
    )
    install_b = ComfyInstallation(
        id="b",
        name="Portrait B",
        source_id="standalone",
        install_path=tmp_path / "install-b",
        comfyui_path=tmp_path / "install-b" / "ComfyUI",
        version=None,
        status="installed",
        last_launched_at=2,
        use_shared_models=False,
        use_shared_input_output=False,
        model_dirs=(model_b,),
        model_dirs_primary=model_b,
    )
    generated = tmp_path / "instance-model-paths" / "b.yaml"
    generated.parent.mkdir(parents=True)
    generated.write_text(
        f"private:\n  base_path: {model_b.as_posix()}\n"
        "  is_default: true\n"
        "  checkpoints: checkpoints\n"
        "  loras: loras\n",
        encoding="utf-8",
    )
    future_timestamp = 9_999_999_999_999
    discovery = ComfyDesktopDiscovery(
        config_dir=tmp_path,
        legacy_config_dir=None,
        installations=(install_a, install_b),
        selected_installation=install_a,
        model_roots=(),
        input_dir=None,
        output_dir=None,
        host_candidates=("127.0.0.1:8201", "127.0.0.1:8202"),
        launcher=ComfyLauncher("", "", "", (), False),
        warnings=(),
        runtime_hints=(
            ComfyRuntimeHint(
                "127.0.0.1:8201",
                "Portrait A",
                101,
                future_timestamp,
            ),
            ComfyRuntimeHint(
                "127.0.0.1:8202",
                "Portrait B",
                102,
                future_timestamp,
            ),
        ),
    )
    probed: list[tuple[str, ...]] = []

    def probe(hosts, **_kwargs):
        probed.append(tuple(hosts))
        return ComfyApiStatus(
            True,
            hosts[0],
            ("portraitBXL.safetensors",),
            (),
            None,
        )

    monkeypatch.setattr(app_service, "probe_comfy_api", probe)

    settings, context, status = ApplicationService(root=tmp_path).setup_snapshot(
        AppSettings(
            comfy_installation_id="b",
            comfy_checkpoint="portraitBXL.safetensors",
        ),
        discovery,
    )

    assert probed[0] == ("127.0.0.1:8202",)
    assert status.host == "127.0.0.1:8202"
    assert context.selected_installation == install_b
    assert context.model_roots[0].path == model_b
    assert settings.comfy_lora_dir == model_b / "loras"
    assert settings.training_base_model == checkpoint_b


def test_setup_rejects_same_named_checkpoint_outside_selected_instance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_root = tmp_path / "selected-models"
    (model_root / "checkpoints").mkdir(parents=True)
    (model_root / "loras").mkdir()
    outside = tmp_path / "other-instance" / "portraitXL.safetensors"
    outside.parent.mkdir()
    outside.write_bytes(b"other")
    install = ComfyInstallation(
        id="selected",
        name="Selected",
        source_id="standalone",
        install_path=tmp_path / "install",
        comfyui_path=tmp_path / "install" / "ComfyUI",
        version=None,
        status="installed",
        last_launched_at=1,
        use_shared_models=False,
        use_shared_input_output=False,
        model_dirs=(model_root,),
    )
    root = ComfyModelRoot(
        path=model_root,
        loras=model_root / "loras",
        checkpoints=model_root / "checkpoints",
        source="Selected",
        exists=True,
    )
    discovery = ComfyDesktopDiscovery(
        config_dir=tmp_path,
        legacy_config_dir=None,
        installations=(install,),
        selected_installation=install,
        model_roots=(root,),
        input_dir=None,
        output_dir=None,
        host_candidates=(),
        launcher=ComfyLauncher("", "", "", (), False),
        warnings=(),
        runtime_hints=(),
    )
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda hosts, **_kwargs: ComfyApiStatus(
            False,
            None,
            (),
            (),
            f"offline: {tuple(hosts)}",
        ),
    )

    settings, _context, _status = ApplicationService(root=tmp_path).setup_snapshot(
        AppSettings(
            comfy_installation_id="selected",
            comfy_checkpoint="portraitXL.safetensors",
            training_base_model=outside,
        ),
        discovery,
    )

    assert settings.training_base_model is None


def test_setup_rebinds_lora_folder_for_one_selected_instance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_root = tmp_path / "selected-models"
    (model_root / "checkpoints").mkdir(parents=True)
    selected_loras = model_root / "loras"
    selected_loras.mkdir()
    other_loras = tmp_path / "other-models" / "loras"
    other_loras.mkdir(parents=True)
    install = ComfyInstallation(
        id="selected",
        name="Selected",
        source_id="standalone",
        install_path=tmp_path / "install",
        comfyui_path=tmp_path / "install" / "ComfyUI",
        version=None,
        status="installed",
        last_launched_at=1,
        use_shared_models=False,
        use_shared_input_output=False,
        model_dirs=(model_root,),
    )
    discovery = ComfyDesktopDiscovery(
        config_dir=tmp_path,
        legacy_config_dir=None,
        installations=(install,),
        selected_installation=install,
        model_roots=(
            ComfyModelRoot(
                path=model_root,
                loras=selected_loras,
                checkpoints=model_root / "checkpoints",
                source="Selected",
                exists=True,
            ),
        ),
        input_dir=None,
        output_dir=None,
        host_candidates=(),
        launcher=ComfyLauncher("", "", "", (), False),
        warnings=(),
    )
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(
            False,
            None,
            (),
            (),
            "offline",
        ),
    )

    settings, _context, _status = ApplicationService(root=tmp_path).setup_snapshot(
        AppSettings(
            comfy_installation_id="selected",
            comfy_lora_dir=other_loras,
        ),
        discovery,
    )

    assert settings.comfy_lora_dir == selected_loras
