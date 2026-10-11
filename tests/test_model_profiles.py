from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src import config, pipeline, train, validate
from src.app_models import AppSettings, ProjectDraft, TrainingPreset
from src.app_service import ApplicationService
from src.app_settings import load_settings, save_settings
from src.model_profiles import (
    MODEL_PROFILE_IDS,
    model_profile,
    profile_checkpoint_issue,
    training_profile_issue,
)
from src.toolkit_vpred import TRAINER_SOURCE, vpred_runner


def _trainer_source(tmp_path: Path, target: str) -> Path:
    source = tmp_path / TRAINER_SOURCE
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "class SDTrainer:\n"
        "    def calculate_loss(self, batch, noise, timesteps):\n"
        "        if self.sd.prediction_type == 'v_prediction':\n"
        f"            target = self.sd.noise_scheduler.get_velocity(batch.{target}, noise, timesteps)\n"
        "        else:\n"
        "            target = noise\n"
        "        return target\n",
        encoding="utf-8",
    )
    return source


@pytest.mark.parametrize("profile_id", MODEL_PROFILE_IDS)
def test_profile_round_trip_and_worker_environment(
    tmp_path: Path, profile_id: str
) -> None:
    settings = AppSettings(model_profile=profile_id)
    saved = load_settings(save_settings(settings, tmp_path / "settings.json"))
    assert saved.model_profile == profile_id
    assert saved.environment_overrides()["MIVIM_MODEL_PROFILE"] == profile_id


@pytest.mark.parametrize("profile_id", MODEL_PROFILE_IDS[1:])
def test_noobai_training_and_editable_workflow_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile_id: str
) -> None:
    monkeypatch.setattr(config, "MODEL_PROFILE", profile_id)
    profile = model_profile(profile_id)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (32, 32)).save(dataset / "one.png")
    (dataset / "one.txt").write_text("subject, portrait", encoding="utf-8")
    checkpoint = tmp_path / "renamed.safetensors"
    checkpoint.write_bytes(b"test checkpoint")
    concrete = train.materialize_config(
        config.ROOT / "configs/person_sdxl.yaml",
        run_name="profile_test",
        dataset_dir=dataset,
        output_dir=tmp_path / "output",
        base_model=checkpoint,
        destination=tmp_path / "run.yaml",
        preset="fast",
    )
    data = yaml.safe_load(concrete.read_text(encoding="utf-8"))
    process = data["config"]["process"][0]
    assert process["model"]["is_v_pred"] == profile.v_prediction
    assert process["train"]["noise_scheduler"] == "ddpm"
    assert process["sample"]["sampler"] == (
        "euler" if profile.v_prediction else "euler_a"
    )
    assert process["sample"]["sample_steps"] == profile.steps
    assert process["sample"]["guidance_scale"] == profile.cfg
    assert process["train"]["steps"] == 900
    train.validate_sdxl_process(process, concrete)
    graph = validate._ready_workflow_graph(
        validate.load_workflow(),
        checkpoint=checkpoint.name,
        lora="test.safetensors",
        prompt="subject, portrait",
        strength=0.8,
        seed=42,
    )
    ui = validate._ready_ui_workflow(graph)
    assert validate._ui_workflow_to_prompt(ui) == graph
    assert graph["11"]["inputs"]["zsnr"] == profile.v_prediction
    assert graph["11"]["inputs"]["sampling"] == (
        "v_prediction" if profile.v_prediction else "eps"
    )
    assert graph["3"]["inputs"]["model"] == ["11", 0]
    assert graph["3"]["inputs"]["sampler_name"] == profile.sampler
    assert graph["3"]["inputs"]["cfg"] == process["sample"]["guidance_scale"]
    assert data["meta"]["mivim_model_profile"] == profile_id


def test_profiles_do_not_reuse_each_others_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signatures = set()
    for profile_id in MODEL_PROFILE_IDS:
        monkeypatch.setattr(config, "MODEL_PROFILE", profile_id)
        signatures.add(
            validate.validation_signature(
                lora="test.safetensors",
                strengths=[0, 0.8],
                prompts=["portrait"],
                trigger="subject",
                seed=42,
                lora_content_identity="lora-bytes",
                checkpoint_content_identity="same-renamed-checkpoint-bytes",
            )
        )
    assert len(signatures) == 3


def test_wrong_or_unknown_profile_rejected_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "COMFY_SDXL_CKPT", "NoobAI-XL-Vpred-1.0.safetensors")
    monkeypatch.setattr(config, "MODEL_PROFILE", "sdxl")
    with pytest.raises(ValueError, match="Choose NoobAI-XL"):
        validate.load_workflow()
    monkeypatch.setattr(config, "MODEL_PROFILE", "unavailable")
    with pytest.raises(ValueError, match="Unknown model profile"):
        validate.load_workflow()


def test_checkpoint_variant_hint_ignores_parent_folder_names() -> None:
    for path in (
        "C:/NoobAI-Vpred/renamed.safetensors",
        r"C:\NoobAI-Vpred\renamed.safetensors",
    ):
        assert profile_checkpoint_issue("noobai-xl-epsilon", path) is None
    assert profile_checkpoint_issue(
        "noobai-xl-epsilon", "checkpoints/NoobAI-Vpred.safetensors"
    )


@pytest.mark.parametrize("target", ["tensor", "latents", "unknown"])
def test_trainer_requires_recognized_velocity_target(
    tmp_path: Path, target: str
) -> None:
    _trainer_source(tmp_path, target)
    issue = training_profile_issue("noobai-xl-vpred", tmp_path)
    assert (issue is None) == (target in ("tensor", "latents"))
    assert training_profile_issue("noobai-xl-epsilon", tmp_path) is None


def test_incompatible_vpred_blocks_pipeline_before_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "AI_TOOLKIT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "MODEL_PROFILE", "noobai-xl-vpred")
    monkeypatch.setattr(pipeline, "arm_desktop_owner_guard", lambda: None)
    args = argparse.Namespace(
        model_profile="noobai-xl-vpred", skip_train=False, skip_validate=False
    )
    with pytest.raises(ValueError, match="latent-target support"):
        pipeline.run(args)
    assert not list(tmp_path.iterdir())
    (tmp_path / "run.py").write_text(
        "raise AssertionError('must not launch')", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="latent-target support"):
        train.train("configs/person_sdxl.yaml")


def test_readiness_exposes_profile_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import app_service
    from src.app_service import ReadinessActionKind
    from src.comfy_desktop import ComfyApiStatus
    from tests.test_desktop import _OfflineLocalService

    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *args, **kwargs: ComfyApiStatus(False, None, (), (), "offline"),
    )
    service = _OfflineLocalService(root=config.ROOT)
    report = service.readiness(
        AppSettings(model_profile="noobai-xl-vpred", ai_toolkit_dir=tmp_path),
        ProjectDraft(project_name="check"),
    )
    item = next(item for item in report.items if item.code == "model_profile")
    assert item.state == "error"
    assert item.action is not None
    assert item.action.kind == ReadinessActionKind.SELECT_MODEL_PROFILE
    assert not report.can_run_full


def test_recovery_keeps_profile_and_rejects_changed_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "MODEL_PROFILE", "noobai-xl-vpred")
    monkeypatch.setattr(config, "RUN_CONFIGS_DIR", tmp_path)
    monkeypatch.setattr(config, "GRIDS_DIR", tmp_path / "grids")
    lora = tmp_path / "test.safetensors"
    lora.write_bytes(b"lora")
    pipeline._write_validation_recovery(
        name="test",
        state="validation_pending",
        lora_name=lora.name,
        installed_lora_path=lora,
        trigger="subject",
        preset="balanced",
        attempt_id="e5f4d2ce-0308-4415-a63d-a1189d6f2f32",
        checkpoint_path=None,
        strengths=[0, 0.8],
        seed=42,
        validation_signature="a" * 64,
    )
    service = ApplicationService(root=tmp_path)
    settings = AppSettings(run_configs_dir=tmp_path, grids_dir=tmp_path / "grids")
    recovery = service.read_validation_recovery(settings, "test")
    assert recovery is not None
    assert recovery.model_profile == "noobai-xl-vpred"
    assert "original model profile" in (
        service.validation_recovery_issue(settings, recovery) or ""
    )
    command = service.build_validation_command(
        replace(settings, model_profile=recovery.model_profile),
        ProjectDraft(project_name="test", preset=TrainingPreset.BALANCED),
        lora.name,
    )
    assert command.environment["MIVIM_MODEL_PROFILE"] == recovery.model_profile


@pytest.mark.parametrize("broken", [False, True])
@pytest.mark.parametrize("target", ["tensor", "latents"])
def test_isolated_vpred_launcher_patches_before_trainer_imports(
    tmp_path: Path, broken: bool, target: str
) -> None:
    trainer_source = _trainer_source(tmp_path, target)
    original_source = trainer_source.read_bytes()
    toolkit = tmp_path / "toolkit"
    toolkit.mkdir()
    (toolkit / "__init__.py").write_text("", encoding="utf-8")
    (toolkit / "config_modules.py").write_text(
        "class ModelConfig:\n    def __init__(self, **kw): self.is_v_pred = kw['is_v_pred']\n",
        encoding="utf-8",
    )
    rescale = "False" if broken else "options.get('rescale_betas_zero_snr', False)"
    (toolkit / "sampler.py").write_text(
        "from types import SimpleNamespace\n"
        "def get_sampler(name, kwargs=None, arch='sd'):\n"
        "    options = dict(kwargs or {})\n"
        f"    options['rescale_betas_zero_snr'] = {rescale}\n"
        "    return SimpleNamespace(config=SimpleNamespace(**options), "
        "get_velocity=lambda latents, noise, steps: latents - noise, alphas_cumprod=[0.0])\n",
        encoding="utf-8",
    )
    run_py = tmp_path / "run.py"
    run_py.write_text(
        "from toolkit.sampler import get_sampler\n"
        "from extensions_built_in.sd_trainer.SDTrainer import SDTrainer\n"
        "from types import SimpleNamespace\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "v = get_sampler('euler', {'prediction_type': 'v_prediction'})\n"
        "eps = get_sampler('euler', {'prediction_type': 'epsilon'})\n"
        "trainer = SDTrainer()\n"
        "trainer.sd = SimpleNamespace(prediction_type='v_prediction', noise_scheduler=v)\n"
        "for tensor in (None, 'RGB image'):\n"
        "    batch = SimpleNamespace(tensor=tensor, latents=7.0)\n"
        "    assert trainer.calculate_loss(batch, 2.0, 0) == 5.0\n"
        "    assert batch.tensor is tensor\n"
        "trainer.sd.prediction_type = 'epsilon'\n"
        "assert trainer.calculate_loss(batch, 2.0, 0) == 2.0\n"
        "Path(sys.argv[1]).write_text(json.dumps([v.config.rescale_betas_zero_snr, "
        "eps.config.rescale_betas_zero_snr]))\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.json"
    result = subprocess.run(
        [sys.executable, "-c", vpred_runner(tmp_path), str(run_py), str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if broken:
        assert result.returncode != 0
        assert not output.exists()
        assert "zero-terminal-SNR" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(output.read_text()) == [True, False]
    assert trainer_source.read_bytes() == original_source
    assert not list(trainer_source.parent.glob("__pycache__/SDTrainer*.pyc"))


def test_vpred_adapter_rejects_changed_source_before_trainer_execution(
    tmp_path: Path,
) -> None:
    source = _trainer_source(tmp_path, "tensor")
    runner = vpred_runner(tmp_path)
    source.write_text("raise AssertionError('must never execute changed code')\n")
    # The import hook must reject drift even before torch/schedulers are imported.
    runner = runner.split("from toolkit import sampler", 1)[0]
    runner += "\nimport extensions_built_in.sd_trainer.SDTrainer\n"
    result = subprocess.run(
        [sys.executable, "-c", runner, str(tmp_path / "run.py")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "changed after the V-pred compatibility check" in result.stderr
    assert "AssertionError" not in result.stderr


@pytest.mark.parametrize(
    "replacement",
    [
        "self.sd.other_type == 'v_prediction'",
        "self.sd.prediction_type != 'v_prediction'",
    ],
)
def test_unrecognized_velocity_control_flow_stays_blocked(
    tmp_path: Path, replacement: str
) -> None:
    source = _trainer_source(tmp_path, "latents")
    source.write_text(
        source.read_text().replace(
            "self.sd.prediction_type == 'v_prediction'", replacement
        ),
        encoding="utf-8",
    )
    assert training_profile_issue("noobai-xl-vpred", tmp_path) is not None
    with pytest.raises(ValueError, match="Unrecognized"):
        vpred_runner(tmp_path)


def test_desktop_profile_keyboard_selection_persists_and_invalidates(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt

    from src.desktop import MainWindow
    from tests.test_desktop import _OfflineLocalService

    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=config.ROOT),
    )
    qtbot.addWidget(window)
    window.show()
    combo = window.model_profile_combo
    request = window._readiness_request
    combo.setFocus()
    combo.showPopup()
    qtbot.keyClick(combo, Qt.Key.Key_End)
    qtbot.keyClick(combo, Qt.Key.Key_Return)
    assert combo.currentData() == "noobai-xl-vpred"
    assert window._readiness_request > request
    assert "V-pred checkpoint" in window.model_profile_hint.text()
    window.save_current_settings(silent=True)
    assert load_settings(tmp_path / "settings.json").model_profile == "noobai-xl-vpred"
    window.close()
