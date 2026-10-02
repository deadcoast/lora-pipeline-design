from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src import config, pipeline, validate
from src.app_models import AppSettings, RunResult, TrainingPreset
from src.app_service import ApplicationService
from src.app_settings import load_settings, save_settings
from src.io_utils import artifact_fingerprint
from src.training_checkpoints import discover_training_checkpoints


def write_checkpoint(path: Path, step: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps(
        {"__metadata__": {"training_info": json.dumps({"step": step})}}
    ).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"test tensor bytes")
    return path


def selection_project(tmp_path: Path) -> tuple[AppSettings, RunResult]:
    weights = tmp_path / "checkpoints" / "person"
    write_checkpoint(weights / "person_000000250.safetensors", 250)
    final = write_checkpoint(weights / "person.safetensors", 500)
    loras = tmp_path / "loras"
    loras.mkdir()
    installed = loras / final.name
    installed.write_bytes(final.read_bytes())
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"base weights")
    grid = tmp_path / "grids" / "person_grid.png"
    grid.parent.mkdir()
    Image.new("RGB", (16, 16)).save(grid)
    settings = AppSettings(
        checkpoints_dir=weights.parent,
        comfy_lora_dir=loras,
        comfy_installation_id="desktop-a",
        training_base_model=base,
        grids_dir=grid.parent,
        run_configs_dir=tmp_path / "configs",
    )
    result = RunResult(
        project_name="person",
        mode="full",
        result_path=grid,
        completed_at="now",
        trigger="ohwx person",
        preset=TrainingPreset.BALANCED,
        checkpoint=settings.comfy_checkpoint,
        comfy_installation_id="desktop-a",
        installed_lora_path=installed,
        checkpoint_path=base,
        checkpoint_fingerprint=artifact_fingerprint(base),
    )
    return settings, result


def test_discovery_matches_metadata_and_prompt_indexes(tmp_path: Path) -> None:
    settings, _ = selection_project(tmp_path)
    folder = settings.checkpoints_dir / "person"
    write_checkpoint(folder / "person_000000750.safetensors", 250)
    (folder / "person_000001000.safetensors").write_bytes(b"broken")
    write_checkpoint(folder / "other_000000250.safetensors", 250)
    samples = folder / "samples"
    samples.mkdir()
    for name in ("100__000000250_3.jpg", "100__000000500_0.jpg"):
        Image.new("RGB", (16, 16)).save(samples / name)
    choices = discover_training_checkpoints(settings.checkpoints_dir, "person")
    assert [choice.step for choice in choices] == [250, 500]
    assert [choice.final for choice in choices] == [False, True]
    assert choices[0].previews == ((3, samples / "100__000000250_3.jpg"),)
    assert choices[1].previews == ((0, samples / "100__000000500_0.jpg"),)


@pytest.mark.parametrize("failure", [None, "changed", "interrupted"])
def test_selected_weights_use_separate_results_and_recover_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    settings, result = selection_project(tmp_path)
    service = ApplicationService(root=Path(__file__).resolve().parents[1])
    assert settings.comfy_lora_dir is not None
    choice = discover_training_checkpoints(settings.checkpoints_dir, "person")[0]
    draft, command = service.build_checkpoint_validation_command(
        settings, result, choice
    )
    assert draft.project_name != result.project_name
    assert "--skip-prep" in command.arguments and "--skip-train" in command.arguments
    assert "--checkpoint-choice" in command.arguments
    original_grid = result.result_path.read_bytes()
    original_final = (settings.comfy_lora_dir / "person.safetensors").read_bytes()  # type: ignore[operator]
    for key in ("CHECKPOINTS_DIR", "COMFY_LORA_DIR", "RUN_CONFIGS_DIR", "GRIDS_DIR"):
        monkeypatch.setattr(config, key, Path(command.environment[key]))
    monkeypatch.setattr(config, "COMFY_INSTALLATION_ID", "desktop-a")
    monkeypatch.setattr(pipeline, "check_orphaned_comfy_prompts", lambda: None)
    monkeypatch.setattr(
        pipeline.trainer, "train", lambda *_a, **_k: pytest.fail("Retrained")
    )
    monkeypatch.setattr(
        pipeline.prep, "prepare", lambda **_k: pytest.fail("Prepared again")
    )
    graphs: list[dict[str, Any]] = []

    class Client:
        def is_up(self) -> bool:
            return True

        def run_and_save(
            self, graph: dict[str, Any], out_dir: Path, prefix: str = ""
        ) -> list[Path]:
            if failure == "interrupted":
                raise RuntimeError("GPU interrupted")
            graphs.append(graph)
            out_dir.mkdir(parents=True, exist_ok=True)
            image = out_dir / f"{prefix}output.png"
            Image.new("RGB", (32, 32), "teal").save(image)
            return [image]

    monkeypatch.setattr(validate, "ComfyClient", Client)
    if failure == "changed":
        choice.path.write_bytes(b"substituted weights after choosing")
    if failure is None:
        pipeline.main(command.arguments[2:])
        outcome = service.read_pipeline_outcome(settings, draft.project_name)
        assert outcome is not None and outcome.training_project_name == "person"
        assert outcome.lora_fingerprint == choice.fingerprint
        assert outcome.installed_lora_path is not None
        assert outcome.installed_lora_path.read_bytes() == choice.path.read_bytes()
        assert outcome.result_path != result.result_path
        assert len(graphs) == 16
        assert outcome.workflow_path is not None
        workflow = json.loads(outcome.workflow_path.read_text())
        lora = next(node for node in workflow["nodes"] if node["type"] == "LoraLoader")
        assert lora["widgets_values"][0] == f"{draft.project_name}.safetensors"
        saved = replace(
            result, training_project_name="person", project_name=draft.project_name
        )
        settings_path = tmp_path / "settings.json"
        save_settings(replace(settings, last_result=saved), settings_path)
        assert load_settings(settings_path).last_result == saved
    else:
        with pytest.raises(SystemExit):
            pipeline.main(command.arguments[2:])
        assert service.read_pipeline_outcome(settings, draft.project_name) is None
    recovery = service.read_validation_recovery(settings, draft.project_name)
    if failure == "changed":
        assert recovery is None
        assert not (
            settings.comfy_lora_dir / f"{draft.project_name}.safetensors"
        ).exists()  # type: ignore[operator]
    else:
        assert recovery is not None and recovery.training_project_name == "person"
        assert recovery.state == (
            "complete" if failure is None else "validation_pending"
        )
    assert result.result_path.read_bytes() == original_grid
    assert (
        settings.comfy_lora_dir / "person.safetensors"
    ).read_bytes() == original_final  # type: ignore[operator]


def test_selection_rejects_stale_review_and_wrong_project(tmp_path: Path) -> None:
    settings, result = selection_project(tmp_path)
    service = ApplicationService(root=tmp_path)
    choice = discover_training_checkpoints(settings.checkpoints_dir, "person")[0]
    with pytest.raises(ValueError, match="does not belong"):
        service.build_checkpoint_validation_command(
            settings, replace(result, project_name="other"), choice
        )
    choice.path.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="changed after review"):
        service.build_checkpoint_validation_command(settings, result, choice)
