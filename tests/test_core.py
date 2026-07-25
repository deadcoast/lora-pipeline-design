from __future__ import annotations

import argparse
import io
import json
import os
import sys
from email.message import Message
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.error import HTTPError

import pytest
import yaml
from PIL import Image

from src import caption as caption_module
from src import config, io_utils, prepare_dataset
from src import doctor as doctor_module
from src import pipeline as pipeline_module
from src import train as train_module
from src import validate as validate_module
from src.comfy_client import ComfyClient

TEST_LORA_IDENTITY = "sha256:test-lora-a"
TEST_CHECKPOINT_IDENTITY = "sha256:test-checkpoint-a"


def _pipeline_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "name": "subject_v1",
        "src": None,
        "config": "configs/person_sdxl.yaml",
        "trigger": "ohwx person",
        "size": 1024,
        "strengths": [0.6, 0.8],
        "seed": 42,
        "lora": None,
        "vision": False,
        "overwrite": False,
        "skip_prep": True,
        "skip_train": False,
        "skip_validate": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_doctor_uses_the_same_real_trainer_runtime_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolkit = tmp_path / "ai-toolkit"
    toolkit.mkdir()
    (toolkit / "run.py").write_text("", encoding="utf-8")
    python = toolkit / "venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(doctor_module.config, "AI_TOOLKIT_DIR", str(toolkit))
    monkeypatch.setattr(doctor_module.config, "AI_TOOLKIT_PYTHON", str(python))
    called: list[Path] = []

    def probe(executable: Path) -> tuple[bool, str]:
        called.append(executable)
        return True, "GPU kernel completed."

    monkeypatch.setattr(doctor_module, "probe_trainer_runtime", probe)
    results: list[doctor_module.CheckResult] = []

    doctor_module.check_ai_toolkit(results)

    assert called == [python]
    assert any("Trainer CUDA verified" in message for _ok, _tag, message in results)


def test_atomic_write_preserves_destination_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"original")
    monkeypatch.setattr(
        io_utils.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        io_utils.atomic_write_bytes(destination, b"replacement")

    assert destination.read_bytes() == b"original"
    assert list(tmp_path.glob(".artifact.bin.*.tmp")) == []


def test_atomic_copy_preserves_destination_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "destination.safetensors"
    source.write_bytes(b"new weights")
    destination.write_bytes(b"old weights")
    monkeypatch.setattr(
        io_utils.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        io_utils.atomic_copy(source, destination)

    assert destination.read_bytes() == b"old weights"
    assert list(tmp_path.glob(".destination.safetensors.*.tmp")) == []


def test_materialize_config_wires_run_paths(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (32, 32), "white").save(dataset / "001.png")
    (dataset / "001.txt").write_text("ohwx person", encoding="utf-8")
    checkpoints = tmp_path / "checkpoints"
    destination = tmp_path / "configs" / "custom.yaml"
    base_model = tmp_path / "models" / "portraitXL.safetensors"
    base_model.parent.mkdir()
    base_model.write_bytes(b"model")

    generated = train_module.materialize_config(
        config.ROOT / "configs" / "person_sdxl.yaml",
        run_name="custom",
        trigger="custom person",
        dataset_dir=dataset,
        output_dir=checkpoints,
        destination=destination,
        base_model=base_model,
    )

    loaded = yaml.safe_load(generated.read_text(encoding="utf-8"))
    job = loaded["config"]
    process = job["process"][0]
    assert job["name"] == "custom"
    assert loaded["meta"]["name"] == "custom"
    assert Path(process["training_folder"]) == checkpoints.resolve()
    assert Path(process["datasets"][0]["folder_path"]) == dataset.resolve()
    assert process["train"]["noise_scheduler"] == "ddpm"
    assert process["sample"]["sampler"] == "ddpm"
    assert process["model"]["arch"] == "sdxl"
    assert Path(process["model"]["name_or_path"]) == base_model.resolve()
    assert process["trigger_word"] == "custom person"
    assert all(
        sample["prompt"].startswith("custom person,")
        for sample in process["sample"]["samples"]
    )


@pytest.mark.parametrize(
    ("preset", "rank", "steps", "resolution", "sample_size"),
    [
        ("fast", 8, 900, [512, 768], 768),
        ("balanced", 16, 2000, [768, 1024], 1024),
        ("quality", 32, 3000, [768, 1024], 1024),
    ],
)
def test_materialize_config_applies_real_training_preset(
    tmp_path: Path,
    preset: str,
    rank: int,
    steps: int,
    resolution: list[int],
    sample_size: int,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (32, 32), "white").save(dataset / "001.png")
    (dataset / "001.txt").write_text("ohwx person", encoding="utf-8")

    generated = train_module.materialize_config(
        config.ROOT / "configs" / "person_sdxl.yaml",
        run_name=f"{preset}_run",
        dataset_dir=dataset,
        output_dir=tmp_path / "checkpoints",
        destination=tmp_path / f"{preset}.yaml",
        preset=preset,
    )

    loaded = yaml.safe_load(generated.read_text(encoding="utf-8"))
    process = loaded["config"]["process"][0]
    assert process["network"]["linear"] == rank
    assert process["network"]["linear_alpha"] == rank
    assert process["train"]["steps"] == steps
    assert process["datasets"][0]["resolution"] == resolution
    assert process["sample"]["width"] == sample_size
    assert loaded["meta"]["mivim_preset"] == preset


def test_materialize_config_rejects_unknown_training_preset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (32, 32), "white").save(dataset / "001.png")
    (dataset / "001.txt").write_text("ohwx person", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown training preset"):
        train_module.materialize_config(
            config.ROOT / "configs" / "person_sdxl.yaml",
            run_name="custom",
            dataset_dir=dataset,
            output_dir=tmp_path / "checkpoints",
            destination=tmp_path / "custom.yaml",
            preset="turbo",
        )


def test_materialize_config_rejects_missing_local_base_model(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (32, 32), "white").save(dataset / "001.png")
    (dataset / "001.txt").write_text("ohwx person", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Training base model not found"):
        train_module.materialize_config(
            config.ROOT / "configs" / "person_sdxl.yaml",
            run_name="missing_model_v1",
            dataset_dir=dataset,
            base_model=tmp_path / "missing.safetensors",
        )


def test_doctor_rejects_flow_scheduler_for_sdxl(tmp_path: Path) -> None:
    template = yaml.safe_load(
        (config.ROOT / "configs" / "person_sdxl.yaml").read_text(encoding="utf-8")
    )
    template["config"]["process"][0]["train"]["noise_scheduler"] = "flowmatch"
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(yaml.safe_dump(template), encoding="utf-8")
    results: list[doctor_module.CheckResult] = []

    doctor_module.check_config_file(results, bad_config)

    assert results
    assert results[0][1] == doctor_module.FAIL
    assert "flowmatch" in results[0][2]


def test_doctor_rejects_diffusers_output_for_lora_handoff(tmp_path: Path) -> None:
    template = yaml.safe_load(
        (config.ROOT / "configs" / "person_sdxl.yaml").read_text(encoding="utf-8")
    )
    template["config"]["process"][0]["save"]["save_format"] = "diffusers"
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(yaml.safe_dump(template), encoding="utf-8")
    results: list[doctor_module.CheckResult] = []

    doctor_module.check_config_file(results, bad_config)

    assert results
    assert results[0][1] == doctor_module.FAIL
    assert "safetensors" in results[0][2]


def test_doctor_rejects_a_disconnected_workflow(tmp_path: Path) -> None:
    workflow = yaml.safe_load(
        (config.ROOT / "workflows" / "validate_sdxl_api.json").read_text(
            encoding="utf-8"
        )
    )
    workflow["9"]["inputs"]["images"] = ["3", 0]
    bad_workflow = tmp_path / "bad-workflow.json"
    bad_workflow.write_text(json.dumps(workflow), encoding="utf-8")
    results: list[doctor_module.CheckResult] = []

    doctor_module.check_workflow_file(results, bad_workflow)

    assert results
    assert results[0][1] == doctor_module.FAIL
    assert "must connect" in results[0][2]


def test_materialize_rejects_invalid_dataset_pairs(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (32, 32), "white").save(dataset / "001.png")
    (dataset / "001.txt").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="caption is empty"):
        train_module.materialize_config(
            config.ROOT / "configs" / "person_sdxl.yaml",
            run_name="custom",
            dataset_dir=dataset,
            output_dir=tmp_path / "checkpoints",
            destination=tmp_path / "config.yaml",
        )


def test_trainer_pair_validation_uses_the_same_exact_file_contract(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (32, 32), "white").save(dataset / "001.png")
    Image.new("RGB", (32, 32), "white").save(dataset / "001.jpg")
    (dataset / "001.txt").write_text("ohwx person\n", encoding="utf-8")
    Image.new("RGB", (32, 32), "blue").save(dataset / "orphan.webp")
    (dataset / "notes.txt").write_text("ohwx person\n", encoding="utf-8")
    Image.new("RGB", (32, 32), "green").save(dataset / "animated.gif")

    with pytest.raises(ValueError) as error:
        train_module._validate_dataset_pairs(dataset)

    message = str(error.value)
    for filename in ("001.jpg", "001.png", "orphan.webp", "notes.txt", "animated.gif"):
        assert filename in message


def test_blip_loader_pins_the_model_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeLoader:
        @classmethod
        def from_pretrained(_cls, name: str, **kwargs: object) -> object:
            calls.append((name, kwargs))
            return FakeModel()

    class FakeModel:
        def to(self, _device: str) -> FakeModel:
            return self

        def eval(self) -> FakeModel:
            return self

    fake_torch: Any = ModuleType("torch")
    fake_torch.cuda = type(
        "FakeCuda", (), {"is_available": staticmethod(lambda: False)}
    )
    fake_transformers: Any = ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeLoader
    fake_transformers.AutoModelForMultimodalLM = FakeLoader
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(caption_module._blip, "processor", None)
    monkeypatch.setattr(caption_module._blip, "model", None)
    monkeypatch.setattr(caption_module._blip, "device", None)

    caption_module._load_blip()

    assert len(calls) == 2
    assert all(name == caption_module.BLIP_MODEL_ID for name, _ in calls)
    assert all(
        kwargs["revision"] == caption_module.BLIP_MODEL_REVISION for _, kwargs in calls
    )
    assert calls[1][1]["use_safetensors"] is True


def test_prepare_rebuilds_an_incomplete_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (80, 40), "white").save(source / "photo.png")
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")

    out = prepare_dataset.prepare(source, "subject_v1", size=32)
    (out / "001.txt").unlink()

    prepare_dataset.prepare(source, "subject_v1", size=32)

    assert (out / "001.txt").read_text(encoding="utf-8") == "ohwx person"
    with Image.open(out / "001.png") as image:
        assert image.size == (32, 16)


def test_prepare_overwrite_removes_stale_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (80, 40), "white").save(source / "a.png")
    Image.new("RGB", (80, 40), "black").save(source / "b.png")
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")

    out = prepare_dataset.prepare(source, "subject_v1", size=32)
    (source / "b.png").unlink()
    with pytest.raises(ValueError, match="Source images changed"):
        prepare_dataset.prepare(source, "subject_v1", size=32)
    prepare_dataset.prepare(source, "subject_v1", size=32, overwrite=True)

    assert (out / "001.png").exists()
    assert not (out / "002.png").exists()
    assert not (out / "002.txt").exists()


def test_prepare_detects_source_content_changes_with_preserved_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_image = source / "photo.bmp"
    Image.new("RGB", (80, 40), "white").save(source_image)
    original_stat = source_image.stat()
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")

    prepare_dataset.prepare(source, "subject_v1", size=32)
    Image.new("RGB", (80, 40), "black").save(source_image)
    assert source_image.stat().st_size == original_stat.st_size
    os.utime(
        source_image,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    with pytest.raises(ValueError, match="Source images changed"):
        prepare_dataset.prepare(source, "subject_v1", size=32)


def test_prepare_ignores_exact_duplicate_sources_without_changing_originals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    first = source / "first.png"
    Image.new("RGB", (80, 40), "white").save(first)
    (source / "second.png").write_bytes(first.read_bytes())
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")

    before = {path.name: path.read_bytes() for path in source.iterdir()}
    prepared = prepare_dataset.prepare(source, "subject_v1", size=32)

    assert len(list(prepared.glob("*.png"))) == 1
    assert len(list(prepared.glob("*.txt"))) == 1
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before


def test_failed_overwrite_preserves_the_previous_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_image = source / "photo.png"
    Image.new("RGB", (80, 40), "white").save(source_image)
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")

    out = prepare_dataset.prepare(source, "subject_v1", size=32)
    previous_image = (out / "001.png").read_bytes()
    previous_caption = (out / "001.txt").read_text(encoding="utf-8")
    source_image.write_bytes(b"not an image")

    with pytest.raises(RuntimeError, match="produced no usable pairs"):
        prepare_dataset.prepare(source, "subject_v1", size=32, overwrite=True)

    assert (out / "001.png").read_bytes() == previous_image
    assert (out / "001.txt").read_text(encoding="utf-8") == previous_caption
    assert not list((tmp_path / "datasets").glob(".subject_v1.prepare-*"))


def test_prepare_requires_overwrite_when_settings_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (80, 40), "white").save(source / "photo.png")
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")

    out = prepare_dataset.prepare(
        source,
        "subject_v1",
        size=32,
        trigger="first person",
    )
    with pytest.raises(ValueError, match="preparation settings changed"):
        prepare_dataset.prepare(
            source,
            "subject_v1",
            size=64,
            trigger="second person",
        )

    prepare_dataset.prepare(
        source,
        "subject_v1",
        size=64,
        trigger="second person",
        overwrite=True,
    )
    with Image.open(out / "001.png") as image:
        assert image.size == (64, 32)
    assert (out / "001.txt").read_text(encoding="utf-8") == "second person"


def test_prepare_does_not_hide_vision_caption_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (80, 40), "white").save(source / "photo.png")
    datasets = tmp_path / "datasets"
    monkeypatch.setattr(config, "DATASETS_DIR", datasets)
    monkeypatch.setattr(
        prepare_dataset,
        "make_caption",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("BLIP unavailable")
        ),
    )

    with pytest.raises(RuntimeError, match="produced no usable pairs"):
        prepare_dataset.prepare(
            source,
            "subject_v1",
            size=32,
            caption_mode="blip",
        )

    out = datasets / "subject_v1"
    assert not (out / "001.png").exists()
    assert not (out / "001.txt").exists()
    assert not (out / prepare_dataset.MANIFEST_NAME).exists()


def test_validate_fails_when_comfy_returns_no_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NoOutputClient:
        def is_up(self) -> bool:
            return True

        def run_and_save(
            self, graph: dict[str, object], out_dir: Path, prefix: str = ""
        ) -> list[Path]:
            return []

    monkeypatch.setattr(validate_module, "ComfyClient", NoOutputClient)
    monkeypatch.setattr(config, "GRIDS_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="produced 0 images"):
        validate_module.validate(
            lora="subject.safetensors",
            strengths=[0.8],
            prompts=["{trigger}, portrait"],
            trigger="ohwx person",
            seed=42,
            run_name="subject_v1",
        )


def test_validate_rejects_an_invalid_seed() -> None:
    with pytest.raises(ValueError, match="Validation seed"):
        validate_module.validate(
            lora="subject.safetensors",
            strengths=[0.8],
            prompts=["{trigger}, portrait"],
            trigger="ohwx person",
            seed=-1,
            run_name="subject_v1",
        )


def test_validate_rejects_absolute_or_parent_model_paths() -> None:
    with pytest.raises(ValueError, match="safe filename"):
        validate_module.validation_signature(
            lora="../private/model.safetensors",
            strengths=[0.8],
            prompts=["{trigger}, portrait"],
            trigger="ohwx person",
            seed=42,
            lora_content_identity=TEST_LORA_IDENTITY,
            checkpoint_content_identity=TEST_CHECKPOINT_IDENTITY,
        )


def test_validation_signature_is_stable_and_changes_with_generation_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "COMFY_HOST", "127.0.0.1:8188")
    monkeypatch.setattr(config, "COMFY_SDXL_CKPT", "portrait-xl.safetensors")

    first = validate_module.validation_signature(
        lora="portrait_v1.safetensors",
        strengths=[0.0, 0.8],
        prompts=["{trigger}, portrait"],
        trigger="ohwx person",
        seed=42,
        lora_content_identity=TEST_LORA_IDENTITY,
        checkpoint_content_identity=TEST_CHECKPOINT_IDENTITY,
    )

    assert (
        validate_module.validation_signature(
            lora="portrait_v1.safetensors",
            strengths=[0.0, 0.8],
            prompts=["{trigger}, portrait"],
            trigger="ohwx person",
            seed=42,
            lora_content_identity=TEST_LORA_IDENTITY,
            checkpoint_content_identity=TEST_CHECKPOINT_IDENTITY,
        )
        == first
    )
    assert (
        validate_module.validation_signature(
            lora="portrait_v1.safetensors",
            strengths=[0.0, 0.8],
            prompts=["{trigger}, portrait"],
            trigger="ohwx person",
            seed=43,
            lora_content_identity=TEST_LORA_IDENTITY,
            checkpoint_content_identity=TEST_CHECKPOINT_IDENTITY,
        )
        != first
    )
    assert (
        validate_module.validation_signature(
            lora="portrait_v1.safetensors",
            strengths=[0.0, 0.8],
            prompts=["{trigger}, outdoors"],
            trigger="ohwx person",
            seed=42,
            lora_content_identity=TEST_LORA_IDENTITY,
            checkpoint_content_identity=TEST_CHECKPOINT_IDENTITY,
        )
        != first
    )
    assert (
        validate_module.validation_signature(
            lora="portrait_v1.safetensors",
            strengths=[0.0, 0.8],
            prompts=["{trigger}, portrait"],
            trigger="ohwx person",
            seed=42,
            lora_content_identity="sha256:test-lora-b",
            checkpoint_content_identity=TEST_CHECKPOINT_IDENTITY,
        )
        != first
    )
    monkeypatch.setattr(config, "COMFY_HOST", "127.0.0.1:8189")
    assert (
        validate_module.validation_signature(
            lora="portrait_v1.safetensors",
            strengths=[0.0, 0.8],
            prompts=["{trigger}, portrait"],
            trigger="ohwx person",
            seed=42,
            lora_content_identity=TEST_LORA_IDENTITY,
            checkpoint_content_identity=TEST_CHECKPOINT_IDENTITY,
        )
        != first
    )


def test_validation_retry_reuses_verified_cells_after_a_mid_sweep_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, Any]] = []
    fail_once = True

    class InterruptedClient:
        def is_up(self) -> bool:
            return True

        def run_and_save(
            self, graph: dict[str, Any], out_dir: Path, prefix: str = ""
        ) -> list[Path]:
            nonlocal fail_once
            calls.append(graph)
            if len(calls) == 3 and fail_once:
                fail_once = False
                raise RuntimeError("simulated ComfyUI interruption")
            out_dir.mkdir(parents=True, exist_ok=True)
            output = out_dir / f"{prefix}output.png"
            Image.new("RGB", (32, 32), (len(calls) * 20, 80, 120)).save(output)
            return [output]

    monkeypatch.setattr(validate_module, "ComfyClient", InterruptedClient)
    monkeypatch.setattr(config, "GRIDS_DIR", tmp_path)

    def run_validation() -> Path:
        return validate_module.validate(
            lora="resume_v1.safetensors",
            strengths=[0.0, 0.8],
            prompts=["{trigger}, portrait", "{trigger}, outdoors"],
            trigger="ohwx person",
            seed=42,
            run_name="resume_v1",
            lora_content_identity=TEST_LORA_IDENTITY,
            checkpoint_content_identity=TEST_CHECKPOINT_IDENTITY,
        )

    with pytest.raises(RuntimeError, match="simulated ComfyUI interruption"):
        run_validation()

    attempt_root = tmp_path / "resume_v1" / "attempts"
    attempt_dir = next(attempt_root.iterdir())
    manifest_path = attempt_dir / validate_module.VALIDATION_MANIFEST_NAME
    interrupted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert interrupted["status"] == "in_progress"
    assert interrupted["completed_cells"] == 2
    assert len(interrupted["cells"]) == 2
    capsys.readouterr()

    calls_before_retry = len(calls)
    grid = run_validation()

    assert len(calls) - calls_before_retry == 2
    assert grid.is_file()
    completed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert completed["status"] == "complete"
    assert completed["processed_cells"] == 4
    assert completed["completed_cells"] == 4
    assert len(completed["cells"]) == 4
    assert len(list((attempt_dir / "cells").glob("*.png"))) == 4
    workflow_path = validate_module.ready_workflow_path("resume_v1")
    ready_workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    assert ready_workflow["version"] == 0.4
    ready_nodes = ready_workflow["nodes"]
    assert {node["type"] for node in ready_nodes} <= validate_module.CORE_NODE_TYPES
    lora_node = next(node for node in ready_nodes if node["type"] == "LoraLoader")
    assert lora_node["widgets_values"][0] == "resume_v1.safetensors"
    assert lora_node["widgets_values"][1] == 0.8
    ready_prompt = validate_module._ui_workflow_to_prompt(ready_workflow)
    assert ready_prompt in calls
    assert str(tmp_path) not in workflow_path.read_text(encoding="utf-8")
    with Image.open(grid) as published_grid:
        assert json.loads(published_grid.info["prompt"]) == ready_prompt
        assert json.loads(published_grid.info["workflow"]) == ready_workflow
    assert completed["ready_workflow"]["embedded_prompt_in_grid"] is True
    assert completed["ready_workflow"]["embedded_workflow_in_grid"] is True
    output = capsys.readouterr().out
    assert "[validate] cell 1/4 reused strength=0 prompt#0" in output
    assert "[validate] cell 2/4 reused strength=0 prompt#1" in output
    assert "[validate] cell 3/4 created strength=0.8 prompt#0" in output
    assert "[validate] cell 4/4 created strength=0.8 prompt#1" in output


def test_validation_changed_inputs_use_an_isolated_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, Any]] = []

    class ImageClient:
        def is_up(self) -> bool:
            return True

        def run_and_save(
            self, graph: dict[str, Any], out_dir: Path, prefix: str = ""
        ) -> list[Path]:
            calls.append(graph)
            out_dir.mkdir(parents=True, exist_ok=True)
            output = out_dir / f"{prefix}output.png"
            Image.new("RGB", (24, 24), (20 * len(calls), 90, 130)).save(output)
            return [output]

    monkeypatch.setattr(validate_module, "ComfyClient", ImageClient)
    monkeypatch.setattr(config, "GRIDS_DIR", tmp_path)
    portrait_signature = validate_module.validation_signature(
        lora="isolated_v1.safetensors",
        strengths=[0.8],
        prompts=["{trigger}, portrait"],
        trigger="ohwx person",
        seed=42,
        lora_content_identity=TEST_LORA_IDENTITY,
        checkpoint_content_identity=TEST_CHECKPOINT_IDENTITY,
    )
    outdoor_signature = validate_module.validation_signature(
        lora="isolated_v1.safetensors",
        strengths=[0.8],
        prompts=["{trigger}, outdoors"],
        trigger="ohwx person",
        seed=42,
        lora_content_identity=TEST_LORA_IDENTITY,
        checkpoint_content_identity=TEST_CHECKPOINT_IDENTITY,
    )
    validate_module.validate(
        lora="isolated_v1.safetensors",
        strengths=[0.8],
        prompts=["{trigger}, portrait"],
        trigger="ohwx person",
        seed=42,
        run_name="isolated_v1",
        lora_content_identity=TEST_LORA_IDENTITY,
        checkpoint_content_identity=TEST_CHECKPOINT_IDENTITY,
    )
    capsys.readouterr()
    validate_module.validate(
        lora="isolated_v1.safetensors",
        strengths=[0.8],
        prompts=["{trigger}, outdoors"],
        trigger="ohwx person",
        seed=42,
        run_name="isolated_v1",
        lora_content_identity=TEST_LORA_IDENTITY,
        checkpoint_content_identity=TEST_CHECKPOINT_IDENTITY,
    )

    attempts = sorted((tmp_path / "isolated_v1" / "attempts").iterdir())
    assert len(calls) == 2
    assert len(attempts) == 2
    assert {attempt.name for attempt in attempts} == {
        portrait_signature,
        outdoor_signature,
    }
    assert all(
        len(list((attempt / "cells").glob("*.png"))) == 1 for attempt in attempts
    )
    signatures = {
        json.loads(
            (attempt / validate_module.VALIDATION_MANIFEST_NAME).read_text(
                encoding="utf-8"
            )
        )["signature"]
        for attempt in attempts
    }
    assert len(signatures) == 2
    assert (
        "[validate] cell 1/1 created strength=0.8 prompt#0" in capsys.readouterr().out
    )


def test_validation_changed_model_bytes_do_not_reuse_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class ImageClient:
        def is_up(self) -> bool:
            return True

        def run_and_save(
            self, graph: dict[str, Any], out_dir: Path, prefix: str = ""
        ) -> list[Path]:
            del graph
            nonlocal calls
            calls += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            output = out_dir / f"{prefix}output.png"
            Image.new("RGB", (24, 24), (calls * 30, 90, 130)).save(output)
            return [output]

    monkeypatch.setattr(validate_module, "ComfyClient", ImageClient)
    monkeypatch.setattr(config, "GRIDS_DIR", tmp_path)

    def run(lora_identity: str) -> None:
        validate_module.validate(
            lora="same_name.safetensors",
            strengths=[0.8],
            prompts=["{trigger}, portrait"],
            trigger="ohwx person",
            seed=42,
            run_name="same_name_v1",
            lora_content_identity=lora_identity,
            checkpoint_content_identity=TEST_CHECKPOINT_IDENTITY,
        )

    run("sha256:first-lora-bytes")
    run("sha256:replacement-lora-bytes")

    attempts = list((tmp_path / "same_name_v1" / "attempts").iterdir())
    assert calls == 2
    assert len(attempts) == 2


def test_validation_does_not_reuse_a_corrupted_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    class ImageClient:
        def is_up(self) -> bool:
            return True

        def run_and_save(
            self, graph: dict[str, Any], out_dir: Path, prefix: str = ""
        ) -> list[Path]:
            nonlocal calls
            calls += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            output = out_dir / f"{prefix}output.png"
            Image.new("RGB", (24, 24), (30 * calls, 100, 140)).save(output)
            return [output]

    monkeypatch.setattr(validate_module, "ComfyClient", ImageClient)
    monkeypatch.setattr(config, "GRIDS_DIR", tmp_path)

    def run_validation() -> Path:
        return validate_module.validate(
            lora="corrupt_v1.safetensors",
            strengths=[0.8],
            prompts=["{trigger}, portrait"],
            trigger="ohwx person",
            seed=42,
            run_name="corrupt_v1",
            lora_content_identity=TEST_LORA_IDENTITY,
            checkpoint_content_identity=TEST_CHECKPOINT_IDENTITY,
        )

    run_validation()
    attempt_dir = next((tmp_path / "corrupt_v1" / "attempts").iterdir())
    cell = next((attempt_dir / "cells").glob("*.png"))
    cell.write_bytes(b"not an image")
    capsys.readouterr()

    run_validation()

    assert calls == 2
    with Image.open(cell) as regenerated:
        regenerated.load()
        assert regenerated.size == (24, 24)
    assert (
        "[validate] cell 1/1 created strength=0.8 prompt#0" in capsys.readouterr().out
    )


def test_validation_grid_has_plain_language_axes_and_baseline_space(
    tmp_path: Path,
) -> None:
    cells: list[Path] = []
    for index in range(4):
        path = tmp_path / f"cell-{index}.png"
        Image.new("RGB", (24, 24), (index * 40, 80, 120)).save(path)
        cells.append(path)
    output = tmp_path / "grid.png"

    validate_module._grid(
        cells,
        cols=2,
        cell=32,
        labels=["", "", "", ""],
        out_path=output,
        row_labels=["No LoRA · baseline", "LoRA strength 0.8"],
        column_labels=["Portrait", "Outdoor"],
    )

    with Image.open(output) as grid:
        assert grid.size == (324, 150)


def test_comfy_output_filename_cannot_escape_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = ComfyClient()
    monkeypatch.setattr(
        client,
        "_history",
        lambda _prompt_id: {
            "prompt": {
                "outputs": {
                    "9": {
                        "images": [
                            {
                                "filename": "../../escape.png",
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                }
            }
        },
    )
    monkeypatch.setattr(client, "_view_bytes", lambda *_args: b"image")

    saved = client.save_outputs("prompt", tmp_path / "safe")

    assert saved == [tmp_path / "safe" / "escape.png"]
    assert not (tmp_path / "escape.png").exists()


def test_comfy_queue_surfaces_workflow_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ComfyClient()
    monkeypatch.setattr(
        client,
        "_http_json",
        lambda *_args, **_kwargs: {"node_errors": {"10": {"errors": ["unknown LoRA"]}}},
    )

    with pytest.raises(RuntimeError, match="rejected the workflow"):
        client.queue_prompt({})


def test_comfy_http_errors_include_the_response_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ComfyClient()
    error = HTTPError(
        "http://localhost/prompt",
        400,
        "Bad Request",
        Message(),
        io.BytesIO(b'{"error": "unknown checkpoint"}'),
    )
    monkeypatch.setattr(
        "src.comfy_client.urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(RuntimeError, match="unknown checkpoint"):
        client.queue_prompt({})


def test_project_name_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="Project name"):
        config.validate_project_name("../outside")


def test_project_name_rejects_names_too_long_for_generated_windows_files() -> None:
    with pytest.raises(ValueError, match="80 characters or fewer"):
        config.validate_project_name("a" * (config.MAX_PROJECT_NAME_LENGTH + 1))

    assert config.validate_project_name("a" * config.MAX_PROJECT_NAME_LENGTH)


@pytest.mark.parametrize("name", ["CON", "con.txt", "person."])
def test_project_name_rejects_windows_reserved_names(name: str) -> None:
    with pytest.raises(ValueError, match="Windows directory"):
        config.validate_project_name(name)


def test_fast_pipeline_preset_defaults_photo_preparation_to_768(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_prepare(**kwargs: object) -> Path:
        captured.update(kwargs)
        prepared = tmp_path / "datasets" / "subject_v1"
        prepared.mkdir(parents=True)
        (prepared / "001.txt").write_text("ohwx person\n", encoding="utf-8")
        return prepared

    monkeypatch.setattr(pipeline_module.prep, "prepare", fake_prepare)
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(config, "RUN_CONFIGS_DIR", tmp_path / "configs")

    pipeline_module.run(
        _pipeline_args(
            src=str(tmp_path),
            size=None,
            preset="fast",
            skip_prep=False,
            skip_train=True,
            skip_validate=True,
        )
    )

    assert captured["size"] == 768


def test_pipeline_uses_only_a_new_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    comfy_loras = tmp_path / "loras"
    comfy_loras.mkdir()
    datasets = tmp_path / "datasets"
    (datasets / "subject_v1").mkdir(parents=True)
    grids = tmp_path / "grids"

    monkeypatch.setattr(config, "CHECKPOINTS_DIR", checkpoints)
    monkeypatch.setattr(config, "COMFY_LORA_DIR", comfy_loras)
    monkeypatch.setattr(config, "DATASETS_DIR", datasets)
    monkeypatch.setattr(config, "GRIDS_DIR", grids)
    monkeypatch.setattr(config, "RUN_CONFIGS_DIR", tmp_path / "configs")

    def fake_train(*_args: object, **_kwargs: object) -> int:
        (checkpoints / "subject_v1_02000.safetensors").write_bytes(b"weights")
        return 0

    captured: dict[str, str] = {}

    def fake_validate(**kwargs: object) -> Path:
        captured["lora"] = str(kwargs["lora"])
        grid = grids / "subject_v1_grid.png"
        grid.parent.mkdir(parents=True, exist_ok=True)
        grid.write_bytes(b"verified grid")
        (grids / "subject_v1_use_in_comfyui.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
        return grid

    monkeypatch.setattr(pipeline_module.trainer, "train", fake_train)
    monkeypatch.setattr(pipeline_module.validator, "validate", fake_validate)

    pipeline_module.run(_pipeline_args())

    assert captured["lora"] == "subject_v1_02000.safetensors"
    assert (comfy_loras / captured["lora"]).read_bytes() == b"weights"
    recovery = json.loads(
        (tmp_path / "configs" / "subject_v1-recovery.json").read_text(encoding="utf-8")
    )
    assert recovery["schema_version"] == 2
    assert recovery["state"] == "complete"
    assert recovery["lora_name"] == captured["lora"]
    assert recovery["lora_fingerprint"] == io_utils.artifact_fingerprint(
        comfy_loras / captured["lora"]
    )
    assert recovery["validation_signature"]
    assert recovery["strengths"] == [0.6, 0.8]


def test_pipeline_preserves_installed_lora_recovery_when_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lora_dir = tmp_path / "ComfyUI" / "models" / "loras"
    lora_dir.mkdir(parents=True)
    installed = lora_dir / "subject_v1.safetensors"
    installed.write_bytes(b"trained weights")
    run_configs = tmp_path / "configs"
    grids = tmp_path / "grids"
    monkeypatch.setattr(config, "COMFY_LORA_DIR", lora_dir)
    monkeypatch.setattr(config, "RUN_CONFIGS_DIR", run_configs)
    monkeypatch.setattr(config, "GRIDS_DIR", grids)
    monkeypatch.setattr(config, "COMFY_SDXL_CKPT", "portraitXL.safetensors")
    monkeypatch.setattr(
        pipeline_module.validator,
        "validate",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated validation failure")
        ),
    )

    with pytest.raises(RuntimeError, match="simulated validation failure"):
        pipeline_module.run(
            _pipeline_args(
                skip_train=True,
                lora=installed.name,
                preset="balanced",
            )
        )

    recovery = json.loads(
        (run_configs / "subject_v1-recovery.json").read_text(encoding="utf-8")
    )
    assert recovery["schema_version"] == 2
    assert recovery["project_name"] == "subject_v1"
    assert recovery["state"] == "validation_pending"
    assert recovery["lora_name"] == installed.name
    assert recovery["installed_lora_path"] == str(installed.resolve())
    assert recovery["expected_grid_path"] == str(
        (grids / "subject_v1_grid.png").resolve()
    )
    assert recovery["trigger"] == "ohwx person"
    assert recovery["preset"] == "balanced"
    assert recovery["checkpoint"] == "portraitXL.safetensors"
    assert recovery["lora_fingerprint"] == io_utils.artifact_fingerprint(installed)
    assert recovery["checkpoint_path"] == ""
    assert recovery["checkpoint_fingerprint"] == ""
    assert recovery["validation_signature"]
    assert recovery["strengths"] == [0.6, 0.8]
    assert recovery["seed"] == 42
    assert not (run_configs / "subject_v1-result.json").exists()


def test_pipeline_rejects_a_noop_command() -> None:
    with pytest.raises(ValueError, match="nothing to run"):
        pipeline_module.run(
            _pipeline_args(
                skip_prep=True,
                skip_train=True,
                skip_validate=True,
            )
        )


def test_pipeline_rejects_existing_lora_while_training() -> None:
    with pytest.raises(ValueError, match="requires --skip-train"):
        pipeline_module.run(_pipeline_args(lora="existing.safetensors"))


def test_pipeline_rejects_a_stale_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "subject_v1_old.safetensors").write_bytes(b"old")
    datasets = tmp_path / "datasets"
    (datasets / "subject_v1").mkdir(parents=True)

    monkeypatch.setattr(config, "CHECKPOINTS_DIR", checkpoints)
    monkeypatch.setattr(config, "DATASETS_DIR", datasets)
    monkeypatch.setattr(pipeline_module.trainer, "train", lambda *_a, **_kw: 0)

    with pytest.raises(RuntimeError, match="No new trained LoRA"):
        pipeline_module.run(_pipeline_args())


def test_checkpoint_discovery_does_not_cross_project_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    expected = checkpoints / "person_000000250.safetensors"
    expected.write_bytes(b"expected")
    (checkpoints / "person_v1_000000250.safetensors").write_bytes(b"other")
    (checkpoints / "person-old_000000250.safetensors").write_bytes(b"other")
    monkeypatch.setattr(config, "CHECKPOINTS_DIR", checkpoints)

    assert pipeline_module._checkpoints("person") == [expected]


def test_comfy_lora_replacement_keeps_a_recoverable_previous_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "outputs" / "portrait_v1.safetensors"
    source.parent.mkdir()
    source.write_bytes(b"new-model")
    lora_dir = tmp_path / "ComfyUI" / "models" / "loras"
    lora_dir.mkdir(parents=True)
    destination = lora_dir / source.name
    destination.write_bytes(b"old-model")
    monkeypatch.setattr(config, "COMFY_LORA_DIR", lora_dir)

    name = pipeline_module._copy_lora_to_comfy(source)

    assert name == source.name
    assert destination.read_bytes() == b"new-model"
    assert (lora_dir / f"{source.name}.previous").read_bytes() == b"old-model"
