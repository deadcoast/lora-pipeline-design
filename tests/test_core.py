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

    generated = train_module.materialize_config(
        config.ROOT / "configs" / "person_sdxl.yaml",
        run_name="custom",
        trigger="custom person",
        dataset_dir=dataset,
        output_dir=checkpoints,
        destination=destination,
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
    assert process["trigger_word"] == "custom person"
    assert all(
        sample["prompt"].startswith("custom person,")
        for sample in process["sample"]["samples"]
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


def test_prepare_rejects_exact_duplicate_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    first = source / "first.png"
    Image.new("RGB", (80, 40), "white").save(first)
    (source / "second.png").write_bytes(first.read_bytes())
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")

    with pytest.raises(ValueError, match="Duplicate source images"):
        prepare_dataset.prepare(source, "subject_v1", size=32)


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


@pytest.mark.parametrize("name", ["CON", "con.txt", "person."])
def test_project_name_rejects_windows_reserved_names(name: str) -> None:
    with pytest.raises(ValueError, match="Windows directory"):
        config.validate_project_name(name)


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

    def fake_train(*_args: object, **_kwargs: object) -> int:
        (checkpoints / "subject_v1_02000.safetensors").write_bytes(b"weights")
        return 0

    captured: dict[str, str] = {}

    def fake_validate(**kwargs: object) -> Path:
        captured["lora"] = str(kwargs["lora"])
        return grids / "subject_v1_grid.png"

    monkeypatch.setattr(pipeline_module.trainer, "train", fake_train)
    monkeypatch.setattr(pipeline_module.validator, "validate", fake_validate)

    pipeline_module.run(_pipeline_args())

    assert captured["lora"] == "subject_v1_02000.safetensors"
    assert (comfy_loras / captured["lora"]).read_bytes() == b"weights"


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
