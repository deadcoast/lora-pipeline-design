from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.app_models import AppSettings
from src.app_service import ApplicationService
from src.library import (
    LibraryIndex,
    LibraryIndexError,
    RunTags,
    checkpoint_style,
    load_library_index,
    next_version_name,
    normalize_tag,
    profile_for_checkpoint,
    save_library_index,
    scan_library,
    tag_error,
)

FINGERPRINT = "sha256:" + "0" * 64


# ----------------------------------------------------------------- tags


def test_tags_are_lowercased_and_trimmed() -> None:
    assert normalize_tag("  Anime ") == "anime"
    assert normalize_tag("") == ""


def test_tag_error_explains_the_one_word_rule() -> None:
    assert tag_error("anime") is None
    assert tag_error("water-colour2") is None
    assert tag_error("") == "Enter one word."
    assert tag_error("water color") == "One word: lowercase letters, numbers, hyphens."
    assert tag_error("-anime") == "One word: lowercase letters, numbers, hyphens."
    assert tag_error("a" * 33) == "Use 32 characters or fewer."


# ---------------------------------------------------------------- index


def test_library_index_round_trips_through_json(tmp_path: Path) -> None:
    path = tmp_path / "library.json"
    index = (
        LibraryIndex()
        .with_character("ruth")
        .with_style("anime")
        .with_checkpoint_style("NoobAI-XL-v1.1.safetensors", "anime")
        .with_run_tags("ruth-mk01", RunTags("ruth", "photo"))
    )

    save_library_index(path, index)

    assert load_library_index(path) == index
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_missing_index_is_empty(tmp_path: Path) -> None:
    assert load_library_index(tmp_path / "library.json") == LibraryIndex()


def test_corrupt_index_raises_and_is_left_untouched(tmp_path: Path) -> None:
    path = tmp_path / "library.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(LibraryIndexError):
        load_library_index(path)

    assert path.read_text(encoding="utf-8") == "{not json"


def test_vocabulary_dedupes_case_and_whitespace() -> None:
    index = LibraryIndex().with_style("Anime").with_style(" anime ").with_style("photo")

    assert index.styles == ("anime", "photo")


def test_vocabulary_rejects_invalid_words() -> None:
    with pytest.raises(ValueError):
        LibraryIndex().with_character("two words")


def test_tagging_a_run_adds_both_words_to_the_vocabulary() -> None:
    index = LibraryIndex().with_run_tags("ruth-mk01", RunTags("Ruth", "anime"))

    assert index.characters == ("ruth",)
    assert index.styles == ("anime",)
    assert index.run_tags["ruth-mk01"] == RunTags("ruth", "anime")


# ---------------------------------------------------------- checkpoints


def _write_sdxl_base(path: Path) -> None:
    header = {
        "model.diffusion_model.label_emb.0.0.weight": {
            "dtype": "F16",
            "shape": [1280, 2816],
            "data_offsets": [0, 0],
        },
        "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": {
            "dtype": "F16",
            "shape": [640, 2048],
            "data_offsets": [0, 0],
        },
    }
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded)


def test_checkpoint_style_prefers_the_index_label_then_the_filename(
    tmp_path: Path,
) -> None:
    index = LibraryIndex().with_checkpoint_style("NoobAI-XL-v1.1.safetensors", "chibi")
    base = tmp_path / "sd_xl_base_1.0.safetensors"
    _write_sdxl_base(base)

    assert checkpoint_style("NoobAI-XL-v1.1.safetensors", index) == "chibi"
    assert (
        checkpoint_style("NoobAI-XL-Vpred-v1.0.safetensors", LibraryIndex()) == "anime"
    )
    assert checkpoint_style(base.name, LibraryIndex(), path=base) == "photo"
    assert checkpoint_style("mystery.safetensors", LibraryIndex()) == ""


def test_profile_follows_the_checkpoint_filename() -> None:
    assert (
        profile_for_checkpoint("NoobAI-XL-Vpred-v1.0.safetensors") == "noobai-xl-vpred"
    )
    assert profile_for_checkpoint("NoobAI-XL-v1.1.safetensors") == "noobai-xl-epsilon"
    assert profile_for_checkpoint("sd_xl_base_1.0.safetensors") == "sdxl"


# ----------------------------------------------------------------- scan


def _workspace(tmp_path: Path) -> tuple[ApplicationService, AppSettings]:
    settings = AppSettings(
        comfy_lora_dir=tmp_path / "loras",
        datasets_dir=Path("ws/datasets"),
        checkpoints_dir=Path("ws/checkpoints"),
        grids_dir=Path("ws/grids"),
        run_configs_dir=Path("ws/configs"),
    )
    return ApplicationService(root=tmp_path), settings


def _write_full_record(
    root: Path,
    settings: AppSettings,
    name: str,
    *,
    completed_at: str,
    character: str = "",
    artstyle: str = "",
    schema_version: int = 5,
    grid: bool = True,
    lora: bool = True,
    mode: str = "full",
) -> Path:
    grids = root / settings.grids_dir
    grids.mkdir(parents=True, exist_ok=True)
    grid_path = grids / f"{name}_grid.png"
    if grid:
        grid_path.write_bytes(b"png")
    workflow = grids / f"{name}_use_in_comfyui.json"
    workflow.write_text("{}", encoding="utf-8")
    assert settings.comfy_lora_dir is not None
    lora_path = settings.comfy_lora_dir / f"{name}.safetensors"
    if lora:
        lora_path.parent.mkdir(parents=True, exist_ok=True)
        lora_path.write_bytes(b"lora")
    record: dict[str, object] = {
        "schema_version": schema_version,
        "model_profile": "noobai-xl-epsilon",
        "project_name": name,
        "training_project_name": "",
        "attempt_id": "attempt",
        "mode": mode,
        "result_path": str(grid_path),
        "result_fingerprint": FINGERPRINT,
        "trigger": name,
        "preset": "balanced",
        "checkpoint": "NoobAI-XL-v1.1.safetensors",
        "lora_name": f"{name}.safetensors",
        "workflow_path": str(workflow),
        "workflow_fingerprint": FINGERPRINT,
        "comfy_installation_id": "inst-1",
        "comfy_host": "127.0.0.1:8188",
        "installed_lora_path": str(lora_path),
        "lora_fingerprint": FINGERPRINT,
        "checkpoint_path": str(root / "NoobAI-XL-v1.1.safetensors"),
        "checkpoint_fingerprint": FINGERPRINT,
        "starting_strength": 0.8,
        "completed_at": completed_at,
    }
    if schema_version >= 5:
        record["character"] = character
        record["artstyle"] = artstyle
        record["source_dir"] = str(root / "photos")
    configs = root / settings.run_configs_dir
    configs.mkdir(parents=True, exist_ok=True)
    (configs / f"{name}-result.json").write_text(json.dumps(record), encoding="utf-8")
    return grid_path


def _write_run_config(root: Path, settings: AppSettings, name: str, steps: int) -> None:
    configs = root / settings.run_configs_dir
    configs.mkdir(parents=True, exist_ok=True)
    (configs / f"{name}.yaml").write_text(
        json.dumps(
            {
                "config": {
                    "name": name,
                    "process": [{"train": {"steps": steps}}],
                }
            }
        ),
        encoding="utf-8",
    )


def test_scan_groups_tagged_runs_by_character_newest_first(tmp_path: Path) -> None:
    service, settings = _workspace(tmp_path)
    _write_full_record(
        tmp_path,
        settings,
        "alex-photo-mk01",
        character="alex",
        artstyle="photo",
        completed_at="2026-10-12T10:00:00+00:00",
    )
    _write_full_record(
        tmp_path,
        settings,
        "ruth-mk02",
        schema_version=4,
        completed_at="2026-10-13T20:00:00+00:00",
    )
    _write_full_record(
        tmp_path,
        settings,
        "ruth-anime-mk03",
        character="ruth",
        artstyle="anime",
        completed_at="2026-10-13T21:00:00+00:00",
    )
    index = LibraryIndex().with_run_tags("ruth-mk02", RunTags("ruth", "anime"))

    library = scan_library(service, settings, index)

    assert [
        (character, [entry.project_name for entry in entries])
        for character, entries in library.characters()
    ] == [
        ("ruth", ["ruth-anime-mk03", "ruth-mk02"]),
        ("alex", ["alex-photo-mk01"]),
    ]
    assert library.unsorted() == ()
    assert library.styles() == ("anime", "photo")


def test_untagged_runs_are_unsorted(tmp_path: Path) -> None:
    service, settings = _workspace(tmp_path)
    _write_full_record(
        tmp_path,
        settings,
        "ruth-mk01",
        schema_version=4,
        completed_at="2026-10-11T20:00:00+00:00",
    )

    library = scan_library(service, settings, LibraryIndex())

    assert library.characters() == ()
    (entry,) = library.unsorted()
    assert entry.project_name == "ruth-mk01"
    assert entry.character == ""
    assert entry.artstyle == ""


def test_index_tags_override_the_record_tags(tmp_path: Path) -> None:
    service, settings = _workspace(tmp_path)
    _write_full_record(
        tmp_path,
        settings,
        "ruth-anime-mk03",
        character="ruth",
        artstyle="anime",
        completed_at="2026-10-13T21:00:00+00:00",
    )
    index = LibraryIndex().with_run_tags("ruth-anime-mk03", RunTags("ruthie", "anime"))

    library = scan_library(service, settings, index)

    ((character, _),) = library.characters()
    assert character == "ruthie"


def test_entry_carries_the_restorable_result_record(tmp_path: Path) -> None:
    service, settings = _workspace(tmp_path)
    grid = _write_full_record(
        tmp_path,
        settings,
        "ruth-anime-mk03",
        character="ruth",
        artstyle="anime",
        completed_at="2026-10-13T21:00:00+00:00",
    )

    (entry,) = scan_library(service, settings, LibraryIndex()).entries

    assert entry.record is not None
    assert entry.record.result_path == grid
    assert entry.record.lora_name == "ruth-anime-mk03.safetensors"
    assert entry.record.model_profile == "noobai-xl-epsilon"
    assert entry.record.character == "ruth"
    assert entry.record.artstyle == "anime"
    assert entry.record.source_dir == tmp_path / "photos"
    assert entry.grid_path == grid
    assert entry.checkpoint == "NoobAI-XL-v1.1.safetensors"
    assert entry.trigger == "ruth-anime-mk03"


def test_statuses_reflect_what_is_still_on_disk(tmp_path: Path) -> None:
    service, settings = _workspace(tmp_path)
    when = "2026-10-13T21:00:00+00:00"
    _write_full_record(tmp_path, settings, "ready", completed_at=when)
    _write_full_record(tmp_path, settings, "nolora", completed_at=when, lora=False)
    _write_full_record(tmp_path, settings, "nogrid", completed_at=when, grid=False)
    _write_full_record(tmp_path, settings, "prepped", completed_at=when, mode="prepare")
    configs = tmp_path / settings.run_configs_dir
    (configs / "pending-recovery.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_name": "pending",
                "state": "validation_pending",
                "lora_name": "pending.safetensors",
                "installed_lora_path": str(tmp_path / "loras" / "pending.safetensors"),
                "expected_grid_path": str(
                    tmp_path / "ws" / "grids" / "pending_grid.png"
                ),
                "trigger": "pending",
                "preset": "balanced",
                "checkpoint": "NoobAI-XL-v1.1.safetensors",
                "updated_at": when,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / settings.checkpoints_dir / "crashed").mkdir(parents=True)
    (configs / "broken-result.json").write_text("{bad", encoding="utf-8")

    library = scan_library(service, settings, LibraryIndex())

    assert {entry.project_name: entry.status for entry in library.entries} == {
        "ready": "ready",
        "nolora": "lora_missing",
        "nogrid": "result_missing",
        "prepped": "prepared",
        "pending": "validation_pending",
        "crashed": "incomplete",
        "broken": "incomplete",
    }
    pending = next(
        entry for entry in library.entries if entry.project_name == "pending"
    )
    assert pending.trigger == "pending"
    assert pending.checkpoint == "NoobAI-XL-v1.1.safetensors"


def test_steps_come_from_the_saved_run_config(tmp_path: Path) -> None:
    service, settings = _workspace(tmp_path)
    when = "2026-10-13T21:00:00+00:00"
    _write_full_record(tmp_path, settings, "counted", completed_at=when)
    _write_run_config(tmp_path, settings, "counted", 1500)
    _write_full_record(tmp_path, settings, "uncounted", completed_at=when)

    steps = {
        entry.project_name: entry.steps
        for entry in scan_library(service, settings, LibraryIndex()).entries
    }

    assert steps == {"counted": 1500, "uncounted": None}


def test_scan_of_an_empty_workspace_is_empty(tmp_path: Path) -> None:
    service, settings = _workspace(tmp_path)

    library = scan_library(service, settings, LibraryIndex())

    assert library.entries == ()
    assert library.characters() == ()
    assert library.styles() == ()


# ----------------------------------------------------------- next name


def test_next_version_name_continues_the_character_and_style_group(
    tmp_path: Path,
) -> None:
    service, settings = _workspace(tmp_path)
    _write_full_record(
        tmp_path,
        settings,
        "ruth-mk02",
        schema_version=4,
        completed_at="2026-10-13T20:00:00+00:00",
    )
    _write_full_record(
        tmp_path,
        settings,
        "ruth-photo-mk07",
        character="ruth",
        artstyle="photo",
        completed_at="2026-10-13T20:30:00+00:00",
    )
    index = LibraryIndex().with_run_tags("ruth-mk02", RunTags("ruth", "anime"))
    library = scan_library(service, settings, index)

    assert next_version_name(library, "ruth", "anime") == "ruth-anime-mk03"
    assert next_version_name(library, "ruth", "photo") == "ruth-photo-mk08"
    assert next_version_name(library, "alex", "anime") == "alex-anime-mk01"


def test_next_version_name_skips_names_already_on_disk(tmp_path: Path) -> None:
    service, settings = _workspace(tmp_path)
    (tmp_path / settings.checkpoints_dir / "ruth-anime-mk01").mkdir(parents=True)
    library = scan_library(service, settings, LibraryIndex())

    assert next_version_name(library, "ruth", "anime") == "ruth-anime-mk02"
