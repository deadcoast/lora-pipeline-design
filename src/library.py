"""Browse every trained model by character and art style, from what is on disk.

The library never claims a run exists: entries come from the result and
recovery records the pipeline already writes, paired with the files they
name. ``library.json`` only adds the tag vocabulary, a style per base
checkpoint, and tag overrides for runs that predate the tag fields.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import yaml

from . import config
from .app_models import AppSettings, RunResult, TrainingPreset
from .io_utils import atomic_write_text
from .model_profiles import is_sdxl_base_checkpoint
from .tags import normalize_tag, tag_error

if TYPE_CHECKING:
    from .app_service import ApplicationService

__all__ = [
    "Library",
    "LibraryEntry",
    "LibraryIndex",
    "LibraryIndexError",
    "LibraryStatus",
    "RunTags",
    "checkpoint_style",
    "library_index_path",
    "load_library_index",
    "next_version_name",
    "normalize_tag",
    "profile_for_checkpoint",
    "save_library_index",
    "scan_library",
    "tag_error",
]

LIBRARY_INDEX_SCHEMA_VERSION = 1
LIBRARY_INDEX_NAME = "library.json"
_MAX_INDEX_BYTES = 1024 * 1024
_MAX_RUN_CONFIG_BYTES = 1024 * 1024
_VERSION_SUFFIX = re.compile(r"mk(\d+)$")

LibraryStatus = Literal[
    "ready",
    "lora_missing",
    "result_missing",
    "validation_pending",
    "prepared",
    "incomplete",
]


class LibraryIndexError(ValueError):
    """``library.json`` exists but cannot be trusted."""


def _checked_tag(text: str) -> str:
    tag = normalize_tag(text)
    issue = tag_error(tag)
    if issue is not None:
        raise ValueError(issue)
    return tag


@dataclass(frozen=True, slots=True)
class RunTags:
    character: str
    artstyle: str


@dataclass(frozen=True, slots=True)
class LibraryIndex:
    """User vocabulary and overrides; everything else is derived from disk."""

    characters: tuple[str, ...] = ()
    styles: tuple[str, ...] = ()
    checkpoint_styles: dict[str, str] = field(default_factory=dict)
    run_tags: dict[str, RunTags] = field(default_factory=dict)

    def with_character(self, text: str) -> LibraryIndex:
        tag = _checked_tag(text)
        if tag in self.characters:
            return self
        return replace(self, characters=tuple(sorted((*self.characters, tag))))

    def with_style(self, text: str) -> LibraryIndex:
        tag = _checked_tag(text)
        if tag in self.styles:
            return self
        return replace(self, styles=tuple(sorted((*self.styles, tag))))

    def with_checkpoint_style(self, checkpoint: str, text: str) -> LibraryIndex:
        tag = _checked_tag(text)
        updated = dict(self.checkpoint_styles)
        updated[checkpoint] = tag
        return replace(self.with_style(tag), checkpoint_styles=updated)

    def with_run_tags(self, project_name: str, tags: RunTags) -> LibraryIndex:
        checked = RunTags(_checked_tag(tags.character), _checked_tag(tags.artstyle))
        updated = dict(self.run_tags)
        updated[project_name] = checked
        return replace(
            self.with_character(checked.character).with_style(checked.artstyle),
            run_tags=updated,
        )


def library_index_path(service: ApplicationService, settings: AppSettings) -> Path:
    """The index lives in the workspace root, beside the datasets folder."""
    return service.project_path(settings.datasets_dir).parent / LIBRARY_INDEX_NAME


def _string_list(value: object, what: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise LibraryIndexError(f"{what} is not a list of words.")
    tags: list[str] = []
    for item in cast(list[str], value):
        tag = normalize_tag(item)
        if tag_error(tag) is None and tag not in tags:
            tags.append(tag)
    return tuple(sorted(tags))


def load_library_index(path: Path) -> LibraryIndex:
    """Read the index, treating a missing file as empty and a bad one as an error."""
    try:
        if path.stat().st_size > _MAX_INDEX_BYTES:
            raise LibraryIndexError("library.json is unexpectedly large.")
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return LibraryIndex()
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LibraryIndexError(f"library.json could not be read: {exc}") from exc
    if not isinstance(value, dict):
        raise LibraryIndexError("library.json is not an object.")
    if value.get("schema_version") != LIBRARY_INDEX_SCHEMA_VERSION:
        raise LibraryIndexError("library.json has an unknown schema version.")
    characters = _string_list(value.get("characters", []), "characters")
    styles = _string_list(value.get("styles", []), "styles")
    raw_checkpoints = value.get("checkpoint_styles", {})
    raw_runs = value.get("run_tags", {})
    if not isinstance(raw_checkpoints, dict) or not isinstance(raw_runs, dict):
        raise LibraryIndexError("library.json tag maps are malformed.")
    checkpoint_styles: dict[str, str] = {}
    for name, style in raw_checkpoints.items():
        if not isinstance(name, str) or not isinstance(style, str):
            raise LibraryIndexError("library.json checkpoint styles are malformed.")
        tag = normalize_tag(style)
        if tag_error(tag) is None:
            checkpoint_styles[name] = tag
    run_tags: dict[str, RunTags] = {}
    for name, tags in raw_runs.items():
        if (
            not isinstance(name, str)
            or not isinstance(tags, dict)
            or not isinstance(tags.get("character"), str)
            or not isinstance(tags.get("artstyle"), str)
        ):
            raise LibraryIndexError("library.json run tags are malformed.")
        character = normalize_tag(cast(str, tags["character"]))
        artstyle = normalize_tag(cast(str, tags["artstyle"]))
        if tag_error(character) is None and tag_error(artstyle) is None:
            run_tags[name] = RunTags(character, artstyle)
    index = LibraryIndex(characters, styles, checkpoint_styles, run_tags)
    for style in checkpoint_styles.values():
        index = index.with_style(style)
    for tags in run_tags.values():
        index = index.with_character(tags.character).with_style(tags.artstyle)
    return index


def save_library_index(path: Path, index: LibraryIndex) -> None:
    payload = {
        "schema_version": LIBRARY_INDEX_SCHEMA_VERSION,
        "characters": list(index.characters),
        "styles": list(index.styles),
        "checkpoint_styles": dict(sorted(index.checkpoint_styles.items())),
        "run_tags": {
            name: {"character": tags.character, "artstyle": tags.artstyle}
            for name, tags in sorted(index.run_tags.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


# ------------------------------------------------------------ checkpoints


def _plain_checkpoint_name(checkpoint: str) -> str:
    return (
        Path(checkpoint.replace("\\", "/"))
        .name.lower()
        .replace("-", "")
        .replace("_", "")
    )


def checkpoint_style(
    checkpoint: str, index: LibraryIndex, *, path: Path | None = None
) -> str:
    """The user's label wins; otherwise recognize the model from its name or file."""
    labelled = index.checkpoint_styles.get(checkpoint, "")
    if labelled:
        return labelled
    if "noob" in _plain_checkpoint_name(checkpoint):
        return "anime"
    if path is not None and is_sdxl_base_checkpoint(path):
        return "photo"
    return ""


def profile_for_checkpoint(checkpoint: str) -> str:
    """Pick the model profile that the checkpoint filename calls for."""
    name = _plain_checkpoint_name(checkpoint)
    if "noob" in name:
        return "noobai-xl-vpred" if "vpred" in name else "noobai-xl-epsilon"
    return "sdxl"


# ------------------------------------------------------------------ scan


@dataclass(frozen=True, slots=True)
class LibraryEntry:
    project_name: str
    character: str
    artstyle: str
    status: LibraryStatus
    model_profile: str
    checkpoint: str
    trigger: str
    preset: str
    steps: int | None
    completed_at: str
    grid_path: Path | None
    lora_installed: bool
    source_dir: Path | None
    record: RunResult | None

    @property
    def tagged(self) -> bool:
        return bool(self.character and self.artstyle)


@dataclass(frozen=True, slots=True)
class Library:
    entries: tuple[LibraryEntry, ...]

    def characters(self) -> tuple[tuple[str, tuple[LibraryEntry, ...]], ...]:
        """Tagged runs grouped by character, most recently active group first."""
        groups: dict[str, list[LibraryEntry]] = {}
        for entry in self.entries:
            if entry.tagged:
                groups.setdefault(entry.character, []).append(entry)
        ordered = sorted(
            groups.items(),
            key=lambda item: item[1][0].completed_at,
            reverse=True,
        )
        return tuple((name, tuple(entries)) for name, entries in ordered)

    def unsorted(self) -> tuple[LibraryEntry, ...]:
        return tuple(entry for entry in self.entries if not entry.tagged)

    def styles(self) -> tuple[str, ...]:
        return tuple(
            sorted({entry.artstyle for entry in self.entries if entry.artstyle})
        )

    def names(self) -> frozenset[str]:
        return frozenset(entry.project_name for entry in self.entries)


def _candidate_names(service: ApplicationService, settings: AppSettings) -> set[str]:
    names: set[str] = set()
    configs = service.project_path(settings.run_configs_dir)
    for suffix in ("-result.json", "-recovery.json"):
        for path in configs.glob(f"*{suffix}"):
            if path.is_file():
                names.add(path.name[: -len(suffix)])
    for folder in (settings.checkpoints_dir, settings.datasets_dir):
        root = service.project_path(folder)
        if root.is_dir():
            names.update(child.name for child in root.iterdir() if child.is_dir())
    valid: set[str] = set()
    for name in names:
        try:
            valid.add(config.validate_project_name(name))
        except ValueError:
            continue
    return valid


def _steps_from_run_config(path: Path) -> int | None:
    try:
        if path.stat().st_size > _MAX_RUN_CONFIG_BYTES:
            return None
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        steps = loaded["config"]["process"][0]["train"]["steps"]
    except (OSError, UnicodeError, yaml.YAMLError, KeyError, IndexError, TypeError):
        return None
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        return None
    return steps


def _record_from_outcome(
    name: str, outcome: object, mode: Literal["prepare", "full"]
) -> RunResult | None:
    from .app_service import PipelineOutcome

    if not isinstance(outcome, PipelineOutcome):
        return None
    try:
        preset = TrainingPreset(outcome.preset)
    except ValueError:
        return None
    return RunResult(
        project_name=name,
        training_project_name=outcome.training_project_name,
        mode=mode,
        result_path=outcome.result_path,
        completed_at=outcome.completed_at,
        trigger=outcome.trigger,
        preset=preset,
        checkpoint=outcome.checkpoint,
        model_profile=outcome.model_profile,
        lora_name=outcome.lora_name,
        result_fingerprint=outcome.result_fingerprint,
        workflow_path=outcome.workflow_path,
        workflow_fingerprint=outcome.workflow_fingerprint,
        comfy_installation_id=outcome.comfy_installation_id,
        comfy_host=outcome.comfy_host,
        installed_lora_path=outcome.installed_lora_path,
        lora_fingerprint=outcome.lora_fingerprint,
        checkpoint_path=outcome.checkpoint_path,
        checkpoint_fingerprint=outcome.checkpoint_fingerprint,
        starting_strength=outcome.starting_strength,
        character=outcome.character,
        artstyle=outcome.artstyle,
        source_dir=outcome.source_dir,
    )


def _entry(
    service: ApplicationService,
    settings: AppSettings,
    index: LibraryIndex,
    name: str,
) -> LibraryEntry:
    outcome = service.read_pipeline_outcome(settings, name)
    recovery = service.read_validation_recovery(settings, name)
    status: LibraryStatus = "incomplete"
    record: RunResult | None = None
    grid_path: Path | None = None
    lora_installed = False
    character = artstyle = ""
    model_profile = checkpoint = trigger = preset = completed_at = ""
    source_dir: Path | None = None
    if outcome is not None:
        character, artstyle = outcome.character, outcome.artstyle
        model_profile = outcome.model_profile
        checkpoint = outcome.checkpoint
        trigger = outcome.trigger
        preset = outcome.preset
        completed_at = outcome.completed_at
        source_dir = outcome.source_dir
        if outcome.mode == "prepare":
            status = "prepared"
            record = _record_from_outcome(name, outcome, "prepare")
        else:
            record = _record_from_outcome(name, outcome, "full")
            grid = service.project_path(outcome.result_path)
            lora_path = outcome.installed_lora_path
            if lora_path is None and outcome.lora_name and settings.comfy_lora_dir:
                lora_path = (
                    service.project_path(settings.comfy_lora_dir) / outcome.lora_name
                )
            lora_installed = lora_path is not None and lora_path.is_file()
            if grid.is_file():
                grid_path = grid
                status = "ready" if lora_installed else "lora_missing"
            else:
                status = "result_missing"
    elif recovery is not None and recovery.state == "validation_pending":
        status = "validation_pending"
        model_profile = recovery.model_profile
        checkpoint = recovery.checkpoint
        trigger = recovery.trigger
        preset = recovery.preset.value
        completed_at = recovery.updated_at
        lora_installed = recovery.installed_lora_path.is_file()
    override = index.run_tags.get(name)
    if override is not None:
        character, artstyle = override.character, override.artstyle
    steps = _steps_from_run_config(
        service.project_path(settings.run_configs_dir) / f"{name}.yaml"
    )
    return LibraryEntry(
        project_name=name,
        character=character,
        artstyle=artstyle,
        status=status,
        model_profile=model_profile,
        checkpoint=checkpoint,
        trigger=trigger,
        preset=preset,
        steps=steps,
        completed_at=completed_at,
        grid_path=grid_path,
        lora_installed=lora_installed,
        source_dir=source_dir,
        record=record,
    )


def scan_library(
    service: ApplicationService, settings: AppSettings, index: LibraryIndex
) -> Library:
    """Build the library from small records and file existence only; no hashing."""
    entries = [
        _entry(service, settings, index, name)
        for name in sorted(_candidate_names(service, settings))
    ]
    entries.sort(
        key=lambda entry: (entry.completed_at, entry.project_name), reverse=True
    )
    return Library(tuple(entries))


def next_version_name(library: Library, character: str, artstyle: str) -> str:
    """Continue the character's style group: ruth + anime after ruth-mk02 is ruth-anime-mk03."""
    character = _checked_tag(character)
    artstyle = _checked_tag(artstyle)
    highest = 0
    for entry in library.entries:
        if entry.character != character or entry.artstyle != artstyle:
            continue
        match = _VERSION_SUFFIX.search(entry.project_name)
        if match:
            highest = max(highest, int(match.group(1)))
    taken = library.names()
    number = highest + 1
    while (name := f"{character}-{artstyle}-mk{number:02d}") in taken:
        number += 1
    return name


def vocabulary_from(library: Library, index: LibraryIndex) -> LibraryIndex:
    """Fold every tag already used by a run into the index so autocomplete sees it."""
    for entry in library.entries:
        if entry.tagged:
            index = index.with_character(entry.character).with_style(entry.artstyle)
    return index


def entries_matching(
    entries: Iterable[LibraryEntry], *, style: str = "", search: str = ""
) -> tuple[LibraryEntry, ...]:
    """Filter by one style chip and a free-text search over name, character, trigger."""
    needle = search.strip().lower()
    return tuple(
        entry
        for entry in entries
        if (not style or entry.artstyle == style)
        and (
            not needle
            or needle in entry.project_name.lower()
            or needle in entry.character.lower()
            or needle in entry.trigger.lower()
        )
    )
