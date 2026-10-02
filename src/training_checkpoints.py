"""Discover a project's saved weights and matching ai-toolkit preview images."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import config
from .io_utils import artifact_fingerprint


@dataclass(frozen=True, slots=True)
class TrainingCheckpoint:
    path: Path
    step: int
    final: bool
    fingerprint: str
    previews: tuple[tuple[int, Path], ...]

    @property
    def label(self) -> str:
        return f"Step {self.step:,}" + (" · final save" if self.final else "")


def checkpoint_step(path: Path) -> int:
    """Read bounded JSON metadata, without loading tensors or importing torch."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        length = int.from_bytes(prefix, "little")
        if len(prefix) != 8 or not 2 <= length <= 4 * 1024 * 1024:
            raise ValueError("Invalid checkpoint metadata length.")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise TypeError("Checkpoint metadata is not an object.")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint training information is missing.")
    info = json.loads(metadata.get("training_info", "{}"))
    step = info.get("step") if isinstance(info, dict) else None
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("Checkpoint training step is missing.")
    return step


def discover_training_checkpoints(
    root: Path, project_name: str
) -> tuple[TrainingCheckpoint, ...]:
    """Read only the named training folder; never borrow another run's samples."""
    config.validate_project_name(project_name)
    folder = (root / project_name).resolve()
    if not folder.is_relative_to(root.resolve()) or not folder.is_dir():
        return ()
    pattern = re.compile(rf"^{re.escape(project_name)}(?:_(\d+))?\.safetensors$")
    sample_pattern = re.compile(r"^\d+__(\d+)_(\d+)\.(?:jpg|jpeg|png|webp)$")
    samples: dict[int, dict[int, Path]] = {}
    for path in sorted((folder / "samples").glob("*")):
        match = sample_pattern.fullmatch(path.name)
        if match and path.is_file() and path.resolve().is_relative_to(folder):
            samples.setdefault(int(match[1]), {})[int(match[2])] = path
    choices: list[TrainingCheckpoint] = []
    for path in folder.glob("*.safetensors"):
        match = pattern.fullmatch(path.name)
        if not match or not path.resolve().is_relative_to(folder):
            continue
        try:
            before = path.stat()
            step = checkpoint_step(path)
            # A renamed file must not acquire another checkpoint's previews.
            if match[1] and int(match[1]) != step:
                continue
            fingerprint = artifact_fingerprint(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                continue
        except (OSError, ValueError, TypeError):
            continue
        choices.append(
            TrainingCheckpoint(
                path.resolve(),
                step,
                match[1] is None,
                fingerprint,
                tuple(sorted(samples.get(step, {}).items())),
            )
        )
    return tuple(sorted(choices, key=lambda item: (item.step, item.final)))
