"""Small atomic filesystem helpers for durable pipeline outputs."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def _temporary_path(destination: Path) -> tuple[int, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    return handle, Path(raw_path)


def atomic_write_bytes(destination: Path, data: bytes) -> None:
    """Replace destination only after all bytes are durable in a sibling file."""
    handle, temporary = _temporary_path(destination)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(handle)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(destination: Path, text: str) -> None:
    atomic_write_bytes(destination, text.encode("utf-8"))


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy a file without exposing a partial destination."""
    if not source.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")
    if source.resolve() == destination.resolve():
        return

    handle, temporary = _temporary_path(destination)
    os.close(handle)
    try:
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != source.stat().st_size:
            raise OSError(f"Incomplete copy from {source} to {destination}")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
