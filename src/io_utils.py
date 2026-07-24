"""Small atomic filesystem helpers for durable pipeline outputs."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Protocol

_HASH_CHUNK_SIZE = 1024 * 1024


class _HashWriter(Protocol):
    def update(self, data: bytes, /) -> object: ...


def _hash_file_into(digest: _HashWriter, path: Path) -> None:
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)


def artifact_fingerprint(path: Path) -> str:
    """Return a content fingerprint for one result file or directory tree.

    Directory fingerprints include every relative filename and file body. Symlinks
    are rejected so a persisted result cannot silently point outside its recorded
    artifact.
    """

    if path.is_symlink():
        raise ValueError(f"Result artifacts cannot be symbolic links: {path}")
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(b"mivim-file-v1\0")
        _hash_file_into(digest, path)
        return f"sha256:{digest.hexdigest()}"
    if not path.is_dir():
        raise FileNotFoundError(f"Result artifact not found: {path}")

    digest.update(b"mivim-tree-v1\0")
    files = sorted(
        (entry for entry in path.rglob("*") if entry.is_file()),
        key=lambda entry: entry.relative_to(path).as_posix(),
    )
    for entry in files:
        if entry.is_symlink():
            raise ValueError(f"Result artifacts cannot contain symbolic links: {entry}")
        relative = entry.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(entry.stat().st_size.to_bytes(8, "big"))
        _hash_file_into(digest, entry)
    return f"sha256:{digest.hexdigest()}"


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
