"""Durable ownership records for ComfyUI work submitted by MIVIM.

ComfyUI runs in a separate application process. If MIVIM is forcibly closed
after submitting a prompt, its OS run lease is released while ComfyUI can keep
using the GPU. A small app-owned reservation bridges that process boundary:
the next accelerator run verifies the recorded prompt has left ComfyUI's queue
before it proceeds.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

from .io_utils import atomic_write_text
from .run_guardian import accelerator_lease_path

_ACTIVITY_SCHEMA_VERSION = 1
_MARKER_SIZE_LIMIT = 16 * 1024
_QUEUE_RESPONSE_SIZE_LIMIT = 16 * 1024 * 1024
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
RUN_BUSY_MARKER = "MIVIM_RUN_BUSY"


class ComfyActivityBusyError(RuntimeError):
    """Raised when previously submitted ComfyUI work may still own the GPU."""


@dataclass(frozen=True, slots=True)
class _PromptRecord:
    schema_version: int
    host: str
    client_id: str
    prompt_id: str
    reserved_at: str

    @classmethod
    def from_json(cls, value: object) -> _PromptRecord | None:
        if not isinstance(value, Mapping):
            return None
        schema = value.get("schema_version")
        host = value.get("host")
        client_id = value.get("client_id")
        prompt_id = value.get("prompt_id")
        reserved_at = value.get("reserved_at")
        if (
            schema != _ACTIVITY_SCHEMA_VERSION
            or not isinstance(host, str)
            or not host
            or not isinstance(client_id, str)
            or not client_id
            or not isinstance(prompt_id, str)
            or not isinstance(reserved_at, str)
            or not reserved_at
        ):
            return None
        return cls(
            schema_version=schema,
            host=host,
            client_id=client_id,
            prompt_id=prompt_id,
            reserved_at=reserved_at,
        )


def _activity_directory(environ: Mapping[str, str] | None = None) -> Path:
    return accelerator_lease_path(environ).parent / "comfy-prompts"


def _normalised_local_host(host: str) -> str:
    candidate = host.strip()
    parsed = urlsplit(candidate if "://" in candidate else f"http://{candidate}")
    hostname = parsed.hostname.casefold() if parsed.hostname else ""
    port = parsed.port or 8188
    if (
        parsed.scheme.casefold() != "http"
        or hostname not in _LOCAL_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or not 1 <= port <= 65535
    ):
        raise ValueError("ComfyUI activity tracking requires a local server address.")
    display_host = f"[{hostname}]" if ":" in hostname else hostname
    return f"{display_host}:{port}"


def _marker_path(
    host: str,
    client_id: str,
    reservation_id: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    identity = hashlib.sha256(f"{host}\0{client_id}".encode()).hexdigest()[:24]
    return _activity_directory(environ) / f"prompt-{identity}-{reservation_id}.json"


def _write_record(path: Path, record: _PromptRecord) -> None:
    atomic_write_text(path, json.dumps(asdict(record), sort_keys=True) + "\n")


@dataclass(slots=True)
class ComfyPromptReservation:
    """One durable reservation created before a prompt is submitted."""

    path: Path
    record: _PromptRecord
    _closed: bool = False

    def attach_prompt(self, prompt_id: str) -> None:
        prompt_id = prompt_id.strip()
        if not prompt_id:
            raise ValueError("ComfyUI prompt_id cannot be empty.")
        if self._closed:
            raise RuntimeError("ComfyUI prompt reservation is already closed.")
        self.record = replace(self.record, prompt_id=prompt_id)
        _write_record(self.path, self.record)

    def clear(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            # A stale marker is intentionally fail-safe: the next run will
            # recheck ComfyUI and remove it once the prompt is confirmed gone.
            pass


def reserve_comfy_prompt(
    host: str,
    client_id: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> ComfyPromptReservation:
    """Persist a reservation before POST /prompt can create remote work."""

    client_id = client_id.strip()
    if not client_id:
        raise ValueError("ComfyUI client_id cannot be empty.")
    normalised_host = _normalised_local_host(host)
    record = _PromptRecord(
        schema_version=_ACTIVITY_SCHEMA_VERSION,
        host=normalised_host,
        client_id=client_id,
        prompt_id="",
        reserved_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    path = _marker_path(
        normalised_host,
        client_id,
        uuid.uuid4().hex,
        environ=environ,
    )
    try:
        _write_record(path, record)
    except OSError as exc:
        raise RuntimeError(
            f"MIVIM could not establish crash-safe ComfyUI prompt tracking: {exc}"
        ) from exc
    return ComfyPromptReservation(path=path, record=record)


def _queue_item_identity(item: object) -> tuple[str | None, str | None]:
    prompt_id: object = None
    client_id: object = None
    if isinstance(item, (list, tuple)):
        if len(item) >= 2:
            prompt_id = item[1]
        if len(item) >= 4 and isinstance(item[3], Mapping):
            client_id = item[3].get("client_id")
    elif isinstance(item, Mapping):
        prompt_id = item.get("prompt_id")
        client_id = item.get("client_id")
        for key in ("extra_data", "metadata"):
            nested = item.get(key)
            if client_id is None and isinstance(nested, Mapping):
                client_id = nested.get("client_id")
    return (
        prompt_id if isinstance(prompt_id, str) and prompt_id else None,
        client_id if isinstance(client_id, str) and client_id else None,
    )


def _queue_records(
    host: str, timeout: float
) -> tuple[tuple[str | None, str | None], ...]:
    request = urllib.request.Request(
        f"http://{host}/queue",
        headers={"Accept": "application/json", "User-Agent": "MIVIM-LoRA-Studio"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(_QUEUE_RESPONSE_SIZE_LIMIT + 1)
    except (HTTPError, OSError, URLError) as exc:
        raise ComfyActivityBusyError(
            f"{RUN_BUSY_MARKER}: MIVIM cannot yet confirm that an earlier "
            "ComfyUI job finished. Launch ComfyUI, then choose Retry so MIVIM "
            f"can check its queue. Technical detail: {exc}"
        ) from exc
    if len(body) > _QUEUE_RESPONSE_SIZE_LIMIT:
        raise ComfyActivityBusyError(
            f"{RUN_BUSY_MARKER}: ComfyUI returned an unexpectedly large queue "
            "status. Restart ComfyUI, then choose Retry."
        )
    try:
        value: Any = json.loads(body)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ComfyActivityBusyError(
            f"{RUN_BUSY_MARKER}: ComfyUI returned an unreadable queue status. "
            "Restart ComfyUI, then choose Retry."
        ) from exc
    if not isinstance(value, Mapping):
        raise ComfyActivityBusyError(
            f"{RUN_BUSY_MARKER}: ComfyUI returned an invalid queue status. "
            "Restart ComfyUI, then choose Retry."
        )
    if "queue_running" not in value or "queue_pending" not in value:
        raise ComfyActivityBusyError(
            f"{RUN_BUSY_MARKER}: ComfyUI returned an incomplete queue status. "
            "Restart ComfyUI, then choose Retry."
        )
    running = value.get("queue_running")
    pending = value.get("queue_pending")
    if not isinstance(running, list) or not isinstance(pending, list):
        raise ComfyActivityBusyError(
            f"{RUN_BUSY_MARKER}: ComfyUI returned an invalid queue status. "
            "Restart ComfyUI, then choose Retry."
        )
    identities = tuple(_queue_item_identity(item) for item in (*running, *pending))
    if any(prompt_id is None for prompt_id, _owner in identities):
        raise ComfyActivityBusyError(
            f"{RUN_BUSY_MARKER}: ComfyUI returned a queue item MIVIM could "
            "not identify safely. Finish or clear the ComfyUI queue, then "
            "choose Retry."
        )
    return identities


def check_orphaned_comfy_prompts(
    *,
    environ: Mapping[str, str] | None = None,
    timeout: float = 3.0,
) -> None:
    """Block accelerator work until every durable MIVIM prompt has left its queue."""

    directory = _activity_directory(environ)
    try:
        markers = tuple(directory.glob("prompt-*.json"))
    except OSError as exc:
        raise ComfyActivityBusyError(
            f"{RUN_BUSY_MARKER}: MIVIM could not inspect its ComfyUI recovery "
            f"records. Technical detail: {exc}"
        ) from exc
    for marker in markers:
        try:
            if marker.is_symlink() or marker.stat().st_size > _MARKER_SIZE_LIMIT:
                try:
                    marker.unlink(missing_ok=True)
                except OSError as exc:
                    raise ComfyActivityBusyError(
                        f"{RUN_BUSY_MARKER}: MIVIM could not remove an unsafe "
                        "ComfyUI recovery record. Close programs using the "
                        f"MIVIM data folder, then choose Retry. Technical detail: {exc}"
                    ) from exc
                continue
            raw_record = marker.read_text(encoding="utf-8")
        except OSError as exc:
            raise ComfyActivityBusyError(
                f"{RUN_BUSY_MARKER}: MIVIM could not read a ComfyUI recovery "
                "record. Close programs using the MIVIM data folder, then "
                f"choose Retry. Technical detail: {exc}"
            ) from exc
        except UnicodeError:
            try:
                marker.unlink(missing_ok=True)
            except OSError as exc:
                raise ComfyActivityBusyError(
                    f"{RUN_BUSY_MARKER}: MIVIM found an unreadable ComfyUI "
                    "recovery record but could not remove it. Close programs "
                    "using the MIVIM data folder, then choose Retry. "
                    f"Technical detail: {exc}"
                ) from exc
            continue
        try:
            record = _PromptRecord.from_json(json.loads(raw_record))
        except (UnicodeError, json.JSONDecodeError):
            record = None
        if record is None:
            try:
                marker.unlink(missing_ok=True)
            except OSError as exc:
                raise ComfyActivityBusyError(
                    f"{RUN_BUSY_MARKER}: MIVIM found an invalid ComfyUI "
                    "recovery record but could not remove it. Close programs "
                    "using the MIVIM data folder, then choose Retry. "
                    f"Technical detail: {exc}"
                ) from exc
            continue
        try:
            verified_host = _normalised_local_host(record.host)
        except ValueError:
            try:
                marker.unlink(missing_ok=True)
            except OSError as exc:
                raise ComfyActivityBusyError(
                    f"{RUN_BUSY_MARKER}: MIVIM found an unsafe ComfyUI "
                    "recovery address but could not remove its record. Close "
                    "programs using the MIVIM data folder, then choose Retry. "
                    f"Technical detail: {exc}"
                ) from exc
            continue
        record = replace(record, host=verified_host)
        queued = _queue_records(record.host, timeout)
        if not record.prompt_id and any(owner is None for _prompt_id, owner in queued):
            raise ComfyActivityBusyError(
                f"{RUN_BUSY_MARKER}: MIVIM cannot safely identify every "
                "ComfyUI queue owner while recovering a just-submitted job. "
                "Finish or clear the ComfyUI queue, then choose Retry."
            )
        is_active = (
            any(prompt_id == record.prompt_id for prompt_id, _owner in queued)
            if record.prompt_id
            else any(owner == record.client_id for _prompt_id, owner in queued)
        )
        if is_active:
            prompt_label = record.prompt_id or "a just-submitted prompt"
            raise ComfyActivityBusyError(
                f"{RUN_BUSY_MARKER}: ComfyUI is still finishing MIVIM job "
                f"{prompt_label}. Wait for it to finish or stop that job in "
                "ComfyUI, then choose Retry. No new work was started."
            )
        try:
            marker.unlink(missing_ok=True)
        except OSError as exc:
            raise ComfyActivityBusyError(
                f"{RUN_BUSY_MARKER}: The earlier ComfyUI job finished, but "
                f"MIVIM could not clear its recovery record. Technical detail: {exc}"
            ) from exc
