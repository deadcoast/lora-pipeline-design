from __future__ import annotations

import json
from pathlib import Path
from typing import Self
from urllib.error import URLError

import pytest

from src import comfy_activity


class _Response:
    def __init__(self, value: object) -> None:
        self.body = json.dumps(value).encode()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


def _environment(tmp_path: Path) -> dict[str, str]:
    return {"MIVIM_RUN_GUARD_DIR": str(tmp_path / "guard")}


def test_active_owned_prompt_blocks_the_next_accelerator_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environ = _environment(tmp_path)
    reservation = comfy_activity.reserve_comfy_prompt(
        "127.0.0.1:8188",
        "client-a",
        environ=environ,
    )
    reservation.attach_prompt("prompt-a")
    monkeypatch.setattr(
        comfy_activity.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response(
            {
                "queue_running": [[1, "prompt-a", {}, {"client_id": "client-a"}, []]],
                "queue_pending": [],
            }
        ),
    )

    with pytest.raises(
        comfy_activity.ComfyActivityBusyError,
        match="still finishing",
    ):
        comfy_activity.check_orphaned_comfy_prompts(environ=environ)

    assert reservation.path.is_file()


def test_finished_prompt_marker_is_cleared_automatically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environ = _environment(tmp_path)
    reservation = comfy_activity.reserve_comfy_prompt(
        "localhost:8188",
        "client-b",
        environ=environ,
    )
    reservation.attach_prompt("prompt-b")
    monkeypatch.setattr(
        comfy_activity.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response({"queue_running": [], "queue_pending": []}),
    )

    comfy_activity.check_orphaned_comfy_prompts(environ=environ)

    assert not reservation.path.exists()


def test_same_client_reservations_never_overwrite_or_clear_each_other(
    tmp_path: Path,
) -> None:
    environ = _environment(tmp_path)
    older = comfy_activity.reserve_comfy_prompt(
        "localhost:8188",
        "shared-client",
        environ=environ,
    )
    older.attach_prompt("older-prompt")
    newer = comfy_activity.reserve_comfy_prompt(
        "localhost:8188",
        "shared-client",
        environ=environ,
    )
    newer.attach_prompt("newer-prompt")

    assert older.path != newer.path
    assert older.path.is_file()
    assert newer.path.is_file()

    older.clear()

    assert not older.path.exists()
    assert newer.path.is_file()


def test_large_valid_queue_status_is_checked_without_false_recovery_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environ = _environment(tmp_path)
    reservation = comfy_activity.reserve_comfy_prompt(
        "localhost:8188",
        "client-finished",
        environ=environ,
    )
    reservation.attach_prompt("prompt-finished")
    monkeypatch.setattr(
        comfy_activity.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response(
            {
                "queue_running": [],
                "queue_pending": [
                    [
                        1,
                        "unrelated-prompt",
                        {"workflow": "x" * 20_000},
                        {"client_id": "another-client"},
                        [],
                    ]
                ],
            }
        ),
    )

    comfy_activity.check_orphaned_comfy_prompts(environ=environ)

    assert not reservation.path.exists()


def test_unidentifiable_queue_item_keeps_recovery_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environ = _environment(tmp_path)
    reservation = comfy_activity.reserve_comfy_prompt(
        "127.0.0.1:8188",
        "client-malformed",
        environ=environ,
    )
    reservation.attach_prompt("prompt-malformed")
    monkeypatch.setattr(
        comfy_activity.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response(
            {
                "queue_running": [],
                "queue_pending": [{"client_id": "client-malformed"}],
            }
        ),
    )

    with pytest.raises(
        comfy_activity.ComfyActivityBusyError,
        match="could not identify safely",
    ):
        comfy_activity.check_orphaned_comfy_prompts(environ=environ)

    assert reservation.path.is_file()


def test_incomplete_queue_schema_keeps_recovery_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environ = _environment(tmp_path)
    reservation = comfy_activity.reserve_comfy_prompt(
        "127.0.0.1:8188",
        "client-incomplete",
        environ=environ,
    )
    reservation.attach_prompt("prompt-incomplete")
    monkeypatch.setattr(
        comfy_activity.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response({}),
    )

    with pytest.raises(
        comfy_activity.ComfyActivityBusyError,
        match="incomplete queue status",
    ):
        comfy_activity.check_orphaned_comfy_prompts(environ=environ)

    assert reservation.path.is_file()


def test_unreadable_recovery_marker_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environ = _environment(tmp_path)
    reservation = comfy_activity.reserve_comfy_prompt(
        "127.0.0.1:8188",
        "client-locked",
        environ=environ,
    )
    reservation.attach_prompt("prompt-locked")
    original_read_text = Path.read_text

    def locked_read(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if path == reservation.path:
            raise PermissionError("file is locked")
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", locked_read)

    with pytest.raises(
        comfy_activity.ComfyActivityBusyError,
        match="could not read a ComfyUI recovery record",
    ):
        comfy_activity.check_orphaned_comfy_prompts(environ=environ)

    assert reservation.path.is_file()


def test_interrupted_submission_requires_identifiable_queue_owners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environ = _environment(tmp_path)
    reservation = comfy_activity.reserve_comfy_prompt(
        "127.0.0.1:8188",
        "client-before-owner-response",
        environ=environ,
    )
    monkeypatch.setattr(
        comfy_activity.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response(
            {
                "queue_running": [],
                "queue_pending": [[1, "prompt-without-owner", {}, {}, []]],
            }
        ),
    )

    with pytest.raises(
        comfy_activity.ComfyActivityBusyError,
        match="identify every ComfyUI queue owner",
    ):
        comfy_activity.check_orphaned_comfy_prompts(environ=environ)

    assert reservation.path.is_file()


def test_interrupted_submission_is_recovered_by_client_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environ = _environment(tmp_path)
    reservation = comfy_activity.reserve_comfy_prompt(
        "127.0.0.1:8188",
        "client-before-response",
        environ=environ,
    )
    monkeypatch.setattr(
        comfy_activity.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response(
            {
                "queue_running": [],
                "queue_pending": [
                    [
                        2,
                        "unknown-prompt",
                        {},
                        {"client_id": "client-before-response"},
                        [],
                    ]
                ],
            }
        ),
    )

    with pytest.raises(comfy_activity.ComfyActivityBusyError):
        comfy_activity.check_orphaned_comfy_prompts(environ=environ)

    assert reservation.path.is_file()


def test_unreachable_comfy_keeps_recovery_marker_and_gives_fix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environ = _environment(tmp_path)
    reservation = comfy_activity.reserve_comfy_prompt(
        "127.0.0.1:8188",
        "client-c",
        environ=environ,
    )
    reservation.attach_prompt("prompt-c")

    def unavailable(_request: object, timeout: float) -> _Response:
        raise URLError("connection refused")

    monkeypatch.setattr(comfy_activity.urllib.request, "urlopen", unavailable)

    with pytest.raises(
        comfy_activity.ComfyActivityBusyError,
        match="Launch ComfyUI",
    ):
        comfy_activity.check_orphaned_comfy_prompts(environ=environ)

    assert reservation.path.is_file()


def test_tampered_remote_host_marker_is_discarded_without_network_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environ = _environment(tmp_path)
    marker_dir = tmp_path / "guard" / "comfy-prompts"
    marker_dir.mkdir(parents=True)
    marker = marker_dir / "prompt-tampered.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "host": "example.com:8188",
                "client_id": "client",
                "prompt_id": "prompt",
                "reserved_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    def unexpected_network(*_args: object, **_kwargs: object) -> _Response:
        pytest.fail("A tampered activity marker must never trigger remote access.")

    monkeypatch.setattr(
        comfy_activity.urllib.request,
        "urlopen",
        unexpected_network,
    )

    comfy_activity.check_orphaned_comfy_prompts(environ=environ)

    assert not marker.exists()
