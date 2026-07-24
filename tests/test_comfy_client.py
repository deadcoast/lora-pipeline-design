from __future__ import annotations

import json
from typing import Any

import pytest

from src.comfy_client import ComfyClient


def _queue_item(prompt_id: str, client_id: str | None) -> list[object]:
    extra_data = {} if client_id is None else {"client_id": client_id}
    return [1, prompt_id, {}, extra_data, []]


def test_cancel_owned_pending_prompt_only_deletes_that_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ComfyClient()
    writes: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        client,
        "_http_json",
        lambda _path: {
            "queue_running": [],
            "queue_pending": [_queue_item("ours", client.client_id)],
        },
    )
    monkeypatch.setattr(
        client, "_post_json", lambda path, payload: writes.append((path, payload))
    )

    result = client.cancel_owned_prompt("ours")

    assert result.status == "pending_delete_requested"
    assert result.requested
    assert writes == [("/queue", {"delete": ["ours"]})]


def test_cancel_owned_running_prompt_uses_targeted_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ComfyClient()
    worker_client_id = "worker-client"
    writes: list[tuple[str, dict[str, Any]]] = []

    def fake_http(path: str) -> dict[str, Any]:
        if path == "/system_stats":
            return {"system": {"comfyui_version": "0.3.56"}}
        return {
            "queue_running": [_queue_item("ours", worker_client_id)],
            "queue_pending": [],
        }

    monkeypatch.setattr(client, "_http_json", fake_http)
    monkeypatch.setattr(
        client, "_post_json", lambda path, payload: writes.append((path, payload))
    )

    result = client.cancel_owned_prompt("ours", client_id=worker_client_id)

    assert result.status == "running_interrupt_requested"
    assert result.requested
    assert writes == [("/interrupt", {"prompt_id": "ours"})]


@pytest.mark.parametrize(
    "stats",
    [
        {"system": {"comfyui_version": "0.3.55"}},
        {"system": {"comfyui_version": "not-a-version"}},
        {},
    ],
)
def test_cancel_never_interrupts_running_prompt_on_older_or_unknown_comfyui(
    monkeypatch: pytest.MonkeyPatch,
    stats: dict[str, Any],
) -> None:
    client = ComfyClient()
    writes: list[tuple[str, dict[str, Any]]] = []

    def fake_http(path: str) -> dict[str, Any]:
        if path == "/system_stats":
            return stats
        return {
            "queue_running": [_queue_item("ours", client.client_id)],
            "queue_pending": [],
        }

    monkeypatch.setattr(client, "_http_json", fake_http)
    monkeypatch.setattr(
        client, "_post_json", lambda path, payload: writes.append((path, payload))
    )

    result = client.cancel_owned_prompt("ours")

    assert result.status == "running_interrupt_unsupported"
    assert not result.requested
    assert "left it running" in result.message
    assert writes == []


def test_cancel_never_interrupts_when_comfyui_version_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ComfyClient()
    writes: list[tuple[str, dict[str, Any]]] = []

    def fake_http(path: str) -> dict[str, Any]:
        if path == "/system_stats":
            raise RuntimeError("version probe failed")
        return {
            "queue_running": [_queue_item("ours", client.client_id)],
            "queue_pending": [],
        }

    monkeypatch.setattr(client, "_http_json", fake_http)
    monkeypatch.setattr(
        client, "_post_json", lambda path, payload: writes.append((path, payload))
    )

    result = client.cancel_owned_prompt("ours")

    assert result.status == "running_interrupt_unsupported"
    assert "unknown version" in result.message
    assert writes == []


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.3.55", False),
        ("0.3.56", True),
        ("v0.3.56+desktop", True),
        ("0.4.0-dev", True),
        ("1.0.0", True),
    ],
)
def test_targeted_interrupt_version_boundary(version: str, expected: bool) -> None:
    parsed = ComfyClient._comfyui_version({"system": {"comfyui_version": version}})

    assert (parsed is not None and parsed >= (0, 3, 56)) is expected


@pytest.mark.parametrize("owner", [None, "another-client"])
def test_cancel_refuses_prompt_without_matching_owner(
    monkeypatch: pytest.MonkeyPatch, owner: str | None
) -> None:
    client = ComfyClient()
    writes: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        client,
        "_http_json",
        lambda _path: {
            "queue_running": [_queue_item("target", owner)],
            "queue_pending": [],
        },
    )
    monkeypatch.setattr(
        client, "_post_json", lambda path, payload: writes.append((path, payload))
    )

    result = client.cancel_owned_prompt("target")

    assert result.status == "ownership_unverified"
    assert not result.requested
    assert "No ComfyUI work was stopped" in result.message
    assert writes == []


def test_cancel_does_nothing_when_prompt_has_left_the_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ComfyClient()
    writes: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        client,
        "_http_json",
        lambda _path: {"queue_running": [], "queue_pending": []},
    )
    monkeypatch.setattr(
        client, "_post_json", lambda path, payload: writes.append((path, payload))
    )

    result = client.cancel_owned_prompt("finished")

    assert result.status == "not_found"
    assert not result.requested
    assert writes == []


def test_cancel_rejects_an_explicit_empty_worker_client_id() -> None:
    client = ComfyClient()

    with pytest.raises(ValueError, match="client_id"):
        client.cancel_owned_prompt("prompt", client_id="")


def test_run_prints_owned_prompt_marker(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = ComfyClient(host="127.0.0.1:8188")
    prompt_id = "prompt-123"
    monkeypatch.setattr(client, "queue_prompt", lambda _graph: prompt_id)

    class FakeWebSocket:
        def connect(self, _url: str, timeout: float) -> None:
            assert timeout == client.timeout

        def settimeout(self, _timeout: float) -> None:
            pass

        def recv(self) -> str:
            return json.dumps(
                {"type": "execution_success", "data": {"prompt_id": prompt_id}}
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr("src.comfy_client.websocket.WebSocket", FakeWebSocket)

    assert client.run({}) == prompt_id
    assert (
        capsys.readouterr().out.strip() == "[comfy] owned prompt queued "
        f"prompt_id={prompt_id} client_id={client.client_id} "
        "host=127.0.0.1:8188"
    )
