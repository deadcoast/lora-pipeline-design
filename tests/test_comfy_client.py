from __future__ import annotations

import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from pathlib import Path
from typing import Any, Self, cast
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request

import pytest

from src.comfy_client import ComfyClient, ComfyUserProfile


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


class _UserDataApi:
    def __init__(
        self,
        *,
        users: dict[str, str] | None = None,
        api_prefix: bool = True,
    ) -> None:
        self.users = users
        self.api_prefix = api_prefix
        self.files: dict[tuple[str, str], bytes] = {}
        self.calls: list[tuple[str, str, bytes | None, dict[str, str]]] = []
        self.drop_next_upload_response = False
        self.corrupt_reads = False
        self._file_lock = threading.Lock()
        self.racy_upload_barrier: threading.Barrier | None = None

    @staticmethod
    def _error(request: Request, status: int, body: bytes) -> HTTPError:
        return HTTPError(
            request.full_url,
            status,
            "test error",
            Message(),
            io.BytesIO(body),
        )

    def __call__(self, request: Request, *, timeout: float) -> _FakeResponse:
        assert timeout > 0
        parsed = urlsplit(request.full_url)
        path = parsed.path
        headers = {key.casefold(): value for key, value in request.header_items()}
        method = request.get_method()
        request_data = cast(bytes | None, request.data)
        self.calls.append((method, parsed.path, request_data, headers))
        if path.startswith("/api/"):
            if not self.api_prefix:
                raise self._error(request, 404, b"missing prefix")
            path = path.removeprefix("/api")

        if path == "/users":
            payload: dict[str, object] = {"storage": "server", "migrated": True}
            if self.users is not None:
                payload["users"] = self.users
            return _FakeResponse(json.dumps(payload).encode("utf-8"))

        if not path.startswith("/userdata/"):
            raise self._error(request, 404, b"unknown route")
        relative_path = unquote(path.removeprefix("/userdata/"))
        user_id = headers.get("comfy-user", "default")
        key = (user_id, relative_path)
        if method == "POST":
            if self.racy_upload_barrier is not None:
                if key in self.files:
                    raise self._error(request, 409, b"File already exists")
                # Model ComfyUI's non-atomic existence-check/write window: all
                # workers observe absence before any of them writes.
                self.racy_upload_barrier.wait(timeout=5.0)
                self.files[key] = request_data or b""
                return _FakeResponse(
                    json.dumps({"path": relative_path}).encode("utf-8")
                )
            with self._file_lock:
                if key in self.files:
                    raise self._error(request, 409, b"File already exists")
                self.files[key] = request_data or b""
                if self.drop_next_upload_response:
                    self.drop_next_upload_response = False
                    raise URLError("response dropped")
            return _FakeResponse(json.dumps({"path": relative_path}).encode("utf-8"))
        with self._file_lock:
            if key not in self.files:
                raise self._error(request, 404, b"missing file")
            body = self.files[key]
        return _FakeResponse(b"different bytes" if self.corrupt_reads else body)


@pytest.mark.parametrize(
    "host",
    (
        "192.168.1.25:8188",
        "https://127.0.0.1:8188",
        "user@127.0.0.1:8188",
        "127.0.0.1:8188/path",
    ),
)
def test_workflow_publication_rejects_non_loopback_or_malformed_hosts(
    host: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def unexpected_request(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("src.comfy_client.urllib.request.urlopen", unexpected_request)

    with pytest.raises(ValueError, match="local ComfyUI|local address"):
        ComfyClient(host=host).list_user_profiles()

    assert not called


@pytest.mark.parametrize(
    "host",
    ("127.0.0.1:8188", "localhost:8188", "[::1]:8188"),
)
def test_workflow_profile_discovery_accepts_explicit_loopback_hosts(
    host: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _UserDataApi()
    monkeypatch.setattr("src.comfy_client.urllib.request.urlopen", api)

    assert ComfyClient(host=host).list_user_profiles() == (
        ComfyUserProfile("default", "Default"),
    )
    assert api.calls[0][1] == "/api/users"


def test_multi_user_profiles_are_sorted_and_system_profiles_are_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _UserDataApi(
        users={
            "profile-b": "Beta",
            "__system": "Internal",
            "profile-a": "Alpha",
        }
    )
    monkeypatch.setattr("src.comfy_client.urllib.request.urlopen", api)
    client = ComfyClient()

    assert client.list_user_profiles() == (
        ComfyUserProfile("profile-a", "Alpha"),
        ComfyUserProfile("profile-b", "Beta"),
    )


def test_workflow_publication_is_verified_idempotent_and_collision_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = tmp_path / "Portrait Test.json"
    workflow_bytes = b'{"version": 0.4, "nodes": []}\n'
    workflow.write_bytes(workflow_bytes)
    api = _UserDataApi()
    monkeypatch.setattr("src.comfy_client.urllib.request.urlopen", api)
    client = ComfyClient()

    first = client.publish_workflow(workflow, user_id="default")
    same = client.publish_workflow(workflow, user_id="default")
    api.files[("default", first.relative_path)] = b'{"user": "edited"}\n'
    renamed = client.publish_workflow(workflow, user_id="default")

    assert first.created and not first.renamed
    content_id = first.fingerprint.removeprefix("sha256:")
    assert first.relative_path == f"workflows/MIVIM/Portrait Test [{content_id}].json"
    assert api.files[("default", first.relative_path)] != workflow_bytes
    assert not same.created and same.relative_path == first.relative_path
    assert renamed.created and renamed.renamed
    assert (
        renamed.relative_path == f"workflows/MIVIM/Portrait Test [{content_id}-2].json"
    )
    assert api.files[("default", renamed.relative_path)] == workflow_bytes
    upload = next(
        call
        for call in api.calls
        if call[0] == "POST" and "Portrait%20Test%20%5B" in call[1]
    )
    assert upload[2] == workflow_bytes
    assert upload[3]["content-type"] == "application/json"


def test_workflow_publication_uses_the_verified_immutable_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = tmp_path / "portrait.json"
    verified_bytes = b'{"version": 0.4, "nodes": [{"id": 1}]}\n'
    workflow.write_bytes(verified_bytes)
    snapshot = workflow.read_bytes()
    expected_fingerprint = ComfyClient._workflow_fingerprint(snapshot)
    workflow.write_bytes(b'{"version": 0.4, "nodes": [{"id": 999}]}\n')
    api = _UserDataApi()
    monkeypatch.setattr("src.comfy_client.urllib.request.urlopen", api)

    publication = ComfyClient().publish_workflow_bytes(
        workflow.name,
        snapshot,
        user_id="default",
        expected_fingerprint=expected_fingerprint,
    )

    assert api.files[("default", publication.relative_path)] == verified_bytes
    assert workflow.read_bytes() != verified_bytes


def test_workflow_publication_reuses_an_exact_legacy_unsuffixed_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_bytes = b'{"version": 0.4, "nodes": []}\n'
    legacy_path = "workflows/MIVIM/portrait.json"
    api = _UserDataApi()
    api.files[("default", legacy_path)] = workflow_bytes
    monkeypatch.setattr("src.comfy_client.urllib.request.urlopen", api)

    publication = ComfyClient().publish_workflow_bytes(
        "portrait.json",
        workflow_bytes,
        user_id="default",
    )

    assert publication.relative_path == legacy_path
    assert not publication.created
    assert not publication.renamed
    assert all(call[0] != "POST" for call in api.calls)


def test_workflow_publication_rejects_a_changed_snapshot_before_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _UserDataApi()
    monkeypatch.setattr("src.comfy_client.urllib.request.urlopen", api)
    client = ComfyClient()
    verified = b'{"version": 0.4, "nodes": []}\n'
    changed = b'{"version": 0.4, "nodes": [{"id": 2}]}\n'

    with pytest.raises(ValueError, match="verified result"):
        client.publish_workflow_bytes(
            "portrait.json",
            changed,
            user_id="default",
            expected_fingerprint=client._workflow_fingerprint(verified),
        )

    assert api.calls == []


def test_concurrent_publications_use_content_addressed_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _UserDataApi()
    api.racy_upload_barrier = threading.Barrier(4)
    monkeypatch.setattr("src.comfy_client.urllib.request.urlopen", api)
    payloads = (
        b'{"version": 0.4, "nodes": [{"id": 1}]}\n',
        b'{"version": 0.4, "nodes": [{"id": 2}]}\n',
    )

    def publish(payload: bytes) -> tuple[bytes, str]:
        publication = ComfyClient().publish_workflow_bytes(
            "Shared Project.json",
            payload,
            user_id="default",
        )
        return payload, publication.relative_path

    with ThreadPoolExecutor(max_workers=4) as executor:
        publications = tuple(
            executor.map(publish, (payloads[0], payloads[1], payloads[0], payloads[1]))
        )

    paths_by_payload = {
        payload: {
            path
            for published_payload, path in publications
            if published_payload == payload
        }
        for payload in payloads
    }
    assert all(len(paths) == 1 for paths in paths_by_payload.values())
    assert paths_by_payload[payloads[0]].isdisjoint(paths_by_payload[payloads[1]])
    for payload, paths in paths_by_payload.items():
        relative_path = next(iter(paths))
        content_id = ComfyClient._workflow_fingerprint(payload).removeprefix("sha256:")
        assert relative_path == (f"workflows/MIVIM/Shared Project [{content_id}].json")
        assert api.files[("default", relative_path)] == payload

    upload_targets: dict[str, set[bytes]] = {}
    for method, raw_path, body, _headers in api.calls:
        if method == "POST" and body is not None:
            upload_targets.setdefault(unquote(raw_path), set()).add(body)
    assert upload_targets
    assert all(len(target_payloads) == 1 for target_payloads in upload_targets.values())


def test_workflow_publication_requires_and_uses_a_current_multi_user_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = tmp_path / "portrait.json"
    workflow.write_text('{"version": 0.4, "nodes": []}\n', encoding="utf-8")
    api = _UserDataApi(users={"alpha": "Alpha", "beta": "Beta"})
    monkeypatch.setattr("src.comfy_client.urllib.request.urlopen", api)
    client = ComfyClient()

    with pytest.raises(ValueError, match="no longer available"):
        client.publish_workflow(workflow, user_id="removed")
    publication = client.publish_workflow(workflow, user_id="beta")

    assert publication.profile == ComfyUserProfile("beta", "Beta")
    assert ("beta", publication.relative_path) in api.files
    user_data_calls = [call for call in api.calls if "/userdata/" in call[1]]
    assert user_data_calls
    assert all(call[3]["comfy-user"] == "beta" for call in user_data_calls)


def test_user_data_api_falls_back_to_legacy_prefix_and_recovers_a_lost_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = tmp_path / "portrait.json"
    workflow.write_text('{"version": 0.4, "nodes": []}\n', encoding="utf-8")
    api = _UserDataApi(api_prefix=False)
    api.drop_next_upload_response = True
    monkeypatch.setattr("src.comfy_client.urllib.request.urlopen", api)

    result = ComfyClient().publish_workflow(workflow, user_id="default")

    assert result.created
    assert any(call[1] == "/api/users" for call in api.calls)
    assert any(call[1] == "/users" for call in api.calls)
    assert any(call[1].startswith("/userdata/") for call in api.calls)


def test_workflow_publication_rejects_a_post_upload_content_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = tmp_path / "portrait.json"
    workflow.write_text('{"version": 0.4, "nodes": []}\n', encoding="utf-8")
    api = _UserDataApi()
    api.corrupt_reads = True
    monkeypatch.setattr("src.comfy_client.urllib.request.urlopen", api)

    with pytest.raises(RuntimeError, match="different workflow contents"):
        ComfyClient().publish_workflow(workflow, user_id="default")


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
