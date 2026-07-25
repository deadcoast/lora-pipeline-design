"""Minimal ComfyUI API client — queue a workflow, wait, fetch the images.

ComfyUI exposes a small HTTP + WebSocket API on the same port as its UI
(default 8188). A "prompt" is a workflow exported in **API format**
(Save (API Format) in the UI): a dict of node_id -> {class_type, inputs}.

This client:
  * POSTs a prompt to /prompt
  * listens on /ws for the "executing (node=None)" signal that means THIS
    prompt finished
  * pulls the output image list from /history/<prompt_id>
  * downloads image bytes from /view

Only stdlib + websocket-client are required.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError

import websocket  # from the websocket-client package

from . import config
from .comfy_activity import reserve_comfy_prompt
from .io_utils import atomic_write_bytes

CancellationStatus = Literal[
    "pending_delete_requested",
    "running_interrupt_requested",
    "running_interrupt_unsupported",
    "not_found",
    "ownership_unverified",
]
TARGETED_INTERRUPT_MIN_VERSION = (0, 3, 56)
_LOCAL_COMFY_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_WORKFLOW_UPLOAD_LIMIT = 20 * 1024 * 1024
_WORKFLOW_COLLISION_LIMIT = 100


@dataclass(frozen=True, slots=True)
class ComfyUserProfile:
    """One regular ComfyUI user-data profile."""

    id: str
    name: str


@dataclass(frozen=True, slots=True)
class WorkflowPublication:
    """Verified result of publishing one workflow to ComfyUI user storage."""

    relative_path: str
    profile: ComfyUserProfile
    fingerprint: str
    created: bool
    renamed: bool


class _ComfyRequestError(RuntimeError):
    """HTTP failure that preserves the response status for safe recovery."""

    def __init__(self, path: str, status: int, detail: str) -> None:
        self.path = path
        self.status = status
        self.detail = detail
        super().__init__(f"ComfyUI request {path} failed with HTTP {status}: {detail}")


@dataclass(frozen=True, slots=True)
class PromptCancellationResult:
    """The safe, user-presentable outcome of an owned-prompt cancel request."""

    status: CancellationStatus
    prompt_id: str
    client_id: str
    message: str

    @property
    def requested(self) -> bool:
        return self.status in {
            "pending_delete_requested",
            "running_interrupt_requested",
        }


class ComfyClient:
    def __init__(self, host: str | None = None, timeout: float = 600.0):
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        self.host = host or config.COMFY_HOST
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    # ---- low-level HTTP ----
    def _request_bytes(
        self,
        path: str,
        *,
        data: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> bytes:
        """Send one HTTP request while preserving status-bearing failures."""

        url = f"http://{self.host}{path}"
        req = urllib.request.Request(
            url,
            data=data,
            headers=dict(headers or {}),
        )
        try:
            with urllib.request.urlopen(req, timeout=min(self.timeout, 30.0)) as resp:
                body = resp.read()
        except HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace").strip()
            raise _ComfyRequestError(
                path,
                exc.code,
                detail or str(exc.reason),
            ) from exc
        except (OSError, URLError) as exc:
            raise RuntimeError(f"ComfyUI request {path} failed: {exc}") from exc
        return body

    def _http_bytes(self, path: str, payload: dict[str, Any] | None = None) -> bytes:
        data = json.dumps(payload).encode() if payload is not None else None
        return self._request_bytes(
            path,
            data=data,
            headers={"Content-Type": "application/json"},
        )

    def _http_json(
        self, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        body = self._http_bytes(path, payload)
        try:
            parsed = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"ComfyUI returned invalid JSON for {path}: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise TypeError(f"ComfyUI returned non-object JSON for {path}")
        return parsed

    def _post_json(self, path: str, payload: dict[str, Any]) -> None:
        """POST JSON to an endpoint whose successful response may be empty."""
        self._http_bytes(path, payload)

    def _require_loopback_host(self) -> None:
        """Reject workflow writes unless the configured server is loopback-only."""

        if "://" in self.host:
            raise ValueError(
                "Workflows require a local ComfyUI address without a URL scheme."
            )
        candidate = f"http://{self.host}"
        try:
            parsed = urllib.parse.urlsplit(candidate)
            hostname = parsed.hostname.casefold() if parsed.hostname else ""
            port = parsed.port or 8188
        except ValueError as exc:
            raise ValueError("ComfyUI has an invalid local address.") from exc
        if (
            parsed.scheme.casefold() != "http"
            or hostname not in _LOCAL_COMFY_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or not 1 <= port <= 65535
        ):
            raise ValueError(
                "Workflows can only be added to a verified local ComfyUI address."
            )

    def _user_api_request(
        self,
        path: str,
        *,
        data: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> bytes:
        """Use the current ``/api`` route with a legacy-prefix fallback."""

        last_not_found: _ComfyRequestError | None = None
        for prefix in ("/api", ""):
            request_path = f"{prefix}{path}"
            try:
                return self._request_bytes(
                    request_path,
                    data=data,
                    headers=headers,
                )
            except _ComfyRequestError as exc:
                if exc.status == 404 and prefix:
                    last_not_found = exc
                    continue
                raise
        if last_not_found is not None:
            raise last_not_found
        raise RuntimeError("ComfyUI user-data request did not run.")

    def list_user_profiles(self) -> tuple[ComfyUserProfile, ...]:
        """Return regular user-data profiles without guessing among them."""

        self._require_loopback_host()
        body = self._user_api_request("/users")
        try:
            value = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"ComfyUI returned invalid user-profile data: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise TypeError("ComfyUI returned invalid user-profile data.")

        raw_users = value.get("users")
        if raw_users is None:
            return (ComfyUserProfile("default", "Default"),)
        if not isinstance(raw_users, Mapping):
            raise TypeError("ComfyUI returned an invalid user-profile list.")

        profiles = tuple(
            sorted(
                (
                    ComfyUserProfile(user_id.strip(), name.strip())
                    for raw_id, raw_name in raw_users.items()
                    if isinstance(raw_id, str)
                    and isinstance(raw_name, str)
                    and (user_id := raw_id.strip())
                    and (name := raw_name.strip())
                    and not user_id.startswith("__")
                    and len(user_id) <= 256
                    and len(name) <= 256
                    and not any(
                        ord(character) < 32 or ord(character) == 127
                        for character in f"{user_id}{name}"
                    )
                ),
                key=lambda profile: (profile.name.casefold(), profile.id),
            )
        )
        if not profiles:
            raise RuntimeError(
                "ComfyUI has no regular user profile that can receive this workflow."
            )
        return profiles

    @staticmethod
    def _workflow_fingerprint(data: bytes) -> str:
        digest = hashlib.sha256()
        digest.update(b"mivim-file-v1\0")
        digest.update(data)
        return f"sha256:{digest.hexdigest()}"

    @staticmethod
    def _workflow_candidate(
        name: str,
        fingerprint: str,
        collision: int,
    ) -> str:
        path = Path(name)
        content_id = fingerprint.removeprefix("sha256:")
        if collision == 1:
            marker = content_id
        else:
            marker = f"{content_id}-{collision}"
        return f"workflows/MIVIM/{path.stem} [{marker}]{path.suffix}"

    @staticmethod
    def _encoded_user_path(relative_path: str) -> str:
        return urllib.parse.quote(relative_path, safe="")

    def _read_user_file(
        self,
        relative_path: str,
        profile: ComfyUserProfile,
    ) -> bytes:
        encoded = self._encoded_user_path(relative_path)
        return self._user_api_request(
            f"/userdata/{encoded}",
            headers={"Comfy-User": profile.id},
        )

    def publish_workflow(
        self,
        workflow_path: Path,
        *,
        user_id: str,
        expected_fingerprint: str | None = None,
    ) -> WorkflowPublication:
        """Publish one path snapshot without rereading it during the upload."""

        self._require_loopback_host()
        if workflow_path.is_symlink():
            raise ValueError("The ready workflow cannot be a symbolic link.")
        try:
            data = workflow_path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"The ready workflow could not be read: {exc}") from exc
        return self.publish_workflow_bytes(
            workflow_path.name,
            data,
            user_id=user_id,
            expected_fingerprint=expected_fingerprint,
        )

    def publish_workflow_bytes(
        self,
        workflow_name: str,
        data: bytes,
        *,
        user_id: str,
        expected_fingerprint: str | None = None,
    ) -> WorkflowPublication:
        """Publish and verify an immutable UI-workflow byte snapshot.

        Every new upload path includes the snapshot's content fingerprint. This
        makes concurrent uploads of different workflow bytes target different
        files even when ComfyUI's ``overwrite=false`` existence check and write
        are not one atomic operation.
        """

        self._require_loopback_host()
        if (
            not isinstance(workflow_name, str)
            or Path(workflow_name).name != workflow_name
            or Path(workflow_name).suffix.casefold() != ".json"
            or workflow_name != workflow_name.strip()
            or any(ord(character) < 32 for character in workflow_name)
        ):
            raise ValueError("The ready workflow does not have a safe JSON filename.")
        if not isinstance(data, bytes):
            raise TypeError("The ready workflow snapshot must be immutable bytes.")
        if not data:
            raise ValueError("The ready workflow is empty.")
        if len(data) > _WORKFLOW_UPLOAD_LIMIT:
            raise ValueError("The ready workflow is unexpectedly large.")
        try:
            workflow = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"The ready workflow is not valid JSON: {exc}") from exc
        if not isinstance(workflow, Mapping):
            raise TypeError("The ready workflow must contain a JSON object.")

        fingerprint = self._workflow_fingerprint(data)
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise ValueError(
                "The ready workflow snapshot no longer matches its verified result."
            )

        profiles = self.list_user_profiles()
        profile = next((item for item in profiles if item.id == user_id), None)
        if profile is None:
            raise ValueError(
                "That ComfyUI profile is no longer available. Choose a profile again."
            )

        headers = {
            "Comfy-User": profile.id,
            "Content-Type": "application/json",
        }

        # MIVIM 0.5 initially used the unsuffixed filename. Reuse an exact legacy
        # publication, but never create a new one: differently-contented concurrent
        # uploads could otherwise race inside ComfyUI's check-then-write handler.
        legacy_path = f"workflows/MIVIM/{workflow_name}"
        try:
            legacy_data = self._read_user_file(legacy_path, profile)
        except _ComfyRequestError as exc:
            if exc.status != 404:
                raise
        else:
            if legacy_data == data:
                return WorkflowPublication(
                    relative_path=legacy_path,
                    profile=profile,
                    fingerprint=fingerprint,
                    created=False,
                    renamed=False,
                )

        for collision in range(1, _WORKFLOW_COLLISION_LIMIT + 1):
            relative_path = self._workflow_candidate(
                workflow_name,
                fingerprint,
                collision,
            )
            encoded = self._encoded_user_path(relative_path)
            upload_path = f"/userdata/{encoded}?overwrite=false&full_info=true"
            try:
                self._user_api_request(
                    upload_path,
                    data=data,
                    headers=headers,
                )
            except _ComfyRequestError as exc:
                if exc.status != 409:
                    raise
                try:
                    existing = self._read_user_file(relative_path, profile)
                except _ComfyRequestError as read_exc:
                    if read_exc.status == 404:
                        continue
                    raise
                if existing == data:
                    return WorkflowPublication(
                        relative_path=relative_path,
                        profile=profile,
                        fingerprint=fingerprint,
                        created=False,
                        renamed=collision > 1,
                    )
                continue
            except RuntimeError as upload_error:
                # A dropped response can happen after ComfyUI atomically saved the
                # file. Read it back before asking a casual user to retry.
                try:
                    saved = self._read_user_file(relative_path, profile)
                except RuntimeError:
                    raise upload_error
                if saved != data:
                    raise
                return WorkflowPublication(
                    relative_path=relative_path,
                    profile=profile,
                    fingerprint=fingerprint,
                    created=True,
                    renamed=collision > 1,
                )

            saved = self._read_user_file(relative_path, profile)
            if saved != data:
                raise RuntimeError(
                    "ComfyUI saved different workflow contents; the handoff was stopped."
                )
            return WorkflowPublication(
                relative_path=relative_path,
                profile=profile,
                fingerprint=fingerprint,
                created=True,
                renamed=collision > 1,
            )

        raise RuntimeError(
            "ComfyUI already has too many workflows with this name. "
            "Rename the MIVIM project before trying again."
        )

    def is_up(self) -> bool:
        try:
            self._http_json("/system_stats")
            return True
        except Exception:  # noqa: BLE001 - every transport/protocol failure means down
            return False

    def queue_prompt(self, graph: dict[str, Any]) -> str:
        """Submit an API-format workflow; return its prompt_id."""
        res = self._http_json("/prompt", {"prompt": graph, "client_id": self.client_id})
        node_errors = res.get("node_errors")
        if node_errors:
            raise RuntimeError(f"ComfyUI rejected the workflow: {node_errors}")
        prompt_id = res.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise RuntimeError(f"ComfyUI did not return a prompt_id: {res}")
        return prompt_id

    @staticmethod
    def _queue_item_identity(item: object) -> tuple[str | None, str | None]:
        """Extract prompt/client IDs from supported local ComfyUI queue shapes."""
        prompt_id: object = None
        owner: object = None
        if isinstance(item, (list, tuple)):
            # Local ComfyUI returns:
            # [number, prompt_id, prompt, extra_data, outputs_to_execute].
            if len(item) >= 2:
                prompt_id = item[1]
            if len(item) >= 4 and isinstance(item[3], dict):
                owner = item[3].get("client_id")
        elif isinstance(item, dict):
            prompt_id = item.get("prompt_id")
            owner = item.get("client_id")
            for key in ("extra_data", "metadata"):
                nested = item.get(key)
                if owner is None and isinstance(nested, dict):
                    owner = nested.get("client_id")
        return (
            prompt_id if isinstance(prompt_id, str) and prompt_id else None,
            owner if isinstance(owner, str) and owner else None,
        )

    @staticmethod
    def _comfyui_version(stats: dict[str, Any]) -> tuple[int, int, int] | None:
        system = stats.get("system")
        if not isinstance(system, dict):
            return None
        raw_version = system.get("comfyui_version")
        if not isinstance(raw_version, str):
            return None
        match = re.fullmatch(
            r"\s*v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?\s*",
            raw_version,
        )
        if match is None:
            return None
        major, minor, patch = match.groups()
        return int(major), int(minor), int(patch)

    def cancel_owned_prompt(
        self, prompt_id: str, client_id: str | None = None
    ) -> PromptCancellationResult:
        """Cancel one prompt only after proving that this client owns it.

        Pending work is removed through ``/queue``. Running work receives a
        targeted ``/interrupt`` request. A missing/mismatched ``client_id`` is
        deliberately a no-op so another ComfyUI user's work is never stopped.
        """
        prompt_id = prompt_id.strip()
        expected_client_id = (
            self.client_id if client_id is None else client_id
        ).strip()
        if not prompt_id:
            raise ValueError("prompt_id cannot be empty")
        if not expected_client_id:
            raise ValueError("client_id cannot be empty")

        queue = self._http_json("/queue")
        if "queue_running" not in queue or "queue_pending" not in queue:
            raise TypeError("ComfyUI returned an incomplete queue response.")
        running = queue.get("queue_running")
        pending = queue.get("queue_pending")
        if not isinstance(running, list) or not isinstance(pending, list):
            raise TypeError("ComfyUI returned an invalid queue response.")

        matching_running = [
            owner
            for item in running
            for queued_prompt, owner in [self._queue_item_identity(item)]
            if queued_prompt == prompt_id
        ]
        matching_pending = [
            owner
            for item in pending
            for queued_prompt, owner in [self._queue_item_identity(item)]
            if queued_prompt == prompt_id
        ]

        if expected_client_id in matching_running:
            try:
                stats = self._http_json("/system_stats")
            except (OSError, RuntimeError, TypeError, ValueError):
                stats = {}
            version = self._comfyui_version(stats)
            if version is None or version < TARGETED_INTERRUPT_MIN_VERSION:
                version_text = (
                    ".".join(str(part) for part in version)
                    if version is not None
                    else "an unknown version"
                )
                return PromptCancellationResult(
                    status="running_interrupt_unsupported",
                    prompt_id=prompt_id,
                    client_id=expected_client_id,
                    message=(
                        f"ComfyUI {version_text} cannot safely stop only this "
                        "running prompt. MIVIM left it running; open ComfyUI and "
                        "stop this job from its queue."
                    ),
                )
            # Targeted interruption was added in ComfyUI 0.3.56. Older
            # versions treat /interrupt as global, so they never reach here.
            self._post_json("/interrupt", {"prompt_id": prompt_id})
            return PromptCancellationResult(
                status="running_interrupt_requested",
                prompt_id=prompt_id,
                client_id=expected_client_id,
                message="Interrupt requested for this running ComfyUI prompt.",
            )
        if expected_client_id in matching_pending:
            self._post_json("/queue", {"delete": [prompt_id]})
            return PromptCancellationResult(
                status="pending_delete_requested",
                prompt_id=prompt_id,
                client_id=expected_client_id,
                message="Removal requested for this queued ComfyUI prompt.",
            )
        if matching_running or matching_pending:
            return PromptCancellationResult(
                status="ownership_unverified",
                prompt_id=prompt_id,
                client_id=expected_client_id,
                message=(
                    "The prompt exists, but its client ownership could not be "
                    "verified. No ComfyUI work was stopped."
                ),
            )
        return PromptCancellationResult(
            status="not_found",
            prompt_id=prompt_id,
            client_id=expected_client_id,
            message=(
                "The prompt is no longer queued or running. No ComfyUI work "
                "needed to be stopped."
            ),
        )

    def _history(self, prompt_id: str) -> dict[str, Any]:
        return self._http_json(f"/history/{prompt_id}")

    def _view_bytes(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        q = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": folder_type}
        )
        url = f"http://{self.host}/view?{q}"
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
        except HTTPError as exc:
            raise RuntimeError(
                f"ComfyUI output download failed with HTTP {exc.code}: {filename}"
            ) from exc
        except (OSError, URLError) as exc:
            raise RuntimeError(
                f"ComfyUI output download failed for {filename}: {exc}"
            ) from exc
        if not data:
            raise RuntimeError(f"ComfyUI returned an empty output file: {filename}")
        return data

    # ---- run + collect ----
    def run(self, graph: dict[str, Any]) -> str:
        """Queue a graph and block until it finishes. Returns prompt_id."""
        ws = websocket.WebSocket()
        try:
            ws.connect(
                f"ws://{self.host}/ws?clientId={self.client_id}", timeout=self.timeout
            )
        except (OSError, websocket.WebSocketException) as exc:
            ws.close()
            raise RuntimeError(
                f"Could not connect to the ComfyUI WebSocket at {self.host}: {exc}"
            ) from exc
        try:
            reservation = reserve_comfy_prompt(self.host, self.client_id)
        except BaseException:
            ws.close()
            raise
        prompt_id: str | None = None
        try:
            prompt_id = self.queue_prompt(graph)
            reservation.attach_prompt(prompt_id)
            print(
                "[comfy] owned prompt queued "
                f"prompt_id={prompt_id} client_id={self.client_id} host={self.host}",
                flush=True,
            )
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                ws.settimeout(remaining)
                try:
                    msg = ws.recv()
                except websocket.WebSocketTimeoutException as exc:
                    raise TimeoutError(
                        "Timed out waiting for ComfyUI to finish the prompt."
                    ) from exc
                except (OSError, websocket.WebSocketException) as exc:
                    raise RuntimeError(
                        f"ComfyUI WebSocket disconnected while running {prompt_id}: {exc}"
                    ) from exc
                if not isinstance(msg, str):
                    continue  # binary previews — ignore
                try:
                    m = json.loads(msg)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"ComfyUI sent invalid WebSocket JSON: {exc}"
                    ) from exc
                if not isinstance(m, dict):
                    continue
                if m.get("type") == "executing":
                    d = m.get("data")
                    if not isinstance(d, dict):
                        continue
                    if d.get("node") is None and d.get("prompt_id") == prompt_id:
                        reservation.clear()
                        return prompt_id  # this prompt is done
                elif m.get("type") == "execution_success":
                    data = m.get("data")
                    if isinstance(data, dict) and data.get("prompt_id") == prompt_id:
                        reservation.clear()
                        return prompt_id
                elif m.get("type") in {"execution_error", "execution_interrupted"}:
                    data = m.get("data", {})
                    event_prompt = (
                        data.get("prompt_id") if isinstance(data, dict) else None
                    )
                    if event_prompt in (None, prompt_id):
                        raise RuntimeError(f"ComfyUI {m.get('type')}: {data}")
            raise TimeoutError("Timed out waiting for ComfyUI to finish the prompt.")
        except BaseException:
            if prompt_id is not None:
                try:
                    cancellation = self.cancel_owned_prompt(prompt_id)
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    print(
                        "[comfy] Could not confirm prompt cleanup; MIVIM will "
                        f"recheck ComfyUI before the next GPU run: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(
                        f"[comfy] {cancellation.message}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if cancellation.status == "not_found":
                        reservation.clear()
            # If submission failed after ComfyUI accepted the request, the
            # reservation still carries client_id and lets the next run find it.
            raise
        finally:
            ws.close()

    def save_outputs(
        self, prompt_id: str, out_dir: Path, prefix: str = ""
    ) -> list[Path]:
        """Download every image this prompt produced into out_dir."""
        if "/" in prefix or "\\" in prefix:
            raise ValueError("Output prefix cannot contain path separators.")
        out_dir.mkdir(parents=True, exist_ok=True)
        history = self._history(prompt_id)
        if prompt_id not in history:
            raise RuntimeError(f"ComfyUI history has no entry for prompt {prompt_id}")
        hist = history[prompt_id]
        if not isinstance(hist, dict):
            raise TypeError(f"ComfyUI returned invalid history for prompt {prompt_id}")
        saved: list[Path] = []
        outputs = hist.get("outputs", {})
        if not isinstance(outputs, dict):
            raise TypeError(f"ComfyUI returned invalid outputs for prompt {prompt_id}")
        for node_out in outputs.values():
            if not isinstance(node_out, dict):
                continue
            for img in node_out.get("images", []):
                if not isinstance(img, dict) or not isinstance(
                    img.get("filename"), str
                ):
                    continue
                filename = img["filename"]
                data = self._view_bytes(
                    filename,
                    str(img.get("subfolder", "")),
                    str(img.get("type", "output")),
                )
                safe_filename = Path(filename).name
                if safe_filename in {"", ".", ".."}:
                    raise ValueError(
                        f"Unsafe output filename from ComfyUI: {filename!r}"
                    )
                fname = f"{prefix}{safe_filename}" if prefix else safe_filename
                dest = out_dir / fname
                atomic_write_bytes(dest, data)
                saved.append(dest)
        return saved

    def run_and_save(
        self, graph: dict[str, Any], out_dir: Path, prefix: str = ""
    ) -> list[Path]:
        pid = self.run(graph)
        return self.save_outputs(pid, out_dir, prefix)
