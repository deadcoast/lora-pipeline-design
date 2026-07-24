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

import json
import re
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError

import websocket  # from the websocket-client package

from . import config
from .io_utils import atomic_write_bytes

CancellationStatus = Literal[
    "pending_delete_requested",
    "running_interrupt_requested",
    "running_interrupt_unsupported",
    "not_found",
    "ownership_unverified",
]
TARGETED_INTERRUPT_MIN_VERSION = (0, 3, 56)


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
    def _http_bytes(self, path: str, payload: dict[str, Any] | None = None) -> bytes:
        url = f"http://{self.host}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=min(self.timeout, 30.0)) as resp:
                body = resp.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ComfyUI request {path} failed with HTTP {exc.code}: "
                f"{detail or exc.reason}"
            ) from exc
        except (OSError, URLError) as exc:
            raise RuntimeError(f"ComfyUI request {path} failed: {exc}") from exc
        return body

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
        running = queue.get("queue_running", [])
        pending = queue.get("queue_pending", [])
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
            prompt_id = self.queue_prompt(graph)
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
                        return prompt_id  # this prompt is done
                elif m.get("type") == "execution_success":
                    data = m.get("data")
                    if isinstance(data, dict) and data.get("prompt_id") == prompt_id:
                        return prompt_id
                elif m.get("type") in {"execution_error", "execution_interrupted"}:
                    data = m.get("data", {})
                    event_prompt = (
                        data.get("prompt_id") if isinstance(data, dict) else None
                    )
                    if event_prompt in (None, prompt_id):
                        raise RuntimeError(f"ComfyUI {m.get('type')}: {data}")
            raise TimeoutError("Timed out waiting for ComfyUI to finish the prompt.")
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
