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
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import websocket  # from the websocket-client package

from . import config
from .io_utils import atomic_write_bytes


class ComfyClient:
    def __init__(self, host: str | None = None, timeout: float = 600.0):
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        self.host = host or config.COMFY_HOST
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    # ---- low-level HTTP ----
    def _http_json(
        self, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"http://{self.host}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ComfyUI request {path} failed with HTTP {exc.code}: "
                f"{detail or exc.reason}"
            ) from exc
        except (OSError, URLError) as exc:
            raise RuntimeError(f"ComfyUI request {path} failed: {exc}") from exc
        try:
            parsed = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"ComfyUI returned invalid JSON for {path}: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise TypeError(f"ComfyUI returned non-object JSON for {path}")
        return parsed

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
