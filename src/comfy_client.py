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
from typing import Optional

import websocket  # from the websocket-client package

from . import config


class ComfyClient:
    def __init__(self, host: Optional[str] = None, timeout: float = 600.0):
        self.host = host or config.COMFY_HOST
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    # ---- low-level HTTP ----
    def _http_json(self, path: str, payload: Optional[dict] = None) -> dict:
        url = f"http://{self.host}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def is_up(self) -> bool:
        try:
            self._http_json("/system_stats")
            return True
        except Exception:
            return False

    def queue_prompt(self, graph: dict) -> str:
        """Submit an API-format workflow; return its prompt_id."""
        res = self._http_json("/prompt", {"prompt": graph, "client_id": self.client_id})
        return res["prompt_id"]

    def _history(self, prompt_id: str) -> dict:
        return self._http_json(f"/history/{prompt_id}")

    def _view_bytes(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        q = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": folder_type}
        )
        url = f"http://{self.host}/view?{q}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read()

    # ---- run + collect ----
    def run(self, graph: dict) -> str:
        """Queue a graph and block until it finishes. Returns prompt_id."""
        ws = websocket.WebSocket()
        ws.connect(f"ws://{self.host}/ws?clientId={self.client_id}", timeout=self.timeout)
        try:
            prompt_id = self.queue_prompt(graph)
            deadline = time.time() + self.timeout
            while time.time() < deadline:
                msg = ws.recv()
                if not isinstance(msg, str):
                    continue  # binary previews — ignore
                m = json.loads(msg)
                if m.get("type") == "executing":
                    d = m["data"]
                    if d.get("node") is None and d.get("prompt_id") == prompt_id:
                        return prompt_id  # this prompt is done
                elif m.get("type") == "execution_error":
                    raise RuntimeError(f"ComfyUI execution error: {m['data']}")
            raise TimeoutError("Timed out waiting for ComfyUI to finish the prompt.")
        finally:
            ws.close()

    def save_outputs(self, prompt_id: str, out_dir: Path, prefix: str = "") -> list[Path]:
        """Download every image this prompt produced into out_dir."""
        out_dir.mkdir(parents=True, exist_ok=True)
        hist = self._history(prompt_id).get(prompt_id, {})
        saved: list[Path] = []
        outputs = hist.get("outputs", {})
        for node_id, node_out in outputs.items():
            for img in node_out.get("images", []):
                data = self._view_bytes(img["filename"], img.get("subfolder", ""), img.get("type", "output"))
                fname = f"{prefix}{img['filename']}" if prefix else img["filename"]
                dest = out_dir / fname
                dest.write_bytes(data)
                saved.append(dest)
        return saved

    def run_and_save(self, graph: dict, out_dir: Path, prefix: str = "") -> list[Path]:
        pid = self.run(graph)
        return self.save_outputs(pid, out_dir, prefix)
