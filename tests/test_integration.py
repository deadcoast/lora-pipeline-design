"""Self-contained integration test — proves the pipeline's ComfyUI integration
actually works, with NO GPU, NO models, and NO real ComfyUI.

It boots the mock ComfyUI (tests/mock_comfy.py) in a background process, then
drives the REAL shipped code:
  * ComfyClient: queue -> websocket completion -> history -> view -> file
  * validate.validate(): full prompt x strength sweep -> stitched grid
  * doctor.run(): preflight checks against the (mock) server

Run:
    python -m tests.test_integration          # from the repo root
Exit code 0 = all passed.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HOST = "127.0.0.1:8765"


def _wait_port(host: str, port: int, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def main() -> int:
    srv = subprocess.Popen([sys.executable, "-m", "tests.mock_comfy"], cwd=str(ROOT))
    failures = []
    restore_config: Callable[[], None] | None = None
    try:
        if not _wait_port("127.0.0.1", 8765):
            print("FAIL: mock server did not start")
            return 1

        # Import AFTER sys.path setup; point shipped config at the mock.
        from PIL import Image

        from src import config, doctor, validate
        from src.comfy_client import ComfyClient

        original_config = (
            config.COMFY_HOST,
            config.COMFY_SDXL_CKPT,
            config.COMFY_LORA_DIR,
            config.AI_TOOLKIT_DIR,
            config.GRIDS_DIR,
        )

        def restore() -> None:
            (
                config.COMFY_HOST,
                config.COMFY_SDXL_CKPT,
                config.COMFY_LORA_DIR,
                config.AI_TOOLKIT_DIR,
                config.GRIDS_DIR,
            ) = original_config

        restore_config = restore
        config.COMFY_HOST = HOST
        config.COMFY_SDXL_CKPT = "sd_xl_base_1.0.safetensors"
        config.COMFY_LORA_DIR = ROOT / "outputs"  # writable
        config.AI_TOOLKIT_DIR = ""  # doctor should flag this as FAIL (expected)
        config.GRIDS_DIR = ROOT / "outputs" / "grids" / "_test"

        # --- Test 1: raw client round-trip ---
        client = ComfyClient(host=HOST)
        assert client.is_up(), "is_up() should be True against mock"
        base = json.loads((ROOT / "workflows" / "validate_sdxl_api.json").read_text())
        graph = validate._build_graph(
            base,
            config.COMFY_SDXL_CKPT,
            "person_v1.safetensors",
            "ohwx person, test",
            0.8,
            42,
        )
        saved = client.run_and_save(
            graph, ROOT / "outputs" / "grids" / "_test_raw", prefix="t_"
        )
        valid_image = False
        if saved and saved[0].exists():
            with Image.open(saved[0]) as image:
                valid_image = image.size == (64, 64)
        if not valid_image:
            failures.append("client round-trip did not save a valid image")
        else:
            print("PASS: client queue->ws->history->view->file")

        # --- Test 2: full validate grid sweep ---
        grid = validate.validate(
            lora="person_v1.safetensors",
            strengths=[0.6, 1.0],
            prompts=validate.DEFAULT_PROMPTS,
            trigger="ohwx person",
            seed=42,
            run_name="itest",
        )
        cells = list((config.GRIDS_DIR / "itest").glob("*.png"))
        if grid.exists() and len(cells) == 2 * len(validate.DEFAULT_PROMPTS):
            print(f"PASS: validate grid ({len(cells)} cells) -> {grid.name}")
        else:
            failures.append(
                f"validate produced {len(cells)} cells / grid exists={grid.exists()}"
            )

        # --- Test 3: doctor detects a reachable server + a real config problem ---
        code = doctor.run()
        # AI_TOOLKIT_DIR is intentionally blank, so doctor MUST return non-zero.
        if code != 0:
            print("PASS: doctor correctly flagged the missing ai-toolkit path")
        else:
            failures.append("doctor should have failed on missing AI_TOOLKIT_DIR")

    finally:
        if restore_config is not None:
            restore_config()
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except subprocess.TimeoutExpired:
            srv.kill()

    print("-" * 50)
    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("ALL INTEGRATION TESTS PASSED")
    return 0


def test_integration() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
