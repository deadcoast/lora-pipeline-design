"""Preflight check — runs on YOUR machine and verifies everything the pipeline
needs BEFORE you spend time preparing data or training.

    python -m src.doctor

It actually connects to your running ComfyUI, confirms the SDXL checkpoint is
loadable, checks the LoRA folder is writable, and validates your ai-toolkit
paths. Exits non-zero if anything critical is wrong, so you can wire it into a
script. Nothing here needs a GPU.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

from . import config

OK, WARN, FAIL = "  [ok]  ", "  [warn]", "  [FAIL]"


def _get(path: str, timeout=10):
    with urllib.request.urlopen(f"http://{config.COMFY_HOST}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def check_comfy_reachable(results):
    try:
        _get("/system_stats")
        results.append((True, OK, f"ComfyUI reachable at {config.COMFY_HOST}"))
    except Exception as e:
        results.append((False, FAIL, f"ComfyUI NOT reachable at {config.COMFY_HOST} ({e}). "
                                     "Start Comfy Desktop and confirm the API port."))


def check_checkpoint(results):
    try:
        info = _get("/object_info/CheckpointLoaderSimple")
        names = info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        if config.COMFY_SDXL_CKPT in names:
            results.append((True, OK, f"SDXL checkpoint '{config.COMFY_SDXL_CKPT}' is loadable"))
        else:
            sample = ", ".join(names[:5]) or "(none found)"
            results.append((False, FAIL, f"Checkpoint '{config.COMFY_SDXL_CKPT}' not in ComfyUI. "
                                         f"Available: {sample} ..."))
    except Exception as e:
        results.append((False, WARN, f"Could not query checkpoints ({e}); is ComfyUI up?"))


def check_lora_dir(results):
    d = config.COMFY_LORA_DIR
    if not d:
        results.append((False, FAIL, "COMFY_LORA_DIR not set in .env"))
        return
    d = Path(d)
    if not d.exists():
        results.append((False, FAIL, f"COMFY_LORA_DIR does not exist: {d}"))
        return
    try:
        probe = d / ".write_probe"
        probe.write_text("ok"); probe.unlink()
        results.append((True, OK, f"LoRA folder writable: {d}"))
    except Exception as e:
        results.append((False, FAIL, f"LoRA folder not writable: {d} ({e})"))


def check_ai_toolkit(results):
    if not config.AI_TOOLKIT_DIR:
        results.append((False, FAIL, "AI_TOOLKIT_DIR not set in .env"))
        return
    d = Path(config.AI_TOOLKIT_DIR)
    run_py = d / "run.py"
    if not d.exists():
        results.append((False, FAIL, f"AI_TOOLKIT_DIR missing: {d}"))
    elif not run_py.exists():
        results.append((False, FAIL, f"ai-toolkit run.py missing at {run_py}"))
    else:
        results.append((True, OK, f"ai-toolkit found: {d}"))
    py = config.AI_TOOLKIT_PYTHON
    if py and py != "python" and not Path(py).exists():
        results.append((False, WARN, f"AI_TOOLKIT_PYTHON not found: {py} (its venv python.exe)"))
    else:
        results.append((True, OK, f"ai-toolkit python: {py}"))


def check_config_file(results):
    cfg = config.ROOT / "configs" / "person_sdxl.yaml"
    if cfg.exists():
        txt = cfg.read_text()
        if "PUT_ABSOLUTE_PATH_TO" in txt:
            results.append((False, WARN, "person_sdxl.yaml still has the placeholder dataset path "
                                         "(PUT_ABSOLUTE_PATH_TO/...); edit it before training."))
        else:
            results.append((True, OK, "person_sdxl.yaml present and dataset path edited"))
    else:
        results.append((False, FAIL, f"Missing config: {cfg}"))


def run() -> int:
    print(f"\nPreflight for lora-pipeline\n{'-'*50}")
    results = []
    check_comfy_reachable(results)
    check_checkpoint(results)
    check_lora_dir(results)
    check_ai_toolkit(results)
    check_config_file(results)

    critical_fail = False
    for ok, tag, msg in results:
        print(f"{tag} {msg}")
        if tag == FAIL:
            critical_fail = True

    print("-" * 50)
    if critical_fail:
        print("RESULT: NOT READY — fix the [FAIL] items above.\n")
        return 1
    print("RESULT: READY. You can prepare a dataset and train.\n")
    return 0


if __name__ == "__main__":
    sys.exit(run())
