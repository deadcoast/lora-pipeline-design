"""Thin wrapper around ai-toolkit's run.py.

Keeps training a one-liner from this repo while letting ai-toolkit live in its
OWN venv (important: it has heavy, conflicting deps vs. ComfyUI).

Usage (from repo root):
    python -m src.train --config configs/person_sdxl.yaml

Requires in .env:
    AI_TOOLKIT_DIR     -> path to your cloned ai-toolkit
    AI_TOOLKIT_PYTHON  -> the python.exe inside ai-toolkit's venv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import config


def train(config_path: str) -> int:
    cfg = Path(config_path)
    if not cfg.is_absolute():
        cfg = config.ROOT / cfg
    if not cfg.exists():
        raise SystemExit(f"Config not found: {cfg}")
    if not config.AI_TOOLKIT_DIR:
        raise SystemExit("AI_TOOLKIT_DIR is not set in .env")

    run_py = Path(config.AI_TOOLKIT_DIR) / "run.py"
    if not run_py.exists():
        raise SystemExit(f"ai-toolkit run.py not found at {run_py} (check AI_TOOLKIT_DIR)")

    cmd = [config.AI_TOOLKIT_PYTHON, str(run_py), str(cfg)]
    print(f"[train] {' '.join(cmd)}")
    print(f"[train] cwd = {config.AI_TOOLKIT_DIR}")
    # Stream ai-toolkit output live; run in its own directory so relative paths resolve.
    proc = subprocess.run(cmd, cwd=config.AI_TOOLKIT_DIR)
    if proc.returncode != 0:
        print(f"[train] ai-toolkit exited with code {proc.returncode}", file=sys.stderr)
    else:
        print("[train] done. Copy the chosen .safetensors into COMFY_LORA_DIR to validate.")
    return proc.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run an ai-toolkit training config.")
    ap.add_argument("--config", required=True, help="Path to the ai-toolkit YAML config.")
    args = ap.parse_args(argv)
    raise SystemExit(train(args.config))


if __name__ == "__main__":
    main()
