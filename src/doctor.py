"""Preflight check — runs on YOUR machine and verifies everything the pipeline
needs BEFORE you spend time preparing data or training.

    python -m src.doctor

It actually connects to your running ComfyUI, confirms the SDXL checkpoint is
loadable, checks the LoRA folder is writable, and validates your ai-toolkit
paths. Exits non-zero if anything critical is wrong, so you can wire it into a
script. Nothing here needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from . import config
from . import train as trainer
from . import validate as validator

OK, WARN, FAIL = "  [ok]  ", "  [warn]", "  [FAIL]"
CheckResult = tuple[bool, str, str]


def _get(path: str, timeout: float = 10) -> dict[str, Any]:
    with urllib.request.urlopen(
        f"http://{config.COMFY_HOST}{path}", timeout=timeout
    ) as r:
        parsed = json.loads(r.read())
    if not isinstance(parsed, dict):
        raise TypeError(f"ComfyUI returned non-object JSON for {path}")
    return parsed


def check_comfy_reachable(results: list[CheckResult]) -> None:
    try:
        _get("/system_stats")
        results.append((True, OK, f"ComfyUI reachable at {config.COMFY_HOST}"))
    except Exception as e:  # noqa: BLE001 - a preflight probe reports every failure
        message = (
            f"ComfyUI NOT reachable at {config.COMFY_HOST} ({e}). "
            "Start Comfy Desktop and confirm the API port."
        )
        results.append((False, FAIL, message))


def check_checkpoint(results: list[CheckResult]) -> None:
    try:
        info = _get("/object_info/CheckpointLoaderSimple")
        names = info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        if config.COMFY_SDXL_CKPT in names:
            results.append(
                (True, OK, f"SDXL checkpoint '{config.COMFY_SDXL_CKPT}' is loadable")
            )
        else:
            sample = ", ".join(names[:5]) or "(none found)"
            message = (
                f"Checkpoint '{config.COMFY_SDXL_CKPT}' not in ComfyUI. "
                f"Available: {sample} ..."
            )
            results.append((False, FAIL, message))
    except Exception as e:  # noqa: BLE001 - a preflight probe reports every failure
        results.append((False, FAIL, f"Could not verify checkpoints ({e})"))


def check_lora_dir(results: list[CheckResult]) -> None:
    d = config.COMFY_LORA_DIR
    if not d:
        results.append((False, FAIL, "COMFY_LORA_DIR not set in .env"))
        return
    d = Path(d)
    if not d.exists():
        results.append((False, FAIL, f"COMFY_LORA_DIR does not exist: {d}"))
        return
    try:
        handle, probe_name = tempfile.mkstemp(prefix=".lora-write-probe-", dir=d)
        os.close(handle)
        Path(probe_name).unlink()
        results.append((True, OK, f"LoRA folder writable: {d}"))
    except OSError as e:
        results.append((False, FAIL, f"LoRA folder not writable: {d} ({e})"))


def _executable_exists(executable: str) -> bool:
    path = Path(executable)
    if path.is_absolute() or path.parent != Path("."):
        return path.is_file()
    return shutil.which(executable) is not None


def check_ai_toolkit(results: list[CheckResult]) -> None:
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
    if not py or not _executable_exists(py):
        results.append(
            (False, FAIL, f"AI_TOOLKIT_PYTHON not found: {py} (its venv python.exe)")
        )
    else:
        results.append((True, OK, f"ai-toolkit python: {py}"))


def check_config_file(results: list[CheckResult], cfg: Path) -> None:
    if not cfg.is_file():
        results.append((False, FAIL, f"Missing config: {cfg}"))
        return
    try:
        loaded = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise TypeError("top level must be a mapping")
        job_config = loaded["config"]
        if not isinstance(job_config, dict):
            raise TypeError("config must be a mapping")
        processes = job_config["process"]
        if not isinstance(processes, list) or not processes:
            raise TypeError("config.process must be a non-empty list")
        datasets: list[dict[str, Any]] = []
        for process in processes:
            if not isinstance(process, dict):
                raise TypeError("process entries must be mappings")
            trainer.validate_sdxl_process(process, cfg)
            process_datasets = process.get("datasets")
            if not isinstance(process_datasets, list):
                raise TypeError("process.datasets must be a list")
            for dataset in process_datasets:
                if not isinstance(dataset, dict):
                    raise TypeError("dataset entries must be mappings")
                datasets.append(dataset)
        name = job_config.get("name")
        if not isinstance(name, str):
            raise TypeError("config.name must be a string")
        config.validate_project_name(name)
        if len(datasets) != 1:
            raise ValueError("expected exactly one dataset")
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        results.append((False, FAIL, f"Invalid training config {cfg}: {exc}"))
        return
    message = (
        f"Training template valid: {cfg.name} "
        "(dataset/output paths are resolved per run)"
    )
    results.append((True, OK, message))


def check_workflow_file(results: list[CheckResult], workflow: Path) -> None:
    try:
        validator.load_workflow(workflow)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        results.append((False, FAIL, f"Invalid validation workflow {workflow}: {exc}"))
        return
    results.append((True, OK, f"Validation workflow valid: {workflow.name}"))


def run(
    config_path: Path | None = None,
    workflow_path: Path | None = None,
) -> int:
    print(f"\nPreflight for lora-pipeline\n{'-' * 50}")
    results: list[CheckResult] = []
    check_comfy_reachable(results)
    check_checkpoint(results)
    check_lora_dir(results)
    check_ai_toolkit(results)
    check_config_file(
        results,
        config_path or (config.ROOT / "configs" / "person_sdxl.yaml"),
    )
    check_workflow_file(results, workflow_path or validator.WORKFLOW)

    critical_fail = False
    for ok, tag, msg in results:
        print(f"{tag} {msg}")
        if tag == FAIL:
            critical_fail = True

    print("-" * 50)
    if critical_fail:
        print("RESULT: NOT READY - fix the [FAIL] items above.\n")
        return 1
    print("RESULT: READY. You can prepare a dataset and train.\n")
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Verify the LoRA pipeline environment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=config.ROOT / "configs" / "person_sdxl.yaml",
        help="Training config template to validate.",
    )
    parser.add_argument(
        "--workflow",
        type=Path,
        default=validator.WORKFLOW,
        help="ComfyUI API-format validation workflow to validate.",
    )
    args = parser.parse_args(argv)
    raise SystemExit(run(args.config, args.workflow))


if __name__ == "__main__":
    main()
