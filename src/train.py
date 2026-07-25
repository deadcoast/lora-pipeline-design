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
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

from . import config
from .comfy_activity import check_orphaned_comfy_prompts
from .dataset_pairs import scan_dataset_pairs
from .io_utils import atomic_write_text
from .prepare_dataset import MANIFEST_NAME, prepared_output_manifest_issue
from .process_control import run_managed_process
from .run_guardian import active_run_guard_covers, pipeline_run_guard

TRAINING_PRESETS = ("fast", "balanced", "quality")
_PRESET_VALUES: dict[str, dict[str, Any]] = {
    "fast": {
        "rank": 8,
        "steps": 900,
        "learning_rate": 1.0e-4,
        "save_every": 225,
        "max_saves": 5,
        "resolutions": [512, 768],
        "sample_size": 768,
        "sample_steps": 20,
    },
    "balanced": {
        "rank": 16,
        "steps": 2000,
        "learning_rate": 1.0e-4,
        "save_every": 250,
        "max_saves": 8,
        "resolutions": [768, 1024],
        "sample_size": 1024,
        "sample_steps": 25,
    },
    "quality": {
        "rank": 32,
        "steps": 3000,
        "learning_rate": 7.5e-5,
        "save_every": 300,
        "max_saves": 10,
        "resolutions": [768, 1024],
        "sample_size": 1024,
        "sample_steps": 30,
    },
}


def _resolve_path(path: str | Path) -> Path:
    cfg = Path(path)
    if not cfg.is_absolute():
        cfg = config.ROOT / cfg
    return cfg


def _load_template(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    cfg = _resolve_path(config_path)
    if not cfg.exists():
        raise FileNotFoundError(f"Config not found: {cfg}")
    try:
        loaded = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in training config {cfg}: {exc}") from exc
    if not isinstance(loaded, dict) or not isinstance(loaded.get("config"), dict):
        raise TypeError(f"Training config {cfg} must contain a mapping named 'config'.")
    return cfg, loaded


def _validate_dataset_pairs(dataset_dir: Path) -> int:
    """Validate the flat image/caption dataset that ai-toolkit will consume."""
    manifest_path = dataset_dir / MANIFEST_NAME
    if manifest_path.is_file():
        try:
            if manifest_path.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("the preparation record is unexpectedly large")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"Dataset {dataset_dir} has a damaged preparation record: {exc}. "
                "Rebuild the prepared photos before training."
            ) from exc
        manifest_issue = prepared_output_manifest_issue(dataset_dir, manifest)
        if manifest_issue is not None:
            raise ValueError(
                f"Dataset {dataset_dir} no longer matches the reviewed prepared "
                f"photos. {manifest_issue} Rebuild and review the photos before "
                "training."
            )
    inventory = scan_dataset_pairs(dataset_dir)
    problems = inventory.problem_messages()
    if problems:
        raise ValueError(
            f"Dataset {dataset_dir} is not trainer-ready: {' '.join(problems)}"
        )
    if not inventory.pairs:
        raise ValueError(
            f"Dataset {dataset_dir} has no matching image + .txt caption pairs."
        )

    for image_path, caption_path in inventory.pairs:
        try:
            caption = caption_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Cannot read caption {caption_path}: {exc}") from exc
        if not caption:
            raise ValueError(f"Dataset caption is empty: {caption_path}")
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Dataset image is invalid: {image_path} ({exc})") from exc
    return len(inventory.pairs)


def validate_sdxl_process(process: dict[str, Any], template: Path) -> None:
    if process.get("type") != "sd_trainer":
        raise ValueError(
            f"Training config {template} process.type must be 'sd_trainer'."
        )
    network = process.get("network")
    save = process.get("save")
    model = process.get("model")
    train_config = process.get("train")
    sample = process.get("sample")
    if (
        not isinstance(network, dict)
        or not isinstance(save, dict)
        or not isinstance(model, dict)
        or not isinstance(train_config, dict)
    ):
        raise TypeError(
            f"Training config {template} needs network, save, model, and train mappings."
        )
    if network.get("type") != "lora":
        raise ValueError(f"Training config {template} network.type must be 'lora'.")
    if save.get("save_format", "safetensors") != "safetensors":
        raise ValueError(
            f"Training config {template} must save safetensors for ComfyUI handoff."
        )
    if model.get("arch") != "sdxl" and model.get("is_xl") is not True:
        raise ValueError(
            f"Training config {template} does not identify its model as SDXL "
            "(use model.arch: sdxl)."
        )
    if (
        not isinstance(model.get("name_or_path"), str)
        or not model["name_or_path"].strip()
    ):
        raise ValueError(f"Training config {template} needs model.name_or_path.")
    for key in ("batch_size", "steps"):
        value = train_config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"Training config {template} train.{key} must be a positive integer."
            )
    learning_rate = train_config.get("lr")
    if (
        not isinstance(learning_rate, int | float)
        or isinstance(learning_rate, bool)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError(
            f"Training config {template} train.lr must be a positive finite number."
        )
    scheduler = train_config.get("noise_scheduler", "ddpm")
    if scheduler == "flowmatch":
        raise ValueError(
            f"Training config {template} uses flowmatch for SDXL; use ddpm."
        )
    if isinstance(sample, dict):
        sampler = sample.get("sampler", scheduler)
        if sampler != scheduler:
            raise ValueError(
                f"Training config {template} uses train scheduler {scheduler!r} "
                f"but sample sampler {sampler!r}."
            )
        prompts = sample.get("prompts")
        if prompts is not None and (
            not isinstance(prompts, list)
            or not prompts
            or not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts)
        ):
            raise TypeError(
                f"Training config {template} sample.prompts must be a "
                "non-empty list of strings."
            )
        samples = sample.get("samples")
        if samples is not None and (
            not isinstance(samples, list)
            or not samples
            or not all(
                isinstance(item, dict)
                and isinstance(item.get("prompt"), str)
                and bool(item["prompt"].strip())
                for item in samples
            )
        ):
            raise TypeError(
                f"Training config {template} sample.samples must be a non-empty "
                "list of mappings with string prompts."
            )


def apply_training_preset(
    process: dict[str, Any],
    preset: str,
    template: Path,
) -> None:
    """Apply one explicit casual-user preset to a validated SDXL process."""

    try:
        values = _PRESET_VALUES[preset]
    except KeyError as exc:
        choices = ", ".join(TRAINING_PRESETS)
        raise ValueError(
            f"Unknown training preset {preset!r}; choose one of: {choices}."
        ) from exc

    network = process.get("network")
    train_config = process.get("train")
    save = process.get("save")
    sample = process.get("sample")
    datasets = process.get("datasets")
    if (
        not isinstance(network, dict)
        or not isinstance(train_config, dict)
        or not isinstance(save, dict)
        or not isinstance(sample, dict)
        or not isinstance(datasets, list)
        or len(datasets) != 1
        or not isinstance(datasets[0], dict)
    ):
        raise TypeError(
            f"Training config {template} cannot accept a MIVIM preset because "
            "its network, train, save, sample, or dataset structure is invalid."
        )

    rank = values["rank"]
    network["linear"] = rank
    network["linear_alpha"] = rank
    train_config["steps"] = values["steps"]
    train_config["lr"] = values["learning_rate"]
    save["save_every"] = values["save_every"]
    save["max_step_saves_to_keep"] = values["max_saves"]
    datasets[0]["resolution"] = list(values["resolutions"])
    sample["sample_every"] = values["save_every"]
    sample["width"] = values["sample_size"]
    sample["height"] = values["sample_size"]
    sample["sample_steps"] = values["sample_steps"]


def materialize_config(
    template_path: str | Path,
    *,
    run_name: str | None = None,
    trigger: str = config.DEFAULT_TRIGGER,
    dataset_dir: Path | None = None,
    output_dir: Path | None = None,
    destination: Path | None = None,
    preset: str | None = None,
    base_model: Path | None = None,
) -> Path:
    """Create the concrete ai-toolkit config used for one pipeline run."""
    template, data = _load_template(template_path)
    job_config = data["config"]

    template_name = job_config.get("name")
    if run_name is None:
        if not isinstance(template_name, str) or not template_name:
            raise ValueError(f"Training config {template} has no usable config.name.")
        run_name = template_name
    run_name = config.validate_project_name(run_name)
    trigger = trigger.strip()
    if not trigger:
        raise ValueError("Training trigger cannot be empty.")

    dataset_dir = (dataset_dir or (config.DATASETS_DIR / run_name)).resolve()
    output_dir = (output_dir or config.CHECKPOINTS_DIR).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Prepared dataset not found: {dataset_dir}. Run dataset preparation first."
        )
    pair_count = _validate_dataset_pairs(dataset_dir)
    if base_model is not None:
        base_model = base_model.resolve()
        if not base_model.is_file():
            raise FileNotFoundError(f"Training base model not found: {base_model}")

    processes = job_config.get("process")
    if not isinstance(processes, list) or not processes:
        raise ValueError(f"Training config {template} must define config.process.")

    dataset_entries: list[dict[str, Any]] = []
    for process in processes:
        if not isinstance(process, dict):
            raise TypeError(f"Training config {template} has an invalid process entry.")
        validate_sdxl_process(process, template)
        if preset is not None:
            apply_training_preset(process, preset, template)
        if base_model is not None:
            model = process["model"]
            model["name_or_path"] = str(base_model)
        process["trigger_word"] = trigger
        sample = process.get("sample")
        if isinstance(sample, dict):
            if "prompts" in sample:
                sample["prompts"] = [
                    prompt.replace("{trigger}", trigger) for prompt in sample["prompts"]
                ]
            if "samples" in sample:
                for item in sample["samples"]:
                    item["prompt"] = item["prompt"].replace("{trigger}", trigger)
        process["training_folder"] = str(output_dir)
        datasets = process.get("datasets")
        if isinstance(datasets, list):
            for dataset in datasets:
                if isinstance(dataset, dict):
                    dataset_entries.append(dataset)

    if len(dataset_entries) != 1:
        raise ValueError(
            f"Training config {template} must define exactly one dataset; "
            f"found {len(dataset_entries)}."
        )

    dataset_entries[0]["folder_path"] = str(dataset_dir)
    job_config["name"] = run_name
    meta = data.get("meta")
    if isinstance(meta, dict):
        meta["name"] = run_name
        if preset is not None:
            meta["mivim_preset"] = preset

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = destination or (config.RUN_CONFIGS_DIR / f"{run_name}.yaml")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        destination,
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
    )
    print(
        f"[train] Run config: {destination} "
        f"({pair_count} pair{'s' if pair_count != 1 else ''})"
    )
    return destination


def _training_project_name(
    config_path: str | Path,
    run_name: str | None,
) -> str:
    if run_name is not None:
        return config.validate_project_name(run_name)
    _template, template_data = _load_template(config_path)
    template_name = template_data["config"].get("name")
    if not isinstance(template_name, str) or not template_name:
        raise ValueError("Training config has no usable config.name.")
    return config.validate_project_name(template_name)


def train(
    config_path: str | Path,
    *,
    run_name: str | None = None,
    trigger: str = config.DEFAULT_TRIGGER,
    dataset_dir: Path | None = None,
    output_dir: Path | None = None,
    preset: str | None = None,
    base_model: Path | None = None,
) -> int:
    if not config.AI_TOOLKIT_DIR:
        raise RuntimeError("AI_TOOLKIT_DIR is not set in .env")

    run_py = Path(config.AI_TOOLKIT_DIR) / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(
            f"ai-toolkit run.py not found at {run_py} (check AI_TOOLKIT_DIR)"
        )

    project = _training_project_name(config_path, run_name)
    resolved_dataset = (dataset_dir or (config.DATASETS_DIR / project)).resolve()
    resolved_output = (output_dir or config.CHECKPOINTS_DIR).resolve()
    workspace = config.RUN_CONFIGS_DIR.resolve().parent
    resources = {
        resolved_dataset,
        resolved_output,
        (config.RUN_CONFIGS_DIR / f"{project}.yaml").resolve(),
    }
    if base_model is not None:
        resources.add(base_model.resolve())
    if not active_run_guard_covers(
        project=project,
        workspace=workspace,
        use_accelerator=True,
        resource_paths=resources,
    ):
        with pipeline_run_guard(
            project=project,
            workspace=workspace,
            mode="train",
            use_accelerator=True,
            resource_paths=resources,
        ):
            check_orphaned_comfy_prompts()
            return _train_with_guard(
                config_path,
                run_py=run_py,
                run_name=project,
                trigger=trigger,
                dataset_dir=resolved_dataset,
                output_dir=resolved_output,
                preset=preset,
                base_model=base_model,
            )
    check_orphaned_comfy_prompts()
    return _train_with_guard(
        config_path,
        run_py=run_py,
        run_name=project,
        trigger=trigger,
        dataset_dir=resolved_dataset,
        output_dir=resolved_output,
        preset=preset,
        base_model=base_model,
    )


def _train_with_guard(
    config_path: str | Path,
    *,
    run_py: Path,
    run_name: str,
    trigger: str,
    dataset_dir: Path,
    output_dir: Path,
    preset: str | None,
    base_model: Path | None,
) -> int:
    cfg = materialize_config(
        config_path,
        run_name=run_name,
        trigger=trigger,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        preset=preset,
        base_model=base_model,
    )
    cmd = [config.AI_TOOLKIT_PYTHON, str(run_py), str(cfg)]
    print(f"[train] {' '.join(cmd)}")
    print(f"[train] cwd = {config.AI_TOOLKIT_DIR}")
    # Stream ai-toolkit output live; run in its own directory so relative paths resolve.
    return_code = run_managed_process(cmd, cwd=Path(config.AI_TOOLKIT_DIR))
    if return_code != 0:
        print(f"[train] ai-toolkit exited with code {return_code}", file=sys.stderr)
    else:
        print(f"[train] done. Checkpoints: {output_dir or config.CHECKPOINTS_DIR}")
    return return_code


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run an ai-toolkit training config.")
    ap.add_argument(
        "--config", required=True, help="Path to the ai-toolkit YAML config."
    )
    ap.add_argument("--name", help="Run name (default: config.name).")
    ap.add_argument(
        "--trigger",
        default=config.DEFAULT_TRIGGER,
        help="Trigger substituted into sample prompts.",
    )
    ap.add_argument(
        "--preset",
        choices=TRAINING_PRESETS,
        help="Optional MIVIM training preset (fast, balanced, or quality).",
    )
    ap.add_argument(
        "--base-model",
        type=Path,
        help="Local SDXL checkpoint used as ai-toolkit's model.name_or_path.",
    )
    ap.add_argument(
        "--dataset",
        type=Path,
        help="Prepared dataset directory (default: DATASETS_DIR/<name>).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=config.CHECKPOINTS_DIR,
        help="Checkpoint directory (default: CHECKPOINTS_DIR).",
    )
    args = ap.parse_args(argv)
    try:
        project = args.name
        if project is None:
            _template, template_data = _load_template(args.config)
            template_name = template_data["config"].get("name")
            if not isinstance(template_name, str) or not template_name:
                raise ValueError("Training config has no usable config.name.")
            project = template_name
        project = config.validate_project_name(project)
        workspace = config.RUN_CONFIGS_DIR.resolve().parent
        dataset_dir = (args.dataset or (config.DATASETS_DIR / project)).resolve()
        output_dir = args.output_dir.resolve()
        resources = {
            dataset_dir,
            output_dir,
            (config.RUN_CONFIGS_DIR / f"{project}.yaml").resolve(),
        }
        if args.base_model is not None:
            resources.add(args.base_model.resolve())
        with pipeline_run_guard(
            project=project,
            workspace=workspace,
            mode="train",
            use_accelerator=True,
            project_scopes=(
                dataset_dir.resolve().parent,
                output_dir.parent,
            ),
            resource_paths=resources,
        ):
            check_orphaned_comfy_prompts()
            code = train(
                args.config,
                run_name=project,
                trigger=args.trigger,
                dataset_dir=args.dataset,
                output_dir=output_dir,
                preset=args.preset,
                base_model=args.base_model,
            )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        ap.error(str(exc))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
