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
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

from . import config
from .io_utils import atomic_write_text


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
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    images: dict[str, Path] = {}
    captions: dict[str, Path] = {}

    for path in dataset_dir.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in image_exts:
            if path.stem in images:
                raise ValueError(
                    f"Dataset {dataset_dir} has multiple images named {path.stem!r}."
                )
            images[path.stem] = path
        elif suffix == ".txt":
            captions[path.stem] = path

    missing_captions = sorted(images.keys() - captions.keys())
    missing_images = sorted(captions.keys() - images.keys())
    if missing_captions or missing_images:
        details: list[str] = []
        if missing_captions:
            details.append(
                f"images without captions: {', '.join(missing_captions[:5])}"
            )
        if missing_images:
            details.append(f"captions without images: {', '.join(missing_images[:5])}")
        raise ValueError(
            f"Dataset {dataset_dir} has incomplete pairs ({'; '.join(details)})."
        )
    if not images:
        raise ValueError(
            f"Dataset {dataset_dir} has no matching image + .txt caption pairs."
        )

    for stem, image_path in images.items():
        caption_path = captions[stem]
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
    return len(images)


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


def materialize_config(
    template_path: str | Path,
    *,
    run_name: str | None = None,
    trigger: str = config.DEFAULT_TRIGGER,
    dataset_dir: Path | None = None,
    output_dir: Path | None = None,
    destination: Path | None = None,
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

    processes = job_config.get("process")
    if not isinstance(processes, list) or not processes:
        raise ValueError(f"Training config {template} must define config.process.")

    dataset_entries: list[dict[str, Any]] = []
    for process in processes:
        if not isinstance(process, dict):
            raise TypeError(f"Training config {template} has an invalid process entry.")
        validate_sdxl_process(process, template)
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


def train(
    config_path: str | Path,
    *,
    run_name: str | None = None,
    trigger: str = config.DEFAULT_TRIGGER,
    dataset_dir: Path | None = None,
    output_dir: Path | None = None,
) -> int:
    if not config.AI_TOOLKIT_DIR:
        raise RuntimeError("AI_TOOLKIT_DIR is not set in .env")

    run_py = Path(config.AI_TOOLKIT_DIR) / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(
            f"ai-toolkit run.py not found at {run_py} (check AI_TOOLKIT_DIR)"
        )

    cfg = materialize_config(
        config_path,
        run_name=run_name,
        trigger=trigger,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
    )
    cmd = [config.AI_TOOLKIT_PYTHON, str(run_py), str(cfg)]
    print(f"[train] {' '.join(cmd)}")
    print(f"[train] cwd = {config.AI_TOOLKIT_DIR}")
    # Stream ai-toolkit output live; run in its own directory so relative paths resolve.
    proc = subprocess.run(cmd, cwd=config.AI_TOOLKIT_DIR, check=False)
    if proc.returncode != 0:
        print(f"[train] ai-toolkit exited with code {proc.returncode}", file=sys.stderr)
    else:
        print(f"[train] done. Checkpoints: {output_dir or config.CHECKPOINTS_DIR}")
    return proc.returncode


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
        code = train(
            args.config,
            run_name=args.name,
            trigger=args.trigger,
            dataset_dir=args.dataset,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        ap.error(str(exc))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
