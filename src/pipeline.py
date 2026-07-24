"""End-to-end orchestrator: prep -> train -> validate.

This is your "application" entry point. Each stage can also be run on its own
(see the individual modules). Stages are opt-out via flags so you can, e.g.,
re-validate an already-trained LoRA without retraining.

Usage (from repo root):
    # full run
    python -m src.pipeline --src "C:/photos/raw" --name person_v1

    # skip prep (dataset already built), just train + validate
    python -m src.pipeline --name person_v1 --skip-prep

    # only validate an existing LoRA already copied into ComfyUI
    python -m src.pipeline --name person_v1 --skip-prep --skip-train \
        --lora person_v1.safetensors
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

from . import config
from . import prepare_dataset as prep
from . import train as trainer
from . import validate as validator
from .io_utils import atomic_copy


def _copy_lora_to_comfy(lora_path: Path) -> str:
    """Copy a trained .safetensors into COMFY_LORA_DIR so ComfyUI can see it.
    Returns the filename ComfyUI will reference."""
    if not lora_path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
    if lora_path.suffix.lower() != ".safetensors":
        raise ValueError(f"Expected a .safetensors checkpoint, got: {lora_path}")
    if not config.COMFY_LORA_DIR or not config.COMFY_LORA_DIR.is_dir():
        raise RuntimeError(
            "COMFY_LORA_DIR is unset or missing; cannot copy the trained LoRA "
            "into ComfyUI for validation."
        )
    dest = config.COMFY_LORA_DIR / lora_path.name
    if lora_path.resolve() != dest.resolve():
        if dest.exists():
            backup = dest.with_name(f"{dest.name}.previous")
            atomic_copy(dest, backup)
            print(f"[pipeline] Preserved previous LoRA -> {backup}")
        atomic_copy(lora_path, dest)
        print(f"[pipeline] Copied LoRA -> {dest}")
    return lora_path.name


def _checkpoints(name: str) -> list[Path]:
    filename = re.compile(rf"^{re.escape(name)}(?:_\d+)?\.safetensors$")
    return sorted(
        (
            path
            for path in config.CHECKPOINTS_DIR.rglob("*.safetensors")
            if filename.fullmatch(path.name)
        ),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )


def _checkpoint_snapshot(name: str) -> dict[Path, tuple[int, int]]:
    return {
        path.resolve(): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in _checkpoints(name)
    }


def run(args: argparse.Namespace) -> Path | None:
    name = config.validate_project_name(args.name)
    preset = getattr(args, "preset", "balanced")
    size = args.size if args.size is not None else (768 if preset == "fast" else 1024)
    if args.skip_prep and args.skip_train and args.skip_validate:
        raise ValueError("All pipeline stages are skipped; there is nothing to run.")
    if args.lora and not args.skip_train:
        raise ValueError(
            "--lora selects an existing checkpoint and requires --skip-train."
        )
    trigger = args.trigger
    dataset_dir = config.DATASETS_DIR / name
    before_training: dict[Path, tuple[int, int]] | None = None

    if not args.skip_prep:
        if not args.src:
            raise ValueError("--src is required unless --skip-prep is set.")
        prep.prepare(
            src=Path(args.src),
            name=name,
            size=size,
            trigger=trigger,
            caption_mode="blip" if args.vision else "template",
            overwrite=args.overwrite,
        )

    if not args.skip_train:
        before_training = _checkpoint_snapshot(name)
        code = trainer.train(
            args.config,
            run_name=name,
            trigger=trigger,
            dataset_dir=dataset_dir,
            output_dir=config.CHECKPOINTS_DIR,
            preset=preset,
            base_model=getattr(args, "base_model", None),
        )
        if code != 0:
            raise RuntimeError(
                f"Training failed (exit {code}); stopping before validation."
            )

    grid_path: Path | None = None
    if not args.skip_validate:
        lora_name = args.lora
        if lora_name:
            local_lora = Path(lora_name)
            if local_lora.is_file():
                lora_name = _copy_lora_to_comfy(local_lora)
            elif local_lora.is_absolute() or local_lora.parent != Path("."):
                raise FileNotFoundError(f"LoRA checkpoint not found: {local_lora}")
        else:
            candidates = _checkpoints(name)
            if before_training is not None:
                candidates = [
                    path
                    for path in candidates
                    if before_training.get(path.resolve())
                    != (path.stat().st_mtime_ns, path.stat().st_size)
                ]
            if not candidates:
                qualifier = "new " if before_training is not None else ""
                raise RuntimeError(
                    f"No {qualifier}trained LoRA found under {config.CHECKPOINTS_DIR}. "
                    "Pass --lora <file> or verify the ai-toolkit output."
                )
            lora_name = _copy_lora_to_comfy(candidates[0])

        grid_path = validator.validate(
            lora=lora_name,
            strengths=args.strengths,
            prompts=validator.DEFAULT_PROMPTS,
            trigger=trigger,
            seed=args.seed,
            run_name=name,
        )

    if grid_path is not None:
        print(f"\n[pipeline] Complete. Review the grid: {grid_path}")
    else:
        print("\n[pipeline] Complete.")
    return grid_path


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="prep -> train -> validate a person LoRA.")
    ap.add_argument("--name", required=True, help="Project/dataset name.")
    ap.add_argument("--src", help="Raw photos folder (required unless --skip-prep).")
    ap.add_argument("--config", default="configs/person_sdxl.yaml")
    ap.add_argument("--trigger", default=config.DEFAULT_TRIGGER)
    ap.add_argument(
        "--size",
        type=int,
        default=None,
        help="Photo long-edge size (defaults to the selected preset).",
    )
    ap.add_argument(
        "--preset",
        choices=trainer.TRAINING_PRESETS,
        default="balanced",
        help="Training plan used to set rank, steps, previews, and resolution.",
    )
    ap.add_argument(
        "--base-model",
        type=Path,
        help="Local SDXL checkpoint path passed to the trainer.",
    )
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.6, 0.8, 1.0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--lora", help="LoRA filename already visible to ComfyUI (skips auto-detect)."
    )
    ap.add_argument(
        "--vision", action="store_true", help="BLIP auto-captioning during prep."
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-prep", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
