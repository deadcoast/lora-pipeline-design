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
import shutil
from pathlib import Path

from . import config
from . import prepare_dataset as prep
from . import train as trainer
from . import validate as validator


def _maybe_copy_lora_to_comfy(lora_path: Path) -> str:
    """Copy a trained .safetensors into COMFY_LORA_DIR so ComfyUI can see it.
    Returns the filename ComfyUI will reference."""
    if config.COMFY_LORA_DIR and config.COMFY_LORA_DIR.exists():
        dest = config.COMFY_LORA_DIR / lora_path.name
        if lora_path.resolve() != dest.resolve():
            shutil.copy2(lora_path, dest)
            print(f"[pipeline] Copied LoRA -> {dest}")
    else:
        print("[pipeline] COMFY_LORA_DIR unset/missing; ensure the LoRA is in models/loras yourself.")
    return lora_path.name


def run(args):
    trigger = args.trigger

    if not args.skip_prep:
        if not args.src:
            raise SystemExit("--src is required unless --skip-prep is set.")
        prep.prepare(
            src=Path(args.src),
            name=args.name,
            size=args.size,
            trigger=trigger,
            caption_mode="blip" if args.vision else "template",
            overwrite=args.overwrite,
        )

    if not args.skip_train:
        code = trainer.train(args.config)
        if code != 0:
            raise SystemExit(f"Training failed (exit {code}); stopping before validation.")

    if not args.skip_validate:
        lora_name = args.lora
        if not lora_name:
            # Guess the newest .safetensors for this run from the local checkpoints dir.
            candidates = sorted(
                config.CHECKPOINTS_DIR.rglob(f"{args.name}*.safetensors"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                raise SystemExit(
                    "No trained LoRA found. Pass --lora <file> (already in models/loras) "
                    "or point CHECKPOINTS_DIR at ai-toolkit's output."
                )
            lora_name = _maybe_copy_lora_to_comfy(candidates[0])

        validator.validate(
            lora=lora_name,
            strengths=args.strengths,
            prompts=validator.DEFAULT_PROMPTS,
            trigger=trigger,
            seed=args.seed,
            run_name=args.name,
        )

    print("\n[pipeline] Complete. Review the grid under:", config.GRIDS_DIR)


def main(argv=None):
    ap = argparse.ArgumentParser(description="prep -> train -> validate a person LoRA.")
    ap.add_argument("--name", required=True, help="Project/dataset name.")
    ap.add_argument("--src", help="Raw photos folder (required unless --skip-prep).")
    ap.add_argument("--config", default="configs/person_sdxl.yaml")
    ap.add_argument("--trigger", default="ohwx person")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.6, 0.8, 1.0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora", help="LoRA filename already visible to ComfyUI (skips auto-detect).")
    ap.add_argument("--vision", action="store_true", help="BLIP auto-captioning during prep.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-prep", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
