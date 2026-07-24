"""Generate a validation contact-sheet for a trained LoRA via ComfyUI.

Loads the API-format workflow, then sweeps over (prompt x LoRA strength),
runs each cell through ComfyUI, and stitches the results into one grid PNG so
you can eyeball likeness vs. flexibility at a glance.

Usage (from repo root, with ComfyUI running):
    python -m src.validate --lora person_v1.safetensors
    python -m src.validate --lora person_v1.safetensors --strengths 0.6 0.8 1.0
    python -m src.validate --lora person_v1.safetensors --trigger "sks person"

The LoRA file must be visible to ComfyUI (i.e. inside models/loras).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from PIL import Image, ImageDraw

from . import config
from .comfy_client import ComfyClient

WORKFLOW = config.ROOT / "workflows" / "validate_sdxl_api.json"

# Default validation prompts. {trigger} is substituted at runtime.
DEFAULT_PROMPTS = [
    "{trigger}, cinematic portrait, soft window light, 85mm, shallow depth of field",
    "{trigger}, candid on a city street at golden hour, 35mm",
    "{trigger}, black and white studio headshot, dramatic lighting",
    "{trigger}, smiling outdoors, natural daylight, sharp focus",
]
NEG = "lowres, bad anatomy, blurry, deformed, extra fingers, watermark, text"


def _build_graph(base: dict, ckpt: str, lora: str, prompt: str, strength: float, seed: int) -> dict:
    g = copy.deepcopy(base)
    g["4"]["inputs"]["ckpt_name"] = ckpt
    g["10"]["inputs"]["lora_name"] = lora
    g["10"]["inputs"]["strength_model"] = strength
    g["10"]["inputs"]["strength_clip"] = strength
    g["6"]["inputs"]["text"] = prompt
    g["7"]["inputs"]["text"] = NEG
    g["3"]["inputs"]["seed"] = seed
    g["9"]["inputs"]["filename_prefix"] = "validate"
    return g


def _grid(images: list[Path], cols: int, cell: int, labels: list[str], out_path: Path):
    rows = (len(images) + cols - 1) // cols
    pad, header = 8, 22
    W = cols * cell + (cols + 1) * pad
    H = rows * (cell + header) + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), (24, 24, 27))
    draw = ImageDraw.Draw(canvas)
    for i, img_path in enumerate(images):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + header + pad)
        try:
            im = Image.open(img_path).convert("RGB")
            im.thumbnail((cell, cell))
            canvas.paste(im, (x, y + header))
        except Exception as e:
            draw.text((x, y + header), f"(missing: {e})", fill=(200, 80, 80))
        if i < len(labels):
            draw.text((x + 2, y + 4), labels[i][:48], fill=(220, 220, 220))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def validate(lora: str, strengths, prompts, trigger: str, seed: int, run_name: str) -> Path:
    base = json.loads(WORKFLOW.read_text())
    client = ComfyClient()
    if not client.is_up():
        raise SystemExit(
            f"ComfyUI is not reachable at {config.COMFY_HOST}.\n"
            f"Start Comfy Desktop (or `python main.py`) and confirm the API port."
        )

    prompts = [p.format(trigger=trigger) for p in prompts]
    work_dir = config.GRIDS_DIR / run_name
    cells: list[Path] = []
    labels: list[str] = []

    for s in strengths:
        for pi, prompt in enumerate(prompts):
            graph = _build_graph(base, config.COMFY_SDXL_CKPT, lora, prompt, s, seed)
            saved = client.run_and_save(graph, work_dir, prefix=f"s{s}_p{pi}_")
            if saved:
                cells.append(saved[0])
                labels.append(f"str {s} · p{pi}")
                print(f"  [ok] strength={s} prompt#{pi} -> {saved[0].name}")

    grid_path = config.GRIDS_DIR / f"{run_name}_grid.png"
    _grid(cells, cols=len(prompts), cell=384, labels=labels, out_path=grid_path)
    print(f"\n[validate] Grid written: {grid_path}")
    return grid_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate a LoRA validation grid via ComfyUI.")
    ap.add_argument("--lora", required=True, help="LoRA filename as seen by ComfyUI (in models/loras).")
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.6, 0.8, 1.0])
    ap.add_argument("--trigger", default="ohwx person")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None, help="Run name for output folder (default: lora stem).")
    args = ap.parse_args(argv)

    run_name = args.name or Path(args.lora).stem
    validate(
        lora=args.lora,
        strengths=args.strengths,
        prompts=DEFAULT_PROMPTS,
        trigger=args.trigger,
        seed=args.seed,
        run_name=run_name,
    )


if __name__ == "__main__":
    main()
