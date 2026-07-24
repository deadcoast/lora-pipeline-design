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
import io
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from . import config
from .comfy_client import ComfyClient
from .io_utils import atomic_write_bytes

WORKFLOW = config.ROOT / "workflows" / "validate_sdxl_api.json"

# Default validation prompts. {trigger} is substituted at runtime.
DEFAULT_PROMPTS = [
    "{trigger}, cinematic portrait, soft window light, 85mm, shallow depth of field",
    "{trigger}, candid on a city street at golden hour, 35mm",
    "{trigger}, black and white studio headshot, dramatic lighting",
    "{trigger}, smiling outdoors, natural daylight, sharp focus",
]
DEFAULT_PROMPT_LABELS = ("Portrait", "City street", "Studio", "Outdoor")
NEG = "lowres, bad anatomy, blurry, deformed, extra fingers, watermark, text"


def _validate_workflow(base: dict[str, Any]) -> None:
    required = {
        "4": ("CheckpointLoaderSimple", ("ckpt_name",)),
        "10": (
            "LoraLoader",
            ("lora_name", "strength_model", "strength_clip", "model", "clip"),
        ),
        "6": ("CLIPTextEncode", ("text", "clip")),
        "7": ("CLIPTextEncode", ("text", "clip")),
        "5": ("EmptyLatentImage", ("width", "height", "batch_size")),
        "3": (
            "KSampler",
            (
                "seed",
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
                "denoise",
                "model",
                "positive",
                "negative",
                "latent_image",
            ),
        ),
        "8": ("VAEDecode", ("samples", "vae")),
        "9": ("SaveImage", ("filename_prefix", "images")),
    }
    for node_id, (class_type, inputs) in required.items():
        node = base.get(node_id)
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            raise TypeError(
                f"Validation workflow is missing node {node_id} with an inputs mapping."
            )
        if node.get("class_type") != class_type:
            raise ValueError(
                f"Validation workflow node {node_id} must be {class_type}, "
                f"got {node.get('class_type')!r}."
            )
        missing = [key for key in inputs if key not in node["inputs"]]
        if missing:
            raise ValueError(
                f"Validation workflow node {node_id} is missing inputs: "
                f"{', '.join(missing)}"
            )

    expected_links = {
        ("10", "model"): ["4", 0],
        ("10", "clip"): ["4", 1],
        ("6", "clip"): ["10", 1],
        ("7", "clip"): ["10", 1],
        ("3", "model"): ["10", 0],
        ("3", "positive"): ["6", 0],
        ("3", "negative"): ["7", 0],
        ("3", "latent_image"): ["5", 0],
        ("8", "samples"): ["3", 0],
        ("8", "vae"): ["4", 2],
        ("9", "images"): ["8", 0],
    }
    for (node_id, input_name), expected in expected_links.items():
        actual = base[node_id]["inputs"][input_name]
        if actual != expected:
            raise ValueError(
                f"Validation workflow node {node_id} input {input_name!r} "
                f"must connect to {expected}, got {actual!r}."
            )
    if base["5"]["inputs"]["batch_size"] != 1:
        raise ValueError("Validation workflow batch_size must be 1.")


def load_workflow(path: Path = WORKFLOW) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Validation workflow not found: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid validation workflow JSON in {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise TypeError(f"Validation workflow {path} must be a JSON object.")
    _validate_workflow(loaded)
    return loaded


def _build_graph(
    base: dict[str, Any],
    ckpt: str,
    lora: str,
    prompt: str,
    strength: float,
    seed: int,
) -> dict[str, Any]:
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


def _grid(
    images: list[Path],
    cols: int,
    cell: int,
    labels: list[str],
    out_path: Path,
    *,
    row_labels: Sequence[str] = (),
    column_labels: Sequence[str] = (),
) -> Path:
    if not images:
        raise ValueError("Cannot build a validation grid without images.")
    if cols <= 0 or cell <= 0:
        raise ValueError("Grid columns and cell size must be greater than zero.")
    if len(labels) != len(images):
        raise ValueError("Each validation image must have exactly one label.")
    rows = (len(images) + cols - 1) // cols
    if row_labels and len(row_labels) != rows:
        raise ValueError("Each validation row must have exactly one label.")
    if column_labels and len(column_labels) != cols:
        raise ValueError("Each validation column must have exactly one label.")
    pad = 10
    cell_header = 30 if any(labels) else 0
    axis_top = 56 if column_labels else 0
    axis_left = 230 if row_labels else 0
    W = axis_left + cols * cell + (cols + 1) * pad
    H = axis_top + rows * (cell + cell_header) + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), (24, 24, 27))
    draw = ImageDraw.Draw(canvas)
    column_font = ImageFont.load_default(size=26)
    row_font = ImageFont.load_default(size=24)
    cell_font = ImageFont.load_default(size=18)
    for column, label in enumerate(column_labels):
        x = axis_left + pad + column * (cell + pad)
        draw.text(
            (x + 5, 13),
            label[:28],
            fill=(238, 238, 242),
            font=column_font,
        )
    for row, label in enumerate(row_labels):
        y = axis_top + pad + row * (cell + cell_header + pad)
        draw.text(
            (14, y + 9),
            label[:24],
            fill=(158, 231, 215),
            font=row_font,
        )
    for i, img_path in enumerate(images):
        r, c = divmod(i, cols)
        x = axis_left + pad + c * (cell + pad)
        y = axis_top + pad + r * (cell + cell_header + pad)
        with Image.open(img_path) as source:
            im = source.convert("RGB")
            im.thumbnail((cell, cell))
        image_x = x + (cell - im.width) // 2
        image_y = y + cell_header + (cell - im.height) // 2
        canvas.paste(im, (image_x, image_y))
        if labels[i]:
            draw.text(
                (x + 3, y + 4),
                labels[i][:48],
                fill=(220, 220, 220),
                font=cell_font,
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = io.BytesIO()
    canvas.save(encoded, "PNG")
    atomic_write_bytes(out_path, encoded.getvalue())
    return out_path


def validate(
    lora: str,
    strengths: Sequence[float],
    prompts: Sequence[str],
    trigger: str,
    seed: int,
    run_name: str,
    workflow_path: Path = WORKFLOW,
) -> Path:
    run_name = config.validate_project_name(run_name)
    if not lora.strip():
        raise ValueError("LoRA filename cannot be empty.")
    strengths = list(strengths)
    if not strengths or any(not math.isfinite(value) for value in strengths):
        raise ValueError("At least one finite LoRA strength is required.")
    if not prompts or any(not prompt.strip() for prompt in prompts):
        raise ValueError("At least one non-empty validation prompt is required.")
    if not 0 <= seed <= 2**64 - 1:
        raise ValueError("Validation seed must be between 0 and 2^64 - 1.")

    if not trigger.strip():
        raise ValueError("Validation trigger cannot be empty.")
    trigger = trigger.strip()

    base = load_workflow(workflow_path)
    client = ComfyClient()
    if not client.is_up():
        raise RuntimeError(
            f"ComfyUI is not reachable at {config.COMFY_HOST}.\n"
            f"Start Comfy Desktop (or `python main.py`) and confirm the API port."
        )

    formatted_prompts = [p.replace("{trigger}", trigger) for p in prompts]
    prompt_labels = (
        list(DEFAULT_PROMPT_LABELS)
        if list(prompts) == DEFAULT_PROMPTS
        else [f"Prompt {index + 1}" for index in range(len(prompts))]
    )
    work_dir = config.GRIDS_DIR / run_name
    cells: list[Path] = []
    labels: list[str] = []

    for s in strengths:
        for pi, prompt in enumerate(formatted_prompts):
            graph = _build_graph(base, config.COMFY_SDXL_CKPT, lora, prompt, s, seed)
            saved = client.run_and_save(graph, work_dir, prefix=f"s{s}_p{pi}_")
            if len(saved) == 1:
                cells.append(saved[0])
                labels.append("")
                print(f"  [ok] strength={s} prompt#{pi} -> {saved[0].name}")
            else:
                raise RuntimeError(
                    f"ComfyUI produced {len(saved)} images for strength={s}, "
                    f"prompt#{pi}; expected exactly one."
                )

    expected = len(strengths) * len(formatted_prompts)
    if len(cells) != expected:
        raise RuntimeError(
            f"Validation produced {len(cells)} of {expected} expected images."
        )
    grid_path = config.GRIDS_DIR / f"{run_name}_grid.png"
    _grid(
        cells,
        cols=len(formatted_prompts),
        cell=384,
        labels=labels,
        out_path=grid_path,
        row_labels=[
            "No LoRA · baseline" if strength == 0 else f"LoRA strength {strength:g}"
            for strength in strengths
        ],
        column_labels=prompt_labels,
    )
    print(f"\n[validate] Grid written: {grid_path}")
    return grid_path


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Generate a LoRA validation grid via ComfyUI."
    )
    ap.add_argument(
        "--lora",
        required=True,
        help="LoRA filename as seen by ComfyUI (in models/loras).",
    )
    ap.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[0.0, 0.6, 0.8, 1.0],
    )
    ap.add_argument("--trigger", default=config.DEFAULT_TRIGGER)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--name", default=None, help="Run name for output folder (default: lora stem)."
    )
    args = ap.parse_args(argv)

    run_name = args.name or Path(args.lora).stem
    try:
        validate(
            lora=args.lora,
            strengths=args.strengths,
            prompts=DEFAULT_PROMPTS,
            trigger=args.trigger,
            seed=args.seed,
            run_name=run_name,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
