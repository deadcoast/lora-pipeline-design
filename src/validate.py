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
import hashlib
import io
import json
import math
import tempfile
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from . import config
from .comfy_activity import check_orphaned_comfy_prompts
from .comfy_client import ComfyClient
from .io_utils import atomic_write_bytes, atomic_write_text
from .model_profiles import (
    MODEL_PROFILE_IDS,
    apply_validation_profile,
    model_profile,
    profile_checkpoint_issue,
)
from .run_guardian import active_run_guard_covers, pipeline_run_guard

WORKFLOW = config.ROOT / "workflows" / "validate_sdxl_api.json"
VALIDATION_MANIFEST_SCHEMA_VERSION = 2
VALIDATION_MANIFEST_NAME = "manifest.json"
READY_WORKFLOW_SUFFIX = "_use_in_comfyui.json"
CORE_NODE_TYPES = frozenset(
    {
        "CheckpointLoaderSimple",
        "LoraLoader",
        "CLIPTextEncode",
        "EmptyLatentImage",
        "KSampler",
        "VAEDecode",
        "SaveImage",
        "ModelSamplingDiscrete",
    }
)

_UI_NODE_SPECS: dict[str, dict[str, object]] = {
    "ModelSamplingDiscrete": {
        "pos": (410, 500),
        "size": (330, 110),
        "inputs": (("model", "MODEL"),),
        "outputs": (("MODEL", "MODEL"),),
        "widgets": ("sampling", "zsnr"),
    },
    "CheckpointLoaderSimple": {
        "pos": (20, 250),
        "size": (330, 110),
        "inputs": (),
        "outputs": (("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")),
        "widgets": ("ckpt_name",),
    },
    "LoraLoader": {
        "pos": (410, 250),
        "size": (330, 150),
        "inputs": (("model", "MODEL"), ("clip", "CLIP")),
        "outputs": (("MODEL", "MODEL"), ("CLIP", "CLIP")),
        "widgets": ("lora_name", "strength_model", "strength_clip"),
    },
    "CLIPTextEncode": {
        "pos": (800, 80),
        "size": (390, 180),
        "inputs": (("clip", "CLIP"),),
        "outputs": (("CONDITIONING", "CONDITIONING"),),
        "widgets": ("text",),
    },
    "EmptyLatentImage": {
        "pos": (810, 610),
        "size": (320, 150),
        "inputs": (),
        "outputs": (("LATENT", "LATENT"),),
        "widgets": ("width", "height", "batch_size"),
    },
    "KSampler": {
        "pos": (1260, 220),
        "size": (330, 470),
        "inputs": (
            ("model", "MODEL"),
            ("positive", "CONDITIONING"),
            ("negative", "CONDITIONING"),
            ("latent_image", "LATENT"),
        ),
        "outputs": (("LATENT", "LATENT"),),
        "widgets": (
            "seed",
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "denoise",
        ),
    },
    "VAEDecode": {
        "pos": (1660, 280),
        "size": (220, 100),
        "inputs": (("samples", "LATENT"), ("vae", "VAE")),
        "outputs": (("IMAGE", "IMAGE"),),
        "widgets": (),
    },
    "SaveImage": {
        "pos": (1950, 200),
        "size": (440, 520),
        "inputs": (("images", "IMAGE"),),
        "outputs": (),
        "widgets": ("filename_prefix",),
    },
}

# Default validation prompts. {trigger} is substituted at runtime.
DEFAULT_PROMPTS = [
    "{trigger}, cinematic portrait, soft window light, 85mm, shallow depth of field",
    "{trigger}, candid on a city street at golden hour, 35mm",
    "{trigger}, black and white studio headshot, dramatic lighting",
    "{trigger}, smiling outdoors, natural daylight, sharp focus",
]


def _validated_model_reference(value: str, label: str) -> str:
    cleaned = value.strip()
    reference = Path(cleaned.replace("\\", "/"))
    if (
        not cleaned
        or reference.is_absolute()
        or reference.name in {"", ".", ".."}
        or ".." in reference.parts
    ):
        raise ValueError(f"{label} must be a safe filename visible to ComfyUI.")
    return cleaned


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
    profile = model_profile(config.MODEL_PROFILE)
    issue = profile_checkpoint_issue(profile.id, config.COMFY_SDXL_CKPT)
    if issue:
        raise ValueError(issue)
    apply_validation_profile(loaded, profile)
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


def ready_workflow_path(run_name: str) -> Path:
    return (
        config.GRIDS_DIR
        / f"{config.validate_project_name(run_name)}{READY_WORKFLOW_SUFFIX}"
    )


def recommended_starting_strength(strengths: Sequence[float]) -> float:
    if not strengths:
        raise ValueError("At least one validation strength is required.")
    positive_strengths = [strength for strength in strengths if strength > 0]
    return min(
        positive_strengths or strengths,
        key=lambda strength: abs(strength - 0.8),
    )


def _ready_workflow_graph(
    workflow: dict[str, Any],
    *,
    checkpoint: str,
    lora: str,
    prompt: str,
    strength: float,
    seed: int,
) -> dict[str, Any]:
    """Create the exact successful API graph with friendly core-node titles."""

    graph = _build_graph(workflow, checkpoint, lora, prompt, strength, seed)
    titles = {
        "CheckpointLoaderSimple": "1 · Load the SDXL checkpoint",
        "LoraLoader": "2 · Load your MIVIM LoRA",
        "CLIPTextEncode": "Prompt text",
        "EmptyLatentImage": "Image size",
        "KSampler": "Generation settings",
        "VAEDecode": "Decode the image",
        "SaveImage": "Save your image",
        "ModelSamplingDiscrete": "Checkpoint prediction settings",
    }
    title_overrides = {
        "6": "3 · Describe the image",
        "7": "Negative prompt",
        "5": "4 · Choose image size",
        "3": "5 · Generation settings",
        "8": "6 · Decode the image",
        "9": "7 · Save your image",
    }
    for node_id, node in graph.items():
        node_type = node.get("class_type")
        if node_type not in CORE_NODE_TYPES:
            raise ValueError(
                f"Ready workflow contains unsupported node type: {node_type!r}"
            )
        node["_meta"] = {"title": title_overrides.get(node_id, titles[str(node_type)])}
    return graph


def _ui_node_spec(
    node_type: str,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    tuple[str, ...],
]:
    try:
        spec = _UI_NODE_SPECS[node_type]
    except KeyError as exc:
        raise ValueError(
            f"Ready workflow contains unsupported node type: {node_type!r}"
        ) from exc
    return (
        cast(tuple[float, float], spec["pos"]),
        cast(tuple[float, float], spec["size"]),
        cast(tuple[tuple[str, str], ...], spec["inputs"]),
        cast(tuple[tuple[str, str], ...], spec["outputs"]),
        cast(tuple[str, ...], spec["widgets"]),
    )


def _ready_ui_workflow(prompt: dict[str, Any]) -> dict[str, Any]:
    """Convert the exact executed prompt into an editable ComfyUI UI workflow."""

    node_positions: dict[str, tuple[float, float]] = {
        "4": (20, 250),
        "10": (410, 250),
        "6": (800, 50),
        "7": (800, 340),
        "5": (810, 610),
        "3": (1260, 220),
        "8": (1660, 280),
        "9": (1950, 200),
    }
    ordered_ids = sorted(prompt, key=lambda value: int(value))
    nodes: list[dict[str, Any]] = []
    nodes_by_id: dict[str, dict[str, Any]] = {}

    for order, node_id in enumerate(ordered_ids):
        prompt_node = prompt[node_id]
        node_type = prompt_node.get("class_type")
        node_inputs = prompt_node.get("inputs")
        if not isinstance(node_type, str) or not isinstance(node_inputs, dict):
            raise TypeError("Ready API prompt contains an invalid node.")
        default_pos, size, _inputs, outputs, widgets = _ui_node_spec(node_type)
        widget_values = [node_inputs[name] for name in widgets]
        if node_type == "KSampler":
            # Current and legacy ComfyUI workflows store seed control directly
            # after the seed. "fixed" preserves the graph MIVIM actually tested.
            widget_values.insert(1, "fixed")
        metadata = prompt_node.get("_meta")
        title = metadata.get("title") if isinstance(metadata, dict) else None
        node: dict[str, Any] = {
            "id": int(node_id),
            "type": node_type,
            "pos": list(node_positions.get(node_id, default_pos)),
            "size": list(size),
            "flags": {},
            "order": order,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {
                    "name": output_name,
                    "type": output_type,
                    "slot_index": slot,
                    "links": [],
                }
                for slot, (output_name, output_type) in enumerate(outputs)
            ],
            "properties": {
                "Node name for S&R": node_type,
                "cnr_id": "comfy-core",
            },
            "widgets_values": widget_values,
        }
        if isinstance(title, str) and title:
            node["title"] = title
        nodes.append(node)
        nodes_by_id[node_id] = node

    links: list[list[object]] = []
    next_link_id = 1
    for target_id in ordered_ids:
        prompt_node = prompt[target_id]
        node_type = cast(str, prompt_node["class_type"])
        node_inputs = cast(dict[str, Any], prompt_node["inputs"])
        _pos, _size, linked_inputs, _outputs, _widgets = _ui_node_spec(node_type)
        target_node = nodes_by_id[target_id]
        for target_slot, (input_name, input_type) in enumerate(linked_inputs):
            value = node_inputs.get(input_name)
            if (
                not isinstance(value, list)
                or len(value) != 2
                or not isinstance(value[0], str)
                or not isinstance(value[1], int)
            ):
                raise ValueError(
                    f"Ready API prompt is missing the {input_name!r} link "
                    f"for node {target_id}."
                )
            source_id, source_slot = value
            try:
                source_node = nodes_by_id[source_id]
                source_output = source_node["outputs"][source_slot]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(
                    f"Ready API prompt contains an invalid link into node {target_id}."
                ) from exc
            if source_output["type"] != input_type:
                raise ValueError(
                    f"Ready API prompt link type mismatch for node {target_id}."
                )
            link_id = next_link_id
            next_link_id += 1
            source_output["links"].append(link_id)
            target_node["inputs"].append(
                {
                    "name": input_name,
                    "type": input_type,
                    "link": link_id,
                }
            )
            links.append(
                [
                    link_id,
                    int(source_id),
                    source_slot,
                    int(target_id),
                    target_slot,
                    input_type,
                ]
            )

    workflow: dict[str, Any] = {
        "last_node_id": max(int(node_id) for node_id in ordered_ids),
        "last_link_id": next_link_id - 1,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {
            "ds": {"scale": 0.72, "offset": [120, 100]},
            "info": {
                "name": "MIVIM tested LoRA workflow",
                "author": "MIVIM",
                "description": (
                    "The exact core-node graph used for this validation result."
                ),
                "version": "1",
                "created": "",
                "modified": "",
                "software": "MIVIM LoRA Studio",
            },
        },
        "version": 0.4,
    }
    if _ui_workflow_to_prompt(workflow) != prompt:
        raise RuntimeError(
            "The editable ComfyUI workflow did not match the tested API graph."
        )
    return workflow


def _ui_workflow_to_prompt(workflow: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a core-node API prompt to verify the exported UI workflow."""

    raw_nodes = workflow.get("nodes")
    raw_links = workflow.get("links")
    if not isinstance(raw_nodes, list) or not isinstance(raw_links, list):
        raise TypeError("Editable ComfyUI workflow is missing nodes or links.")
    nodes: dict[str, dict[str, Any]] = {}
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            raise TypeError("Editable ComfyUI workflow contains an invalid node.")
        node_id = raw_node.get("id")
        node_type = raw_node.get("type")
        widgets_values = raw_node.get("widgets_values", [])
        if (
            not isinstance(node_id, int)
            or not isinstance(node_type, str)
            or not isinstance(widgets_values, list)
        ):
            raise TypeError("Editable ComfyUI workflow contains an invalid node.")
        _pos, _size, _inputs, _outputs, widgets = _ui_node_spec(node_type)
        values = list(widgets_values)
        if node_type == "KSampler":
            if len(values) != len(widgets) + 1 or values[1] != "fixed":
                raise ValueError("Editable KSampler must preserve the tested seed.")
            values.pop(1)
        if len(values) != len(widgets):
            raise ValueError(f"Editable {node_type} node has invalid widget values.")
        inputs = dict(zip(widgets, values, strict=True))
        metadata: dict[str, str] = {}
        title = raw_node.get("title")
        if isinstance(title, str) and title:
            metadata["title"] = title
        node: dict[str, Any] = {"class_type": node_type, "inputs": inputs}
        if metadata:
            node["_meta"] = metadata
        nodes[str(node_id)] = node

    for raw_link in raw_links:
        if (
            not isinstance(raw_link, list)
            or len(raw_link) != 6
            or not all(isinstance(value, int) for value in raw_link[:5])
            or not isinstance(raw_link[5], str)
        ):
            raise TypeError("Editable ComfyUI workflow contains an invalid link.")
        _link_id, source_id, source_slot, target_id, target_slot, link_type = raw_link
        try:
            target = nodes[str(target_id)]
            target_type = cast(str, target["class_type"])
            _pos, _size, linked_inputs, _outputs, _widgets = _ui_node_spec(target_type)
            input_name, expected_type = linked_inputs[target_slot]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                "Editable ComfyUI workflow link target is invalid."
            ) from exc
        if link_type != expected_type or str(source_id) not in nodes:
            raise ValueError("Editable ComfyUI workflow link type is invalid.")
        target["inputs"][input_name] = [str(source_id), source_slot]
    return nodes


def _stable_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validation_signature(
    *,
    workflow: dict[str, Any],
    comfy_host: str,
    checkpoint: str,
    lora: str,
    lora_content_identity: str,
    checkpoint_content_identity: str,
    strengths: Sequence[float],
    prompts: Sequence[str],
    trigger: str,
    seed: int,
) -> tuple[str, dict[str, object]]:
    """Identify every input that can change a generated validation cell."""

    inputs: dict[str, object] = {
        "validation_schema_version": VALIDATION_MANIFEST_SCHEMA_VERSION,
        "workflow_sha256": _stable_digest(workflow),
        "comfy_host": comfy_host,
        "checkpoint": checkpoint,
        "lora": lora,
        "lora_content_identity": lora_content_identity,
        "checkpoint_content_identity": checkpoint_content_identity,
        "strengths": list(strengths),
        "prompts": list(prompts),
        "negative_prompt": NEG,
        "trigger": trigger,
        "seed": seed,
    }
    return _stable_digest(inputs), inputs


def validation_signature(
    *,
    lora: str,
    strengths: Sequence[float],
    prompts: Sequence[str],
    trigger: str,
    seed: int,
    lora_content_identity: str,
    checkpoint_content_identity: str,
) -> str:
    """Return the shipped workflow's exact semantic validation signature."""

    lora = _validated_model_reference(lora, "LoRA")
    _validated_model_reference(config.COMFY_SDXL_CKPT, "Checkpoint")
    lora_content_identity = lora_content_identity.strip()
    checkpoint_content_identity = checkpoint_content_identity.strip()
    if not lora_content_identity or not checkpoint_content_identity:
        raise ValueError(
            "LoRA and checkpoint content identities are required for validation reuse."
        )
    normalized_trigger = trigger.strip()
    workflow = load_workflow()
    formatted_prompts = [
        prompt.replace("{trigger}", normalized_trigger) for prompt in prompts
    ]
    signature, _inputs = _validation_signature(
        workflow=workflow,
        comfy_host=config.COMFY_HOST,
        checkpoint=config.COMFY_SDXL_CKPT,
        lora=lora,
        lora_content_identity=lora_content_identity,
        checkpoint_content_identity=checkpoint_content_identity,
        strengths=list(strengths),
        prompts=formatted_prompts,
        trigger=normalized_trigger,
        seed=seed,
    )
    return signature


def _verified_image(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[str, int, int] | None:
    """Return a reusable image fingerprint only after Pillow fully decodes it."""

    try:
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            return None
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
        if width <= 0 or height <= 0:
            return None
    except (OSError, SyntaxError, ValueError):
        return None
    return digest, width, height


def _normalise_cell_image(source: Path, destination: Path) -> tuple[str, int, int]:
    """Decode one ComfyUI output and atomically promote a deterministic PNG."""

    try:
        with Image.open(source) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                raise ValueError("image dimensions must be greater than zero")
            normalized = image.convert("RGB")
            encoded = io.BytesIO()
            normalized.save(encoded, "PNG")
    except (OSError, SyntaxError, ValueError) as exc:
        raise RuntimeError(
            f"ComfyUI returned an invalid validation image: {source}"
        ) from exc
    atomic_write_bytes(destination, encoded.getvalue())
    verified = _verified_image(destination)
    if verified is None:
        raise RuntimeError(
            f"Validation image could not be verified after saving: {destination}"
        )
    return verified


def _new_manifest(
    *,
    signature: str,
    signature_inputs: dict[str, object],
    run_name: str,
    total_cells: int,
) -> dict[str, Any]:
    return {
        "schema_version": VALIDATION_MANIFEST_SCHEMA_VERSION,
        "signature": signature,
        "run_name": run_name,
        "signature_inputs": signature_inputs,
        "status": "in_progress",
        "total_cells": total_cells,
        "processed_cells": 0,
        "completed_cells": 0,
        "cells": {},
        "updated_at": datetime.now(UTC).isoformat(),
    }


def _load_manifest(path: Path, signature: str) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > 1024 * 1024:
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(loaded, dict)
        or loaded.get("schema_version") != VALIDATION_MANIFEST_SCHEMA_VERSION
        or loaded.get("signature") != signature
        or not isinstance(loaded.get("cells"), dict)
    ):
        return None
    return loaded


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    atomic_write_text(
        path,
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def _cell_record_matches(
    record: object,
    *,
    cell_id: str,
    cell_path: Path,
    attempt_dir: Path,
    strength_index: int,
    prompt_index: int,
    strength: float,
    prompt: str,
) -> tuple[str, int, int] | None:
    if not isinstance(record, dict):
        return None
    expected_path = cell_path.relative_to(attempt_dir).as_posix()
    expected_digest = record.get("sha256")
    if (
        record.get("id") != cell_id
        or record.get("path") != expected_path
        or record.get("strength_index") != strength_index
        or record.get("prompt_index") != prompt_index
        or record.get("strength") != strength
        or record.get("prompt") != prompt
        or not isinstance(expected_digest, str)
    ):
        return None
    return _verified_image(cell_path, expected_sha256=expected_digest)


def _record_cell(
    manifest: dict[str, Any],
    *,
    cell_id: str,
    cell_path: Path,
    attempt_dir: Path,
    strength_index: int,
    prompt_index: int,
    strength: float,
    prompt: str,
    fingerprint: tuple[str, int, int],
) -> None:
    digest, width, height = fingerprint
    cells = manifest["cells"]
    if not isinstance(cells, dict):
        raise TypeError("Validation manifest cells must be a mapping.")
    cells[cell_id] = {
        "id": cell_id,
        "path": cell_path.relative_to(attempt_dir).as_posix(),
        "strength_index": strength_index,
        "prompt_index": prompt_index,
        "strength": strength,
        "prompt": prompt,
        "sha256": digest,
        "width": width,
        "height": height,
    }


def _grid(
    images: list[Path],
    cols: int,
    cell: int,
    labels: list[str],
    out_path: Path,
    *,
    row_labels: Sequence[str] = (),
    column_labels: Sequence[str] = (),
    png_text: dict[str, str] | None = None,
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
    metadata = None
    if png_text:
        metadata = PngImagePlugin.PngInfo()
        for key, value in png_text.items():
            metadata.add_text(key, value)
    canvas.save(encoded, "PNG", pnginfo=metadata)
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
    *,
    lora_content_identity: str | None = None,
    checkpoint_content_identity: str | None = None,
) -> Path:
    run_name = config.validate_project_name(run_name)
    workspace = config.RUN_CONFIGS_DIR.resolve().parent
    resources = {config.GRIDS_DIR.resolve()}
    if config.COMFY_LORA_DIR is not None:
        resources.add(config.COMFY_LORA_DIR.resolve())
    if not active_run_guard_covers(
        project=run_name,
        workspace=workspace,
        use_accelerator=True,
        resource_paths=resources,
    ):
        with pipeline_run_guard(
            project=run_name,
            workspace=workspace,
            mode="validate",
            use_accelerator=True,
            resource_paths=resources,
        ):
            check_orphaned_comfy_prompts()
            return _validate_with_guard(
                lora=lora,
                strengths=strengths,
                prompts=prompts,
                trigger=trigger,
                seed=seed,
                run_name=run_name,
                workflow_path=workflow_path,
                lora_content_identity=lora_content_identity,
                checkpoint_content_identity=checkpoint_content_identity,
            )
    check_orphaned_comfy_prompts()
    return _validate_with_guard(
        lora=lora,
        strengths=strengths,
        prompts=prompts,
        trigger=trigger,
        seed=seed,
        run_name=run_name,
        workflow_path=workflow_path,
        lora_content_identity=lora_content_identity,
        checkpoint_content_identity=checkpoint_content_identity,
    )


def _validate_with_guard(
    lora: str,
    strengths: Sequence[float],
    prompts: Sequence[str],
    trigger: str,
    seed: int,
    run_name: str,
    workflow_path: Path = WORKFLOW,
    *,
    lora_content_identity: str | None = None,
    checkpoint_content_identity: str | None = None,
) -> Path:
    run_name = config.validate_project_name(run_name)
    lora = _validated_model_reference(lora, "LoRA")
    _validated_model_reference(config.COMFY_SDXL_CKPT, "Checkpoint")
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
    unverified_identity = f"unverified-attempt:{uuid.uuid4()}"
    lora_content_identity = (
        lora_content_identity.strip()
        if lora_content_identity is not None
        else unverified_identity
    )
    checkpoint_content_identity = (
        checkpoint_content_identity.strip()
        if checkpoint_content_identity is not None
        else unverified_identity
    )
    if not lora_content_identity or not checkpoint_content_identity:
        raise ValueError("Validation model content identities cannot be empty.")
    if unverified_identity in {
        lora_content_identity,
        checkpoint_content_identity,
    }:
        print(
            "[validate] Exact model identity was not supplied; earlier cells "
            "will not be reused for this invocation."
        )

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
    ready_strength = recommended_starting_strength(strengths)
    ready_graph = _ready_workflow_graph(
        base,
        checkpoint=config.COMFY_SDXL_CKPT,
        lora=lora,
        prompt=formatted_prompts[0],
        strength=ready_strength,
        seed=seed,
    )
    signature, signature_inputs = _validation_signature(
        workflow=base,
        comfy_host=config.COMFY_HOST,
        checkpoint=config.COMFY_SDXL_CKPT,
        lora=lora,
        lora_content_identity=lora_content_identity,
        checkpoint_content_identity=checkpoint_content_identity,
        strengths=strengths,
        prompts=formatted_prompts,
        trigger=trigger,
        seed=seed,
    )
    work_dir = config.GRIDS_DIR / run_name
    attempt_dir = work_dir / "attempts" / signature
    cells_dir = attempt_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = attempt_dir / VALIDATION_MANIFEST_NAME
    expected = len(strengths) * len(formatted_prompts)
    manifest = _load_manifest(manifest_path, signature) or _new_manifest(
        signature=signature,
        signature_inputs=signature_inputs,
        run_name=run_name,
        total_cells=expected,
    )
    expected_cell_ids = {
        f"strength-{strength_index:03d}-prompt-{prompt_index:03d}"
        for strength_index in range(len(strengths))
        for prompt_index in range(len(formatted_prompts))
    }
    loaded_cells = manifest["cells"]
    if not isinstance(loaded_cells, dict):
        raise TypeError("Validation manifest cells must be a mapping.")
    manifest["cells"] = {
        key: value
        for key, value in loaded_cells.items()
        if isinstance(key, str) and key in expected_cell_ids
    }
    manifest["status"] = "in_progress"
    manifest["total_cells"] = expected
    manifest["processed_cells"] = 0
    manifest.pop("grid", None)
    _write_manifest(manifest_path, manifest)
    cells: list[Path] = []
    labels: list[str] = []

    cell_number = 0
    for strength_index, s in enumerate(strengths):
        for pi, prompt in enumerate(formatted_prompts):
            cell_number += 1
            cell_id = f"strength-{strength_index:03d}-prompt-{pi:03d}"
            cell_path = cells_dir / f"{cell_id}.png"
            manifest_cells = manifest["cells"]
            if not isinstance(manifest_cells, dict):
                raise TypeError("Validation manifest cells must be a mapping.")
            fingerprint = _cell_record_matches(
                manifest_cells.get(cell_id),
                cell_id=cell_id,
                cell_path=cell_path,
                attempt_dir=attempt_dir,
                strength_index=strength_index,
                prompt_index=pi,
                strength=s,
                prompt=prompt,
            )
            action = "reused"
            if fingerprint is None:
                manifest_cells.pop(cell_id, None)
                manifest["status"] = "in_progress"
                manifest["completed_cells"] = len(manifest_cells)
                _write_manifest(manifest_path, manifest)
                graph = (
                    ready_graph
                    if pi == 0 and s == ready_strength
                    else _build_graph(
                        base,
                        config.COMFY_SDXL_CKPT,
                        lora,
                        prompt,
                        s,
                        seed,
                    )
                )
                with tempfile.TemporaryDirectory(
                    prefix=f"mivim-{cell_id}-"
                ) as incoming:
                    saved = client.run_and_save(
                        graph,
                        Path(incoming),
                        prefix=f"{cell_id}-",
                    )
                    if len(saved) != 1:
                        raise RuntimeError(
                            f"ComfyUI produced {len(saved)} images for strength={s}, "
                            f"prompt#{pi}; expected exactly one."
                        )
                    fingerprint = _normalise_cell_image(saved[0], cell_path)
                _record_cell(
                    manifest,
                    cell_id=cell_id,
                    cell_path=cell_path,
                    attempt_dir=attempt_dir,
                    strength_index=strength_index,
                    prompt_index=pi,
                    strength=s,
                    prompt=prompt,
                    fingerprint=fingerprint,
                )
                action = "created"
            cells.append(cell_path)
            labels.append("")
            manifest["processed_cells"] = cell_number
            manifest["completed_cells"] = len(manifest_cells)
            _write_manifest(manifest_path, manifest)
            print(
                f"[validate] cell {cell_number}/{expected} {action} "
                f"strength={s:g} prompt#{pi}"
            )

    if len(cells) != expected:
        raise RuntimeError(
            f"Validation produced {len(cells)} of {expected} expected images."
        )
    grid_path = config.GRIDS_DIR / f"{run_name}_grid.png"
    attempt_grid_path = attempt_dir / "grid.png"
    ready_prompt_json = json.dumps(
        ready_graph,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    ready_ui_workflow = _ready_ui_workflow(ready_graph)
    ready_workflow_json = json.dumps(
        ready_ui_workflow,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    _grid(
        cells,
        cols=len(formatted_prompts),
        cell=384,
        labels=labels,
        out_path=attempt_grid_path,
        row_labels=[
            "No LoRA · baseline" if strength == 0 else f"LoRA strength {strength:g}"
            for strength in strengths
        ],
        column_labels=prompt_labels,
        png_text={
            "prompt": ready_prompt_json,
            "workflow": ready_workflow_json,
        },
    )
    grid_fingerprint = _verified_image(attempt_grid_path)
    if grid_fingerprint is None:
        raise RuntimeError(
            f"Validation grid could not be verified after saving: {attempt_grid_path}"
        )
    grid_bytes = attempt_grid_path.read_bytes()
    atomic_write_bytes(grid_path, grid_bytes)
    workflow_path = ready_workflow_path(run_name)
    atomic_write_text(workflow_path, ready_workflow_json + "\n")
    try:
        with Image.open(grid_path) as published_grid:
            embedded_prompt = published_grid.info.get("prompt")
            embedded_workflow = published_grid.info.get("workflow")
        saved_workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "The ready-to-load ComfyUI workflow could not be verified."
        ) from exc
    if (
        embedded_prompt != ready_prompt_json
        or embedded_workflow != ready_workflow_json
        or saved_workflow != ready_ui_workflow
        or _ui_workflow_to_prompt(saved_workflow) != ready_graph
    ):
        raise RuntimeError(
            "The ready-to-load workflow did not match the validation grid metadata."
        )
    grid_digest, grid_width, grid_height = grid_fingerprint
    manifest["status"] = "complete"
    manifest["processed_cells"] = expected
    manifest["completed_cells"] = expected
    manifest["grid"] = {
        "path": attempt_grid_path.relative_to(attempt_dir).as_posix(),
        "published_path": str(grid_path.resolve()),
        "sha256": grid_digest,
        "width": grid_width,
        "height": grid_height,
    }
    manifest["ready_workflow"] = {
        "path": str(workflow_path.resolve()),
        "sha256": hashlib.sha256(workflow_path.read_bytes()).hexdigest(),
        "api_prompt_sha256": hashlib.sha256(
            ready_prompt_json.encode("utf-8")
        ).hexdigest(),
        "starting_strength": ready_strength,
        "embedded_prompt_in_grid": True,
        "embedded_workflow_in_grid": True,
    }
    _write_manifest(manifest_path, manifest)
    print(f"\n[validate] Grid written: {grid_path}")
    print(f"[validate] Ready workflow written: {workflow_path}")
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
    ap.add_argument(
        "--model-profile", choices=MODEL_PROFILE_IDS, default=config.MODEL_PROFILE
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--name", default=None, help="Run name for output folder (default: lora stem)."
    )
    args = ap.parse_args(argv)
    config.MODEL_PROFILE = model_profile(args.model_profile).id

    run_name = args.name or Path(args.lora).stem
    try:
        run_name = config.validate_project_name(run_name)
        workspace = config.RUN_CONFIGS_DIR.resolve().parent
        project_scopes = [config.GRIDS_DIR.resolve().parent]
        resources = {config.GRIDS_DIR.resolve()}
        if config.COMFY_LORA_DIR is not None:
            project_scopes.append(config.COMFY_LORA_DIR.resolve().parent)
            resources.add(config.COMFY_LORA_DIR.resolve())
        with pipeline_run_guard(
            project=run_name,
            workspace=workspace,
            mode="validate",
            use_accelerator=True,
            project_scopes=project_scopes,
            resource_paths=resources,
        ):
            check_orphaned_comfy_prompts()
            validate(
                lora=args.lora,
                strengths=args.strengths,
                prompts=DEFAULT_PROMPTS,
                trigger=args.trigger,
                seed=args.seed,
                run_name=run_name,
            )
    except (
        FileNotFoundError,
        RuntimeError,
        TimeoutError,
        TypeError,
        ValueError,
    ) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
