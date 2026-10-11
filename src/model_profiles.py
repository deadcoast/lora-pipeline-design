"""Explicit SDXL profiles shared by training, validation, and the desktop."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelProfile:
    id: str
    label: str
    description: str
    v_prediction: bool = False
    sampler: str = "dpmpp_2m"
    cfg: float = 6.0
    steps: int = 28


MODEL_PROFILES = (
    ModelProfile(
        "sdxl", "SDXL · Standard", "Use for standard Epsilon SDXL checkpoints."
    ),
    ModelProfile(
        "noobai-xl-epsilon",
        "NoobAI-XL · Epsilon-pred",
        "Anime illustration with Epsilon prediction. "
        "Choose the Epsilon checkpoint from the model's download page.",
        sampler="euler_ancestral",
        cfg=5.5,
    ),
    ModelProfile(
        "noobai-xl-vpred",
        "NoobAI-XL · V-pred",
        "Anime illustration with V-prediction. "
        "Requires the V-pred checkpoint; Studio applies its sampling settings.",
        v_prediction=True,
        sampler="euler",
        cfg=4.5,
        steps=30,
    ),
)
MODEL_PROFILE_IDS = tuple(profile.id for profile in MODEL_PROFILES)


def model_profile(value: str) -> ModelProfile:
    for profile in MODEL_PROFILES:
        if profile.id == value:
            return profile
    raise ValueError(
        f"Unknown model profile {value!r}. Choose a model profile in Connect."
    )


def apply_training_profile(process: dict[str, Any], profile: ModelProfile) -> None:
    if profile.id == "sdxl":
        return
    process["model"]["is_v_pred"] = profile.v_prediction
    # DDPM supplies add_noise/get_velocity for training. Preview sampling uses
    # Euler independently; the V-pred launcher supplies ZSNR to both schedulers.
    process["train"]["noise_scheduler"] = "ddpm"
    sample = process["sample"]
    sample["sampler"] = "euler" if profile.v_prediction else "euler_a"
    sample["guidance_scale"] = profile.cfg
    sample["sample_steps"] = profile.steps
    sample["width"] = sample["height"] = 1024


def apply_validation_profile(graph: dict[str, Any], profile: ModelProfile) -> None:
    if profile.id == "sdxl":
        return
    graph["3"]["inputs"].update(
        sampler_name=profile.sampler,
        scheduler="normal",
        cfg=profile.cfg,
        steps=profile.steps,
    )
    graph["11"] = {
        "class_type": "ModelSamplingDiscrete",
        "inputs": {
            "model": ["10", 0],
            "sampling": "v_prediction" if profile.v_prediction else "eps",
            "zsnr": profile.v_prediction,
        },
    }
    graph["3"]["inputs"]["model"] = ["11", 0]


def profile_checkpoint_issue(profile_id: str, checkpoint: str) -> str | None:
    """Catch an obvious mismatch without treating filenames as model metadata."""
    profile = model_profile(profile_id)
    name = checkpoint.lower().replace("-", "").replace("_", "")
    if "noob" in name and "vpred" in name and not profile.v_prediction:
        return "This filename indicates NoobAI V-pred. Choose NoobAI-XL · V-pred in Connect."
    if "noob" in name and ("eps" in name or "epsilon" in name) and profile.v_prediction:
        return "This filename indicates NoobAI Epsilon. Choose NoobAI-XL · Epsilon-pred in Connect."
    return None


def training_profile_issue(profile_id: str, toolkit_dir: Path | None) -> str | None:
    """Admit V-pred only when the trainer uses latents for its velocity target.

    Known ai-toolkit revisions use batch.tensor here, which is an RGB tensor
    or None when latent caching is enabled. Never spend a run on that path.
    This is a compatibility check, not evidence of model/GPU qualification.
    """
    if not model_profile(profile_id).v_prediction:
        return None
    source_path = (
        toolkit_dir / "extensions_built_in/sd_trainer/SDTrainer.py"
        if toolkit_dir is not None
        else None
    )
    try:
        if source_path is None:
            raise FileNotFoundError
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return (
            "Choose an ai-toolkit checkout with verified V-pred latent-target support."
        )
    targets = [
        call.args[0]
        for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        and function.name == "calculate_loss"
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "get_velocity"
        and call.args
    ]
    if any(
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "batch"
        and target.attr == "tensor"
        for target in targets
    ):
        return (
            "This ai-toolkit uses image tensors for V-pred training targets; "
            "cached-latent training would fail. Choose Epsilon or a compatible "
            "trainer checkout. V-pred validation remains available through the CLI."
        )
    if not any(
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "batch"
        and target.attr == "latents"
        for target in targets
    ):
        return (
            "This ai-toolkit V-pred training path has not been verified. "
            "Choose Epsilon or a trainer with a supported latent-target path."
        )
    return None
