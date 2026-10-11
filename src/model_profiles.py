"""Explicit SDXL profiles shared by training, validation, and the desktop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .toolkit_vpred import TRAINER_SOURCE, velocity_target

# Shared by the training previews and the final strength comparison. Keep the
# prompts neutral about drawing style so they can reveal what the LoRA learned.
ANIME_PREVIEW_PROMPTS = (
    (
        "masterpiece, best quality, {trigger}, solo, portrait, looking at viewer, "
        "neutral expression, simple background"
    ),
    (
        "masterpiece, best quality, {trigger}, solo, full body, standing, "
        "shirt, pants, shoes, simple background"
    ),
    (
        "masterpiece, best quality, {trigger}, solo, upper body, profile, "
        "looking to the side, simple background"
    ),
    (
        "masterpiece, best quality, {trigger}, solo, full body, running, "
        "jacket, pants, outdoors, park"
    ),
)
ANIME_PREVIEW_LABELS = ("Face", "Full body", "Profile", "Action")
ANIME_PREVIEW_NEGATIVE = (
    "worst quality, low quality, lowres, blurry, bad anatomy, extra fingers, "
    "signature, watermark, text"
)


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


def apply_training_profile(
    process: dict[str, Any], profile: ModelProfile, *, default_previews: bool = False
) -> None:
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
    sample["width"], sample["height"] = 832, 1216
    if default_previews:
        sample.pop("prompts", None)
        sample["samples"] = [{"prompt": prompt} for prompt in ANIME_PREVIEW_PROMPTS]
        sample["neg"] = ANIME_PREVIEW_NEGATIVE


def apply_validation_profile(graph: dict[str, Any], profile: ModelProfile) -> None:
    if profile.id == "sdxl":
        return
    graph["3"]["inputs"].update(
        sampler_name=profile.sampler,
        scheduler="normal",
        cfg=profile.cfg,
        steps=profile.steps,
    )
    graph["5"]["inputs"].update(width=832, height=1216)
    graph["7"]["inputs"]["text"] = ANIME_PREVIEW_NEGATIVE
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
    name = (
        Path(checkpoint.replace("\\", "/"))
        .name.lower()
        .replace("-", "")
        .replace("_", "")
    )
    if "noob" in name and "vpred" in name and not profile.v_prediction:
        return "This filename indicates NoobAI V-pred. Choose NoobAI-XL · V-pred in Connect."
    if "noob" in name and ("eps" in name or "epsilon" in name) and profile.v_prediction:
        return "This filename indicates NoobAI Epsilon. Choose NoobAI-XL · Epsilon-pred in Connect."
    return None


def training_profile_issue(profile_id: str, toolkit_dir: Path | None) -> str | None:
    """Admit only a recognized latent target or the known repairable image target."""
    if not model_profile(profile_id).v_prediction:
        return None
    source_path = toolkit_dir / TRAINER_SOURCE if toolkit_dir is not None else None
    try:
        if source_path is None:
            raise FileNotFoundError
        velocity_target(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return (
            "Choose an ai-toolkit checkout with verified V-pred latent-target support."
        )
    except ValueError:
        return (
            "This ai-toolkit V-pred training path has not been verified. "
            "Choose Epsilon or a trainer with a supported latent-target path."
        )
    return None
