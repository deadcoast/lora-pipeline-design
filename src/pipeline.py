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
import json
import re
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from . import config
from . import prepare_dataset as prep
from . import train as trainer
from . import validate as validator
from .comfy_activity import check_orphaned_comfy_prompts
from .io_utils import artifact_fingerprint, atomic_copy, atomic_write_text
from .model_profiles import MODEL_PROFILE_IDS, model_profile, profile_checkpoint_issue
from .process_control import arm_desktop_owner_guard
from .run_guardian import pipeline_run_guard

RECOVERY_SCHEMA_VERSION = 3


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


def _validation_recovery_path(name: str) -> Path:
    return config.RUN_CONFIGS_DIR / f"{name}-recovery.json"


def _write_validation_recovery(
    *,
    name: str,
    state: str,
    lora_name: str,
    installed_lora_path: Path,
    trigger: str,
    preset: str,
    attempt_id: str,
    checkpoint_path: Path | None,
    strengths: Sequence[float],
    seed: int,
    validation_signature: str,
    known_lora_fingerprint: str | None = None,
    known_checkpoint_fingerprint: str | None = None,
    updated_at: str | None = None,
    training_project_name: str = "",
) -> dict[str, object]:
    """Atomically record the installed handoff before ComfyUI validation starts."""

    if state not in {"validation_pending", "complete"}:
        raise ValueError(f"Unsupported validation recovery state: {state}")
    installed_lora = installed_lora_path.resolve()
    if not installed_lora.is_file():
        raise FileNotFoundError(f"Installed LoRA not found: {installed_lora}")
    resolved_checkpoint: Path | None = None
    checkpoint_fingerprint = ""
    if checkpoint_path is not None:
        resolved_checkpoint = checkpoint_path.resolve()
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(
                f"Training checkpoint not found: {resolved_checkpoint}"
            )
        checkpoint_fingerprint = known_checkpoint_fingerprint or artifact_fingerprint(
            resolved_checkpoint
        )
    timestamp = updated_at or datetime.now(UTC).isoformat()
    payload: dict[str, object] = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "project_name": name,
        "training_project_name": training_project_name,
        "state": state,
        "lora_name": lora_name,
        "installed_lora_path": str(installed_lora),
        "expected_grid_path": str((config.GRIDS_DIR / f"{name}_grid.png").resolve()),
        "trigger": trigger,
        "preset": preset,
        "model_profile": config.MODEL_PROFILE,
        "checkpoint": config.COMFY_SDXL_CKPT,
        "attempt_id": attempt_id,
        "comfy_installation_id": config.COMFY_INSTALLATION_ID,
        "comfy_host": config.COMFY_HOST,
        "lora_fingerprint": (
            known_lora_fingerprint or artifact_fingerprint(installed_lora)
        ),
        "checkpoint_path": (
            str(resolved_checkpoint) if resolved_checkpoint is not None else ""
        ),
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "validation_signature": validation_signature,
        "strengths": list(strengths),
        "seed": seed,
        "updated_at": timestamp,
    }
    destination = _validation_recovery_path(name)
    atomic_write_text(
        destination,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )
    return payload


def run(args: argparse.Namespace) -> Path | None:
    """Run one pipeline invocation while holding its cross-process safety leases."""

    arm_desktop_owner_guard()
    config.MODEL_PROFILE = model_profile(
        getattr(args, "model_profile", config.MODEL_PROFILE)
    ).id
    if not args.skip_train or not args.skip_validate:
        issue = profile_checkpoint_issue(config.MODEL_PROFILE, config.COMFY_SDXL_CKPT)
        if issue:
            raise ValueError(issue)
    name = config.validate_project_name(args.name)
    if args.skip_prep and args.skip_train and args.skip_validate:
        raise ValueError("All pipeline stages are skipped; there is nothing to run.")
    if args.lora and not args.skip_train:
        raise ValueError(
            "--lora selects an existing checkpoint and requires --skip-train."
        )
    if not args.skip_train and not args.skip_validate:
        mode = "full"
    elif not args.skip_train:
        mode = "train"
    elif not args.skip_validate:
        mode = "validate"
    else:
        mode = "prepare"
    use_accelerator = (
        not args.skip_train
        or not args.skip_validate
        or (not args.skip_prep and args.vision)
    )
    workspace = config.RUN_CONFIGS_DIR.resolve().parent
    dataset_dir = (config.DATASETS_DIR / name).resolve()
    mutable_roots = {
        config.DATASETS_DIR.resolve().parent,
        config.RUN_CONFIGS_DIR.resolve().parent,
    }
    resources = {
        (config.RUN_CONFIGS_DIR / f"{name}.yaml").resolve(),
        (config.RUN_CONFIGS_DIR / f"{name}-recovery.json").resolve(),
        (config.RUN_CONFIGS_DIR / f"{name}-result.json").resolve(),
    }
    if not args.skip_prep or not args.skip_train:
        resources.add(dataset_dir)
    if not args.skip_train:
        mutable_roots.add(config.CHECKPOINTS_DIR.resolve().parent)
        resources.add(config.CHECKPOINTS_DIR.resolve())
    if not args.skip_validate:
        mutable_roots.add(config.GRIDS_DIR.resolve().parent)
        resources.add(config.GRIDS_DIR.resolve())
        if args.skip_train and not args.lora:
            resources.add(config.CHECKPOINTS_DIR.resolve())
        if config.COMFY_LORA_DIR is not None:
            mutable_roots.add(config.COMFY_LORA_DIR.resolve().parent)
            resources.add(config.COMFY_LORA_DIR.resolve())
    base_model = getattr(args, "base_model", None)
    checkpoint_choice = getattr(args, "checkpoint_choice", None)
    if checkpoint_choice is not None:
        resources.add(Path(checkpoint_choice).resolve())
    if base_model is not None:
        resources.add(Path(base_model).resolve())
    with pipeline_run_guard(
        project=name,
        workspace=workspace,
        mode=mode,
        use_accelerator=use_accelerator,
        project_scopes=mutable_roots,
        resource_paths=resources,
    ):
        if use_accelerator:
            check_orphaned_comfy_prompts()
        return _run_with_guard(args)


def _run_with_guard(args: argparse.Namespace) -> Path | None:
    name = config.validate_project_name(args.name)
    training_project = getattr(args, "training_project", "")
    if training_project:
        config.validate_project_name(training_project)
    choice = getattr(args, "checkpoint_choice", None)
    if choice is not None and (
        not args.skip_prep or not args.skip_train or args.skip_validate or args.lora
    ):
        raise ValueError(
            "A checkpoint choice requires validation only, without --lora."
        )
    if choice is not None and (
        (config.RUN_CONFIGS_DIR / f"{name}-result.json").exists()
        or (config.RUN_CONFIGS_DIR / f"{name}-recovery.json").exists()
        or (config.GRIDS_DIR / f"{name}_grid.png").exists()
    ):
        raise ValueError(
            "This checkpoint choice already has outputs. Use a new version name."
        )
    attempt_id = getattr(args, "attempt_id", "") or str(uuid.uuid4())
    try:
        attempt_id = str(uuid.UUID(attempt_id))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("The run attempt identifier is invalid.") from exc
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
    installed_lora_name = ""
    installed_lora_result_path: Path | None = None
    verified_lora_fingerprint = ""
    verified_checkpoint_path: Path | None = None
    verified_checkpoint_fingerprint = ""

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
    validation_recovery: dict[str, object] | None = None
    if not args.skip_validate:
        lora_name = args.lora
        if choice is not None:
            source = Path(choice).resolve()
            expected = getattr(args, "expected_lora_fingerprint", "")
            if (
                source.suffix.lower() != ".safetensors"
                or not expected
                or artifact_fingerprint(source) != expected
            ):
                raise ValueError(
                    "The selected checkpoint changed after review. Compare it again."
                )
            if config.COMFY_LORA_DIR is None or not config.COMFY_LORA_DIR.is_dir():
                raise ValueError("Reconnect the ComfyUI LoRA folder before validation.")
            lora_name = f"{name}.safetensors"
            destination = config.COMFY_LORA_DIR / lora_name
            if destination.exists():
                raise ValueError(
                    "A LoRA already uses this choice name. Use a new version name."
                )
            atomic_copy(source, destination)
            if artifact_fingerprint(destination) != expected:
                raise ValueError(
                    "The checkpoint changed while being copied. Validation was not started."
                )
            print(f"[pipeline] Copied selected checkpoint -> {destination}")
        elif lora_name:
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
        installed_lora_name = Path(lora_name).name
        installed_lora_path = (
            config.COMFY_LORA_DIR / installed_lora_name
            if config.COMFY_LORA_DIR is not None
            else Path(installed_lora_name)
        )
        installed_lora_result_path = installed_lora_path.resolve()
        verified_lora_fingerprint = artifact_fingerprint(installed_lora_path)
        base_model = getattr(args, "base_model", None)
        if base_model is not None:
            verified_checkpoint_path = Path(base_model).resolve()
            verified_checkpoint_fingerprint = artifact_fingerprint(
                verified_checkpoint_path
            )
        checkpoint_content_identity = (
            verified_checkpoint_fingerprint or f"unverified-attempt:{attempt_id}"
        )
        validation_signature = validator.validation_signature(
            lora=installed_lora_name,
            strengths=args.strengths,
            prompts=validator.DEFAULT_PROMPTS,
            trigger=trigger,
            seed=args.seed,
            lora_content_identity=verified_lora_fingerprint,
            checkpoint_content_identity=checkpoint_content_identity,
        )
        validation_recovery = _write_validation_recovery(
            name=name,
            training_project_name=training_project,
            state="validation_pending",
            lora_name=installed_lora_name,
            installed_lora_path=installed_lora_path,
            trigger=trigger,
            preset=preset,
            attempt_id=attempt_id,
            checkpoint_path=verified_checkpoint_path,
            strengths=args.strengths,
            seed=args.seed,
            validation_signature=validation_signature,
            known_lora_fingerprint=verified_lora_fingerprint,
            known_checkpoint_fingerprint=verified_checkpoint_fingerprint,
        )
        print(
            f"[pipeline] Validation recovery saved -> {_validation_recovery_path(name)}"
        )

        grid_path = validator.validate(
            lora=lora_name,
            strengths=args.strengths,
            prompts=validator.DEFAULT_PROMPTS,
            trigger=trigger,
            seed=args.seed,
            run_name=name,
            lora_content_identity=verified_lora_fingerprint,
            checkpoint_content_identity=checkpoint_content_identity,
        )
        post_validation_lora_fingerprint = artifact_fingerprint(
            installed_lora_result_path
        )
        post_validation_checkpoint_fingerprint = (
            artifact_fingerprint(verified_checkpoint_path)
            if verified_checkpoint_path is not None
            else ""
        )
        if (
            post_validation_lora_fingerprint != verified_lora_fingerprint
            or post_validation_checkpoint_fingerprint != verified_checkpoint_fingerprint
        ):
            raise RuntimeError(
                "A LoRA or checkpoint changed while ComfyUI validation was "
                "running. The grid was kept but was not approved; retry validation "
                "with the stable model files."
            )
        verified_lora_fingerprint = post_validation_lora_fingerprint
        verified_checkpoint_fingerprint = post_validation_checkpoint_fingerprint

    if grid_path is not None:
        print(f"\n[pipeline] Complete. Review the grid: {grid_path}")
        outcome_mode = "validate" if args.skip_train else "full"
        result_path = grid_path
        workflow_path = validator.ready_workflow_path(name).resolve()
        workflow_fingerprint = artifact_fingerprint(workflow_path)
        starting_strength = validator.recommended_starting_strength(args.strengths)
    else:
        print("\n[pipeline] Complete.")
        outcome_mode = "prepare" if args.skip_train else "full"
        result_path = dataset_dir if args.skip_train else config.CHECKPOINTS_DIR
        workflow_path = None
        workflow_fingerprint = ""
        starting_strength = 0.8
    outcome = {
        "schema_version": 4,
        "model_profile": config.MODEL_PROFILE,
        "project_name": name,
        "training_project_name": training_project,
        "attempt_id": attempt_id,
        "mode": outcome_mode,
        "result_path": str(result_path.resolve()),
        "result_fingerprint": artifact_fingerprint(result_path),
        "trigger": trigger,
        "preset": preset,
        "checkpoint": config.COMFY_SDXL_CKPT,
        "lora_name": installed_lora_name,
        "workflow_path": str(workflow_path) if workflow_path is not None else "",
        "workflow_fingerprint": workflow_fingerprint,
        "comfy_installation_id": config.COMFY_INSTALLATION_ID,
        "comfy_host": config.COMFY_HOST,
        "installed_lora_path": (
            str(installed_lora_result_path)
            if installed_lora_result_path is not None
            else ""
        ),
        "lora_fingerprint": verified_lora_fingerprint,
        "checkpoint_path": (
            str(verified_checkpoint_path)
            if verified_checkpoint_path is not None
            else ""
        ),
        "checkpoint_fingerprint": verified_checkpoint_fingerprint,
        "starting_strength": starting_strength,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    atomic_write_text(
        config.RUN_CONFIGS_DIR / f"{name}-result.json",
        json.dumps(outcome, indent=2, ensure_ascii=False) + "\n",
    )
    if validation_recovery is not None:
        _write_validation_recovery(
            name=name,
            training_project_name=training_project,
            state="complete",
            lora_name=installed_lora_name,
            installed_lora_path=Path(str(validation_recovery["installed_lora_path"])),
            trigger=trigger,
            preset=preset,
            attempt_id=attempt_id,
            checkpoint_path=verified_checkpoint_path,
            strengths=args.strengths,
            seed=args.seed,
            validation_signature=str(validation_recovery["validation_signature"]),
            known_lora_fingerprint=verified_lora_fingerprint,
            known_checkpoint_fingerprint=verified_checkpoint_fingerprint,
        )
        print(
            "[pipeline] Validation recovery completed -> "
            f"{_validation_recovery_path(name)}"
        )
    return grid_path


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="prep -> train -> validate a person LoRA.")
    ap.add_argument(
        "--model-profile", choices=MODEL_PROFILE_IDS, default=config.MODEL_PROFILE
    )
    ap.add_argument(
        "--attempt-id",
        default="",
        help=argparse.SUPPRESS,
    )
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
    ap.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[0.0, 0.6, 0.8, 1.0],
    )
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
    ap.add_argument(
        "--checkpoint-choice",
        type=Path,
        help="Saved training checkpoint to validate as a new version.",
    )
    ap.add_argument(
        "--expected-lora-fingerprint",
        default="",
        help="Reviewed checkpoint content identity.",
    )
    ap.add_argument(
        "--training-project",
        default="",
        help="Original training project for checkpoint comparison.",
    )
    args = ap.parse_args(argv)
    try:
        run(args)
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
