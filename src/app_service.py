"""Qt-free application services for the MIVIM LoRA desktop companion.

The command-line modules remain useful automation adapters.  This module is the
user-facing application boundary: it turns paths and API responses into plain
readiness items and constructs the exact pipeline process the desktop UI runs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from PIL import Image, UnidentifiedImageError

from . import config
from .app_models import AppSettings, ProjectDraft
from .comfy_desktop import (
    ComfyApiStatus,
    ComfyDesktopDiscovery,
    discover_comfy_desktop,
    probe_comfy_api,
)

ReadinessState = Literal["ready", "warning", "error"]
RunMode = Literal["full", "prepare", "validate"]

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


@dataclass(frozen=True, slots=True)
class SourceSummary:
    """Fast, reviewable summary of a selected photo folder."""

    folder: Path
    image_count: int
    unreadable_count: int
    small_count: int
    duplicate_count: int

    @property
    def usable_count(self) -> int:
        return self.image_count - self.unreadable_count

    @property
    def unique_usable_count(self) -> int:
        return self.usable_count - self.duplicate_count


@dataclass(frozen=True, slots=True)
class ReadinessItem:
    """One user-facing setup check and its direct recovery action."""

    code: str
    title: str
    state: ReadinessState
    detail: str
    action: str = ""


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Capability-level readiness; preparation can work without a trainer."""

    items: tuple[ReadinessItem, ...]
    can_prepare: bool
    can_run_full: bool
    comfy_status: ComfyApiStatus
    source: SourceSummary | None


@dataclass(frozen=True, slots=True)
class PipelineCommand:
    """The exact child process launched by the desktop application."""

    program: str
    arguments: tuple[str, ...]
    environment: dict[str, str]
    working_directory: Path
    mode: RunMode
    expected_result: Path


def _normalise_host(host: str) -> str:
    value = host.strip().removeprefix("http://").removeprefix("https://")
    return value.rstrip("/")


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _probe_trainer_runtime(executable: Path) -> tuple[bool, str]:
    """Exercise the same Python runtime the real trainer process will use."""

    script = (
        "import torch; "
        "print('MIVIM_CUDA_READY' if torch.cuda.is_available() "
        "else 'MIVIM_CUDA_MISSING')"
    )
    try:
        result = subprocess.run(
            [str(executable), "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if sys.platform == "win32"
                else 0
            ),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"The trainer Python could not be tested: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        reason = detail[-1] if detail else f"exit code {result.returncode}"
        return False, f"PyTorch could not load in the trainer environment: {reason}"
    if "MIVIM_CUDA_READY" not in result.stdout:
        return False, "PyTorch loaded, but it cannot see a CUDA-capable NVIDIA GPU."
    return True, "PyTorch and CUDA were verified in the selected trainer environment."


def _find_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def inspect_source(folder: Path, *, target_size: int = 1024) -> SourceSummary:
    """Inspect real inputs without modifying them.

    Verifying every selected image is intentionally part of the product flow:
    casual users should learn about broken or undersized inputs before a long
    training run, not from a traceback after it starts.
    """

    images = _find_images(folder)
    unreadable = 0
    small = 0
    duplicate = 0
    digests: set[str] = set()
    for path in images:
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            if max(width, height) < target_size:
                small += 1
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            fingerprint = digest.hexdigest()
            if fingerprint in digests:
                duplicate += 1
            else:
                digests.add(fingerprint)
        except (OSError, UnidentifiedImageError, ValueError):
            unreadable += 1
    return SourceSummary(folder, len(images), unreadable, small, duplicate)


def _trainer_python(toolkit: Path) -> Path | None:
    candidates = (
        toolkit / "venv" / "Scripts" / "python.exe",
        toolkit / ".venv" / "Scripts" / "python.exe",
        toolkit / "venv" / "bin" / "python",
        toolkit / ".venv" / "bin" / "python",
    )
    return next((path for path in candidates if path.is_file()), None)


def _directory_is_writable(path: Path) -> tuple[bool, str | None]:
    """Verify the actual handoff folder with a short-lived local probe."""

    try:
        handle, probe = tempfile.mkstemp(prefix=".mivim-write-check-", dir=path)
        os.close(handle)
        Path(probe).unlink()
    except OSError as exc:
        return False, str(exc)
    return True, None


def discover_ai_toolkit(root: Path = config.ROOT) -> tuple[Path | None, Path | None]:
    """Check a small set of conventional locations; never scan the whole disk."""

    raw_env = os.environ.get("AI_TOOLKIT_DIR", "").strip()
    candidates = [
        Path(raw_env) if raw_env else None,
        root.parent / "ai-toolkit",
        Path.home() / "ai-toolkit",
        Path.home() / "github" / "ai-toolkit",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "run.py").is_file():
            return resolved, _trainer_python(resolved)
    return None, None


class ApplicationService:
    """Coordinates discovery, checks, and executable pipeline requests."""

    def __init__(
        self,
        *,
        root: Path = config.ROOT,
        appdata: Path | None = None,
    ) -> None:
        self.root = root.resolve()
        self.appdata = appdata

    def discover_comfy(self) -> ComfyDesktopDiscovery:
        return discover_comfy_desktop(appdata=self.appdata)

    def project_path(self, value: Path) -> Path:
        return value if value.is_absolute() else self.root / value

    def resolve_checkpoint_path(
        self,
        discovery: ComfyDesktopDiscovery | None,
        checkpoint: str,
    ) -> Path | None:
        """Resolve one API checkpoint name inside Desktop's configured roots."""

        if discovery is None or not checkpoint.strip():
            return None
        relative = Path(checkpoint.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            return None
        for model_root in discovery.model_roots:
            root = model_root.checkpoints.resolve()
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            if candidate.is_file():
                return candidate
        return None

    def apply_discovery(
        self,
        settings: AppSettings,
        discovery: ComfyDesktopDiscovery | None = None,
    ) -> AppSettings:
        """Fill only missing/default setup values from read-only discovery."""

        applied, _status = self.setup_snapshot(settings, discovery)
        return applied

    def setup_snapshot(
        self,
        settings: AppSettings,
        discovery: ComfyDesktopDiscovery | None = None,
    ) -> tuple[AppSettings, ComfyApiStatus]:
        """Apply discovery and return the same live probe used for the decision."""

        found = discovery or self.discover_comfy()
        host_candidates = tuple(found.host_candidates)
        requested_host = _normalise_host(settings.comfy_host)
        ordered_hosts = (
            (*host_candidates, requested_host)
            if requested_host == "127.0.0.1:8188"
            else (requested_host, *host_candidates)
        )
        probe_order = tuple(dict.fromkeys(value for value in ordered_hosts if value))
        live = probe_comfy_api(probe_order, timeout=0.35)
        host = (
            live.host
            if live.reachable
            else (probe_order[0] if probe_order else requested_host)
        )

        lora_dir = settings.comfy_lora_dir
        if lora_dir is None:
            model_root = next((item for item in found.model_roots if item.exists), None)
            if model_root is None and found.model_roots:
                model_root = found.model_roots[0]
            if model_root is not None:
                lora_dir = model_root.loras

        checkpoint = settings.comfy_checkpoint
        if live.checkpoints:
            if checkpoint not in live.checkpoints:
                checkpoint = next(
                    (
                        candidate
                        for candidate in live.checkpoints
                        if "xl" in candidate.casefold()
                    ),
                    live.checkpoints[0],
                )
        elif not checkpoint:
            local_names: list[str] = []
            for model_root in found.model_roots:
                if model_root.checkpoints.is_dir():
                    local_names.extend(
                        str(path.relative_to(model_root.checkpoints)).replace("\\", "/")
                        for path in model_root.checkpoints.rglob("*")
                        if path.is_file()
                        and path.suffix.lower() in {".safetensors", ".ckpt", ".pt"}
                    )
            if local_names:
                checkpoint = min(set(local_names), key=str.casefold)
        training_base_model = self.resolve_checkpoint_path(found, checkpoint)
        if training_base_model is None:
            training_base_model = settings.training_base_model

        toolkit_dir = settings.ai_toolkit_dir
        toolkit_python = settings.ai_toolkit_python
        if toolkit_dir is None or toolkit_python is None:
            found_toolkit, found_python = discover_ai_toolkit(self.root)
            toolkit_dir = toolkit_dir or found_toolkit
            toolkit_python = toolkit_python or found_python

        return (
            replace(
                settings,
                comfy_host=host or "127.0.0.1:8188",
                comfy_lora_dir=lora_dir,
                comfy_checkpoint=checkpoint,
                training_base_model=training_base_model,
                ai_toolkit_dir=toolkit_dir,
                ai_toolkit_python=toolkit_python,
            ),
            live,
        )

    def readiness(
        self,
        settings: AppSettings,
        draft: ProjectDraft,
    ) -> ReadinessReport:
        items: list[ReadinessItem] = []
        source: SourceSummary | None = None
        project_name_valid = True

        try:
            config.validate_project_name(draft.project_name)
        except ValueError as exc:
            project_name_valid = False
            items.append(
                ReadinessItem(
                    "project",
                    "Project name",
                    "error",
                    str(exc),
                    "Use a short name such as alex_v1.",
                )
            )
        else:
            items.append(
                ReadinessItem(
                    "project",
                    "Project name",
                    "ready",
                    f"Outputs will be grouped under “{draft.project_name}”.",
                )
            )

        if not project_name_valid:
            items.append(
                ReadinessItem(
                    "collision",
                    "Existing project data",
                    "warning",
                    "Choose a valid project name before existing data can be checked.",
                )
            )
        else:
            existing_dataset = (
                self.project_path(settings.datasets_dir) / draft.project_name
            )
            if existing_dataset.exists() and not draft.overwrite_existing:
                items.append(
                    ReadinessItem(
                        "collision",
                        "Existing project data",
                        "error",
                        f"Prepared data already exists for “{draft.project_name}”.",
                        "Use a new version name or allow rebuilding",
                    )
                )
            elif existing_dataset.exists():
                items.append(
                    ReadinessItem(
                        "collision",
                        "Existing project data",
                        "warning",
                        "The existing prepared photos will be rebuilt for this run.",
                    )
                )
            else:
                items.append(
                    ReadinessItem(
                        "collision",
                        "Existing project data",
                        "ready",
                        "This project name has a new, empty output location.",
                    )
                )

        if not draft.trigger.strip():
            items.append(
                ReadinessItem(
                    "trigger",
                    "Identity phrase",
                    "error",
                    "The identity phrase is empty.",
                    "Keep the recommended phrase or enter a unique one.",
                )
            )
        else:
            items.append(
                ReadinessItem(
                    "trigger",
                    "Identity phrase",
                    "ready",
                    f"“{draft.trigger.strip()}” will identify this person.",
                )
            )

        source_path = draft.source_dir
        if source_path is None or not source_path.is_dir():
            items.append(
                ReadinessItem(
                    "photos",
                    "Training photos",
                    "error",
                    "Choose a folder containing photos of one person.",
                    "Choose photo folder",
                )
            )
        else:
            source = inspect_source(
                source_path,
                target_size=draft.preset.image_size,
            )
            if source.usable_count == 0:
                items.append(
                    ReadinessItem(
                        "photos",
                        "Training photos",
                        "error",
                        "No readable JPG, PNG, WebP, BMP, or TIFF images were found.",
                        "Choose another folder",
                    )
                )
            elif source.duplicate_count:
                items.append(
                    ReadinessItem(
                        "photos",
                        "Training photos",
                        "error",
                        f"{source.duplicate_count} exact duplicate photo(s) were found. "
                        "Duplicates can bias identity training.",
                        "Remove duplicate copies, then check again",
                    )
                )
            elif source.unreadable_count:
                items.append(
                    ReadinessItem(
                        "photos",
                        "Training photos",
                        "warning",
                        f"{source.usable_count} usable; "
                        f"{source.unreadable_count} unreadable file(s) will be skipped.",
                        "Review the photo folder",
                    )
                )
            elif source.usable_count < 10:
                items.append(
                    ReadinessItem(
                        "photos",
                        "Training photos",
                        "warning",
                        f"{source.usable_count} usable photo(s). "
                        "15–40 varied photos usually gives a better identity.",
                        "Add more varied photos",
                    )
                )
            elif source.usable_count > 100:
                items.append(
                    ReadinessItem(
                        "photos",
                        "Training photos",
                        "error",
                        f"{source.usable_count} photos were found recursively. This "
                        "looks like a broad photo library rather than one focused "
                        "identity dataset.",
                        "Choose a folder with 15–40 varied photos",
                    )
                )
            elif source.usable_count > 60:
                items.append(
                    ReadinessItem(
                        "photos",
                        "Training photos",
                        "warning",
                        f"{source.usable_count} photos were found. A focused set of "
                        "15–40 varied images is usually easier to train well.",
                        "Review the selected folder",
                    )
                )
            else:
                note = f"{source.usable_count} usable photo(s)" + (
                    f"; {source.small_count} will be enlarged."
                    if source.small_count
                    else "."
                )
                items.append(ReadinessItem("photos", "Training photos", "ready", note))

        if not draft.use_vision_captions:
            items.append(
                ReadinessItem(
                    "captions",
                    "Photo descriptions",
                    "ready",
                    "Using the built-in local caption template.",
                )
            )
        elif _module_available("torch") and _module_available("transformers"):
            items.append(
                ReadinessItem(
                    "captions",
                    "Automatic photo descriptions",
                    "ready",
                    "The local vision-captioning components are installed.",
                )
            )
        else:
            items.append(
                ReadinessItem(
                    "captions",
                    "Automatic photo descriptions",
                    "error",
                    "The optional PyTorch and Transformers components are not installed.",
                    "Turn off automatic descriptions",
                )
            )

        host = _normalise_host(settings.comfy_host)
        comfy = probe_comfy_api((host,), timeout=0.75)
        if comfy.reachable:
            items.append(
                ReadinessItem(
                    "comfy",
                    "ComfyUI",
                    "ready",
                    f"Connected securely on this computer at {comfy.host}.",
                )
            )
        else:
            items.append(
                ReadinessItem(
                    "comfy",
                    "ComfyUI",
                    "error",
                    "Comfy Desktop is installed but its selected instance is not running.",
                    "Open Comfy Desktop, launch an instance, then retry",
                )
            )

        selected_checkpoint = settings.comfy_checkpoint.strip()
        if not selected_checkpoint:
            items.append(
                ReadinessItem(
                    "checkpoint",
                    "SDXL base model",
                    "error",
                    "No checkpoint is selected.",
                    "Install or choose an SDXL checkpoint in ComfyUI",
                )
            )
        elif comfy.reachable and selected_checkpoint not in comfy.checkpoints:
            sample = ", ".join(comfy.checkpoints[:3])
            detail = f"ComfyUI cannot see “{selected_checkpoint}”."
            if sample:
                detail += f" Available models include: {sample}."
            items.append(
                ReadinessItem(
                    "checkpoint",
                    "SDXL base model",
                    "error",
                    detail,
                    "Choose a model ComfyUI can see",
                )
            )
        elif "xl" not in selected_checkpoint.casefold():
            items.append(
                ReadinessItem(
                    "checkpoint",
                    "SDXL base model",
                    "warning",
                    f"ComfyUI reports “{selected_checkpoint}”, but does not expose "
                    "enough metadata to confirm that it is SDXL.",
                    "Confirm this is an SDXL checkpoint",
                )
            )
        else:
            items.append(
                ReadinessItem(
                    "checkpoint",
                    "SDXL base model",
                    "ready",
                    selected_checkpoint,
                )
            )

        base_model = settings.training_base_model
        if base_model is None:
            items.append(
                ReadinessItem(
                    "base_model",
                    "Training base model",
                    "error",
                    "The selected ComfyUI checkpoint file could not be located for "
                    "the trainer.",
                    "Choose the local SDXL checkpoint file",
                )
            )
        elif not base_model.is_file():
            items.append(
                ReadinessItem(
                    "base_model",
                    "Training base model",
                    "error",
                    f"The local checkpoint no longer exists: {base_model}",
                    "Choose the local SDXL checkpoint file",
                )
            )
        else:
            items.append(
                ReadinessItem(
                    "base_model",
                    "Training base model",
                    "ready",
                    f"Training and validation will use the same local model: {base_model}",
                )
            )

        lora_dir = settings.comfy_lora_dir
        if lora_dir is None:
            items.append(
                ReadinessItem(
                    "lora_dir",
                    "ComfyUI LoRA library",
                    "error",
                    "The LoRA library could not be discovered.",
                    "Choose the ComfyUI models\\loras folder",
                )
            )
        elif not lora_dir.is_dir():
            items.append(
                ReadinessItem(
                    "lora_dir",
                    "ComfyUI LoRA library",
                    "error",
                    f"The folder does not exist: {lora_dir}",
                    "Choose another folder",
                )
            )
        else:
            writable, write_error = _directory_is_writable(lora_dir)
            if not writable:
                items.append(
                    ReadinessItem(
                        "lora_dir",
                        "ComfyUI LoRA library",
                        "error",
                        f"The folder cannot accept a trained model: {write_error}",
                        "Choose a writable models\\loras folder",
                    )
                )
            else:
                items.append(
                    ReadinessItem(
                        "lora_dir",
                        "ComfyUI LoRA library",
                        "ready",
                        str(lora_dir),
                    )
                )

        toolkit = settings.ai_toolkit_dir
        toolkit_python = settings.ai_toolkit_python
        if toolkit is None or not (toolkit / "run.py").is_file():
            items.append(
                ReadinessItem(
                    "trainer",
                    "LoRA trainer",
                    "error",
                    "ai-toolkit has not been connected yet.",
                    "Choose the ai-toolkit folder",
                )
            )
        elif toolkit_python is None or not toolkit_python.is_file():
            items.append(
                ReadinessItem(
                    "trainer",
                    "LoRA trainer",
                    "error",
                    "The trainer's isolated Python environment was not found.",
                    "Choose its venv Python",
                )
            )
        else:
            trainer_ready, trainer_detail = _probe_trainer_runtime(toolkit_python)
            items.append(
                ReadinessItem(
                    "trainer",
                    "LoRA trainer",
                    "ready" if trainer_ready else "error",
                    trainer_detail
                    if trainer_ready
                    else f"{trainer_detail} ({toolkit})",
                    "" if trainer_ready else "Repair the ai-toolkit environment",
                )
            )

        storage_root = self.project_path(settings.checkpoints_dir)
        storage_probe = next(
            (
                parent
                for parent in (storage_root, *storage_root.parents)
                if parent.exists()
            ),
            self.root,
        )
        writable, write_error = _directory_is_writable(storage_probe)
        free_gib = shutil.disk_usage(storage_probe).free / (1024**3)
        storage_state: ReadinessState
        if not writable or free_gib < 5:
            storage_state = "error"
        elif free_gib < 20:
            storage_state = "warning"
        else:
            storage_state = "ready"
        if not writable:
            storage_detail = f"The project workspace is not writable: {write_error}"
        else:
            storage_detail = f"{free_gib:.1f} GB free near the project outputs." + (
                " Training can require 20 GB or more." if free_gib < 20 else ""
            )
        items.append(
            ReadinessItem(
                "storage",
                "Storage",
                storage_state,
                storage_detail,
                (
                    "Choose another workspace"
                    if not writable
                    else ("Free some disk space" if free_gib < 20 else "")
                ),
            )
        )
        items.append(
            ReadinessItem(
                "training_storage",
                "Training storage reserve",
                "ready" if writable and free_gib >= 20 else "error",
                (
                    f"{free_gib:.1f} GB is available for checkpoints and caches."
                    if writable and free_gib >= 20
                    else "A full training run requires at least 20 GB free in the "
                    "selected workspace."
                ),
                "Free space or choose another workspace"
                if not writable or free_gib < 20
                else "",
            )
        )

        by_code = {item.code: item for item in items}
        prepare_codes = {
            "project",
            "collision",
            "trigger",
            "photos",
            "captions",
            "storage",
        }
        full_codes = prepare_codes | {
            "comfy",
            "checkpoint",
            "lora_dir",
            "trainer",
            "training_storage",
        }
        full_codes.add("base_model")
        can_prepare = all(
            by_code.get(code) is not None and by_code[code].state != "error"
            for code in prepare_codes
        )
        can_run_full = all(
            by_code.get(code) is not None and by_code[code].state != "error"
            for code in full_codes
        )
        return ReadinessReport(
            tuple(items),
            can_prepare=can_prepare,
            can_run_full=can_run_full,
            comfy_status=comfy,
            source=source,
        )

    def build_command(
        self,
        settings: AppSettings,
        draft: ProjectDraft,
        *,
        mode: RunMode,
    ) -> PipelineCommand:
        """Build the same supported pipeline invocation documented for the CLI."""

        config.validate_project_name(draft.project_name)
        if draft.source_dir is None or not draft.source_dir.is_dir():
            raise FileNotFoundError(
                f"Training photo folder not found: {draft.source_dir}"
            )
        worker_prefix = (
            ["--pipeline-worker"]
            if os.environ.get("MIVIM_PACKAGED_APP") == "1"
            else ["-m", "src.pipeline"]
        )
        args = [
            *worker_prefix,
            "--src",
            str(draft.source_dir),
            "--name",
            draft.project_name,
            "--trigger",
            draft.trigger.strip(),
            "--size",
            str(draft.preset.image_size),
            "--preset",
            draft.preset.value,
            "--strengths",
            *(str(value) for value in draft.preset.validation_strengths),
        ]
        if draft.use_vision_captions:
            args.append("--vision")
        if draft.overwrite_existing:
            args.append("--overwrite")
        if settings.training_base_model is not None:
            args.extend(("--base-model", str(settings.training_base_model)))
        if mode == "prepare":
            args.extend(("--skip-train", "--skip-validate"))

        environment = dict(os.environ)
        environment.update(settings.environment_overrides())
        environment["PYTHONUNBUFFERED"] = "1"
        environment["MIVIM_DESKTOP_RUN"] = "1"
        expected = (
            self.project_path(settings.grids_dir) / f"{draft.project_name}_grid.png"
            if mode == "full"
            else self.project_path(settings.datasets_dir) / draft.project_name
        )
        return PipelineCommand(
            sys.executable,
            tuple(args),
            environment,
            self.root,
            mode,
            expected,
        )

    def build_validation_command(
        self,
        settings: AppSettings,
        draft: ProjectDraft,
        lora_name: str,
    ) -> PipelineCommand:
        """Build a validation-only retry after training already installed a LoRA."""

        config.validate_project_name(draft.project_name)
        cleaned_lora = Path(lora_name).name
        if (
            cleaned_lora != lora_name
            or Path(cleaned_lora).suffix.casefold() != ".safetensors"
        ):
            raise ValueError(f"Invalid installed LoRA filename: {lora_name}")
        worker_prefix = (
            ["--pipeline-worker"]
            if os.environ.get("MIVIM_PACKAGED_APP") == "1"
            else ["-m", "src.pipeline"]
        )
        arguments = (
            *worker_prefix,
            "--name",
            draft.project_name,
            "--trigger",
            draft.trigger.strip(),
            "--preset",
            draft.preset.value,
            "--strengths",
            *(str(value) for value in draft.preset.validation_strengths),
            "--skip-prep",
            "--skip-train",
            "--lora",
            cleaned_lora,
        )
        environment = dict(os.environ)
        environment.update(settings.environment_overrides())
        environment["PYTHONUNBUFFERED"] = "1"
        environment["MIVIM_DESKTOP_RUN"] = "1"
        expected = (
            self.project_path(settings.grids_dir) / f"{draft.project_name}_grid.png"
        )
        return PipelineCommand(
            sys.executable,
            tuple(arguments),
            environment,
            self.root,
            "validate",
            expected,
        )


def stage_from_log(line: str) -> tuple[int, str] | None:
    """Translate actual pipeline output into stable, human-readable progress."""

    lowered = line.lower()
    if "[prepare]" in lowered:
        return 12, "Preparing photos"
    if "dataset '" in lowered and "prepared at:" in lowered:
        return 28, "Photos ready"
    if "[train] run config:" in lowered:
        return 34, "Preparing the trainer"
    if "[train]" in lowered:
        return 42, "Training the LoRA"
    if "[pipeline] copied lora" in lowered:
        return 78, "Installing in ComfyUI"
    if "[ok] strength=" in lowered:
        return 86, "Creating comparison images"
    if "[validate] grid written:" in lowered:
        return 96, "Building the result"
    if "[pipeline] complete" in lowered:
        return 100, "Complete"
    return None
