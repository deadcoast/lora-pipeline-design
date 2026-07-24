"""Qt-free application services for the MIVIM LoRA desktop companion.

The command-line modules remain useful automation adapters.  This module is the
user-facing application boundary: it turns paths and API responses into plain
readiness items and constructs the exact pipeline process the desktop UI runs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

from . import config
from .app_models import (
    AppSettings,
    ProjectDraft,
    TrainingPreset,
    ValidationRecovery,
)
from .comfy_desktop import (
    ComfyApiStatus,
    ComfyDesktopDiscovery,
    discover_comfy_desktop,
    probe_comfy_api,
    select_comfy_installation,
)
from .dataset_pairs import (
    SUPPORTED_DATASET_IMAGE_EXTENSIONS,
    DatasetPairInventory,
    scan_dataset_pairs,
)

ReadinessState = Literal["ready", "warning", "error"]
RunMode = Literal["full", "prepare", "validate"]
SourceIssueKind = Literal[
    "unreadable",
    "small",
    "duplicate",
    "near_duplicate",
    "blurry",
]
SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    (*SUPPORTED_DATASET_IMAGE_EXTENSIONS, ".heic", ".heif")
)


@dataclass(frozen=True, slots=True)
class SourceIssue:
    """One concrete source-photo concern that the desktop can show by filename."""

    kind: SourceIssueKind
    path: Path
    detail: str
    related_path: Path | None = None


@dataclass(frozen=True, slots=True)
class SourceSummary:
    """Fast, reviewable summary of a selected photo folder."""

    folder: Path
    image_count: int
    unreadable_count: int
    small_count: int
    duplicate_count: int
    issues: tuple[SourceIssue, ...] = ()

    @property
    def usable_count(self) -> int:
        return self.image_count - self.unreadable_count

    @property
    def unique_usable_count(self) -> int:
        return self.usable_count - self.duplicate_count

    @property
    def blurry_count(self) -> int:
        return sum(issue.kind == "blurry" for issue in self.issues)

    @property
    def near_duplicate_count(self) -> int:
        return sum(issue.kind == "near_duplicate" for issue in self.issues)


@dataclass(frozen=True, slots=True)
class PreparedDatasetSummary:
    """Reviewable state for safely resuming a previously prepared dataset."""

    folder: Path
    inventory: DatasetPairInventory
    empty_caption_files: tuple[Path, ...]
    missing_trigger_caption_files: tuple[Path, ...]
    unreadable_image_files: tuple[Path, ...]
    manifest_error: str | None

    @property
    def image_count(self) -> int:
        return len(self.inventory.image_files)

    @property
    def caption_count(self) -> int:
        return len(self.inventory.caption_files)

    @property
    def missing_caption_count(self) -> int:
        return len(self.inventory.images_without_captions)

    @property
    def empty_caption_count(self) -> int:
        return len(self.empty_caption_files)

    @property
    def missing_trigger_count(self) -> int:
        return len(self.missing_trigger_caption_files)

    @property
    def unreadable_count(self) -> int:
        return len(self.unreadable_image_files)

    @property
    def file_problem_messages(self) -> tuple[str, ...]:
        messages = list(self.inventory.problem_messages())
        messages.extend(f"{path.name} is empty." for path in self.empty_caption_files)
        messages.extend(
            f"{path.name} is without the identity phrase."
            for path in self.missing_trigger_caption_files
        )
        messages.extend(
            f"{path.name} cannot be opened as a training image."
            for path in self.unreadable_image_files
        )
        return tuple(messages)

    @property
    def problem_detail(self) -> str:
        messages = list(self.file_problem_messages)
        if self.manifest_error is not None:
            messages.append(self.manifest_error)
        return " ".join(messages) or "The prepared files are incomplete."

    @property
    def ready(self) -> bool:
        return (
            self.inventory.structure_ready
            and not self.empty_caption_files
            and not self.missing_trigger_caption_files
            and not self.unreadable_image_files
            and self.manifest_error is None
        )


@dataclass(frozen=True, slots=True)
class ProjectArtifactSummary:
    """Existing project-owned files that could be mixed into a new run."""

    dataset: Path | None
    checkpoints: tuple[Path, ...]
    validation_grid: Path | None
    validation_work_dir: Path | None
    run_config: Path | None
    result_record: Path | None
    installed_loras: tuple[Path, ...]

    @property
    def non_dataset_paths(self) -> tuple[Path, ...]:
        optional = (
            self.validation_grid,
            self.validation_work_dir,
            self.run_config,
            self.result_record,
        )
        return (
            *self.checkpoints,
            *(path for path in optional if path is not None),
            *self.installed_loras,
        )

    @property
    def has_non_dataset_outputs(self) -> bool:
        return bool(self.non_dataset_paths)

    def describe_non_dataset_outputs(self) -> str:
        """Return a compact, plain-language list for the readiness screen."""

        descriptions: list[str] = []
        if self.checkpoints:
            descriptions.append(
                f"{len(self.checkpoints)} training checkpoint"
                f"{'s' if len(self.checkpoints) != 1 else ''}"
            )
        if self.validation_grid is not None:
            descriptions.append("a comparison grid")
        if self.validation_work_dir is not None:
            descriptions.append("validation work files")
        if self.run_config is not None:
            descriptions.append("a trainer run config")
        if self.result_record is not None:
            descriptions.append("a completed-run record")
        if self.installed_loras:
            descriptions.append(
                f"{len(self.installed_loras)} installed ComfyUI LoRA"
                f"{'s' if len(self.installed_loras) != 1 else ''}"
            )
        return ", ".join(descriptions)


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
    prepared: PreparedDatasetSummary | None = None


@dataclass(frozen=True, slots=True)
class PipelineCommand:
    """The exact child process launched by the desktop application."""

    program: str
    arguments: tuple[str, ...]
    environment: dict[str, str]
    working_directory: Path
    mode: RunMode
    expected_result: Path


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    """Durable worker result consumed by the desktop handoff."""

    project_name: str
    mode: Literal["prepare", "full", "validate"]
    result_path: Path
    trigger: str
    preset: str
    checkpoint: str
    lora_name: str
    completed_at: str


def _normalise_host(host: str) -> str:
    value = host.strip().removeprefix("http://").removeprefix("https://")
    return value.rstrip("/")


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


_TRAINER_PROBE_PREFIX = "MIVIM_TRAINER_PROBE="


def _trainer_probe_result(
    result: subprocess.CompletedProcess[str],
) -> tuple[bool, str]:
    """Parse the isolated trainer probe without trusting incidental torch output."""

    payload: object | None = None
    for line in reversed(result.stdout.splitlines()):
        if not line.startswith(_TRAINER_PROBE_PREFIX):
            continue
        try:
            payload = json.loads(line.removeprefix(_TRAINER_PROBE_PREFIX))
        except json.JSONDecodeError:
            payload = None
        break

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        reason = detail[-1] if detail else f"exit code {result.returncode}"
        return False, f"PyTorch could not load in the trainer environment: {reason}"
    if not isinstance(payload, dict):
        return (
            False,
            (
                "The trainer's PyTorch check did not return usable GPU details. "
                "Reinstall PyTorch in the selected ai-toolkit environment."
            ),
        )

    status = payload.get("status")
    torch_version = str(payload.get("torch_version") or "unknown version")
    cuda_version = str(payload.get("cuda_version") or "unknown CUDA build")
    gpu_name = str(payload.get("gpu_name") or "the selected NVIDIA GPU")
    capability = payload.get("capability")
    architecture = "unknown architecture"
    if (
        isinstance(capability, list)
        and len(capability) == 2
        and all(isinstance(part, int) for part in capability)
    ):
        architecture = f"sm_{capability[0]}{capability[1]}"
    raw_arches = payload.get("arch_list")
    arches = (
        tuple(str(value) for value in raw_arches if isinstance(value, str))
        if isinstance(raw_arches, list)
        else ()
    )
    build = f"PyTorch {torch_version} / CUDA {cuda_version}"

    if status == "ready":
        return (
            True,
            (
                f"{gpu_name} ({architecture}) ran a real CUDA test kernel with "
                f"{build} in the selected trainer environment."
            ),
        )
    if status == "cuda_missing":
        return (
            False,
            (
                f"{build} loaded, but it cannot see a CUDA-capable NVIDIA GPU. "
                "Install the NVIDIA driver and a CUDA-enabled PyTorch build in this "
                "ai-toolkit environment, then retry."
            ),
        )

    error = str(payload.get("error") or "the CUDA test kernel could not start")
    supported = ", ".join(arches) if arches else "no architecture list"
    return (
        False,
        (
            f"{gpu_name} uses {architecture}, but {build} could not run a CUDA test "
            f"kernel. This PyTorch wheel reports {supported}. {error}. Install a "
            f"PyTorch CUDA build compatible with {architecture} in the selected "
            "ai-toolkit environment, then retry."
        ),
    )


def probe_trainer_runtime(executable: Path) -> tuple[bool, str]:
    """Exercise one real CUDA kernel in the exact Python used by ai-toolkit."""

    script = "\n".join(
        (
            "import json",
            "import torch",
            "payload = {",
            "    'torch_version': str(torch.__version__),",
            "    'cuda_version': str(torch.version.cuda or ''),",
            "    'arch_list': list(getattr(torch.cuda, 'get_arch_list', lambda: [])()),",
            "}",
            "if not torch.cuda.is_available():",
            "    payload['status'] = 'cuda_missing'",
            "else:",
            "    try:",
            "        payload['gpu_name'] = torch.cuda.get_device_name(0)",
            "        payload['capability'] = list(torch.cuda.get_device_capability(0))",
            "        probe = torch.arange(32, dtype=torch.float32, device='cuda')",
            "        probe = (probe + 1.0).square().sum()",
            "        torch.cuda.synchronize()",
            "        payload['probe_value'] = float(probe.cpu().item())",
            "        payload['status'] = 'ready'",
            "    except BaseException as exc:",
            "        payload['status'] = 'kernel_failed'",
            "        payload['error'] = f'{type(exc).__name__}: {exc}'",
            f"print('{_TRAINER_PROBE_PREFIX}' + json.dumps(payload, sort_keys=True))",
        )
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
    return _trainer_probe_result(result)


def _find_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _project_model_files(folder: Path | None, project_name: str) -> tuple[Path, ...]:
    """Find only ai-toolkit-style files owned by one exact project name."""

    if folder is None or not folder.is_dir():
        return ()
    filename = re.compile(
        rf"^{re.escape(project_name)}(?:_\d+)?\.safetensors$",
        re.IGNORECASE,
    )
    try:
        return tuple(
            sorted(
                path
                for path in folder.rglob("*.safetensors")
                if path.is_file() and filename.fullmatch(path.name)
            )
        )
    except OSError:
        # The dedicated storage/LoRA checks provide the actionable permission error.
        return ()


def caption_contains_trigger(caption: str, trigger: str) -> bool:
    """Return whether a caption contains the required phrase as complete words."""

    normalized_caption = " ".join(caption.casefold().split())
    normalized_trigger = " ".join(trigger.casefold().split())
    if not normalized_caption or not normalized_trigger:
        return False

    start = 0
    while (position := normalized_caption.find(normalized_trigger, start)) >= 0:
        before = normalized_caption[position - 1] if position else ""
        end = position + len(normalized_trigger)
        after = normalized_caption[end] if end < len(normalized_caption) else ""
        before_is_word = before.isalnum() or before == "_"
        after_is_word = after.isalnum() or after == "_"
        if not before_is_word and not after_is_word:
            return True
        start = position + 1
    return False


def _photo_signature(image: Image.Image) -> tuple[int, bytes]:
    """Return small local-only signatures used for conservative similarity checks."""

    grayscale = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS)
    values = tuple(grayscale.tobytes())
    difference_hash = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            difference_hash <<= 1
            difference_hash |= values[offset + column] > values[offset + column + 1]
    sample = image.convert("RGB").resize((16, 16), Image.Resampling.LANCZOS)
    return difference_hash, sample.tobytes()


def _looks_blurry(image: Image.Image) -> bool:
    """Flag very low-detail photos for review without blocking preparation."""

    grayscale = ImageOps.grayscale(image)
    grayscale.thumbnail((256, 256), Image.Resampling.LANCZOS)
    if grayscale.width < 8 or grayscale.height < 8:
        return True
    edges = grayscale.filter(ImageFilter.FIND_EDGES).crop(
        (2, 2, grayscale.width - 2, grayscale.height - 2)
    )
    return ImageStat.Stat(edges).var[0] < 35.0


def _near_duplicate(
    first: tuple[int, bytes],
    second: tuple[int, bytes],
) -> bool:
    hash_distance = (first[0] ^ second[0]).bit_count()
    if hash_distance > 4:
        return False
    squared_error = sum(
        (left - right) ** 2 for left, right in zip(first[1], second[1], strict=True)
    )
    rms = math.sqrt(squared_error / len(first[1]))
    return rms <= 8.0


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
    issues: list[SourceIssue] = []
    digests: dict[str, Path] = {}
    signatures: list[tuple[Path, tuple[int, bytes]]] = []
    for path in images:
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.load()
                width, height = image.size
            if max(width, height) < target_size:
                small += 1
                issues.append(
                    SourceIssue(
                        "small",
                        path,
                        f"Only {width}×{height}; it will be enlarged to "
                        f"{target_size} px.",
                    )
                )
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            fingerprint = digest.hexdigest()
            original = digests.get(fingerprint)
            if original is not None:
                duplicate += 1
                issues.append(
                    SourceIssue(
                        "duplicate",
                        path,
                        f"Exact copy of {original.relative_to(folder).as_posix()}.",
                        original,
                    )
                )
            else:
                digests[fingerprint] = path
                signature = _photo_signature(image)
                similar = next(
                    (
                        previous
                        for previous, previous_signature in signatures
                        if _near_duplicate(signature, previous_signature)
                    ),
                    None,
                )
                if similar is not None:
                    issues.append(
                        SourceIssue(
                            "near_duplicate",
                            path,
                            "Looks very similar to "
                            f"{similar.relative_to(folder).as_posix()}.",
                            similar,
                        )
                    )
                signatures.append((path, signature))
            if _looks_blurry(image):
                issues.append(
                    SourceIssue(
                        "blurry",
                        path,
                        "Very little sharp detail was detected; review at full size.",
                    )
                )
        except (OSError, UnidentifiedImageError, ValueError):
            unreadable += 1
            issues.append(
                SourceIssue(
                    "unreadable",
                    path,
                    "The file could not be opened as an image and will be skipped.",
                )
            )
    return SourceSummary(
        folder,
        len(images),
        unreadable,
        small,
        duplicate,
        tuple(issues),
    )


def inspect_prepared_dataset(
    folder: Path,
    draft: ProjectDraft,
) -> PreparedDatasetSummary:
    """Verify generated image/caption pairs before reusing them for training."""

    inventory = scan_dataset_pairs(folder)
    empty_captions: list[Path] = []
    missing_trigger_captions: list[Path] = []
    unreadable_images: list[Path] = []
    for image_path, caption_path in inventory.pairs:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError, ValueError):
            unreadable_images.append(image_path)
        try:
            caption = caption_path.read_text(encoding="utf-8").strip()
            if not caption:
                empty_captions.append(caption_path)
            elif not caption_contains_trigger(caption, draft.trigger):
                missing_trigger_captions.append(caption_path)
        except (OSError, UnicodeError):
            empty_captions.append(caption_path)

    manifest_error: str | None = None
    manifest_path = folder / ".source-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        settings = manifest["settings"]
        sources = manifest["sources"]
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != 2
            or not isinstance(settings, dict)
            or not isinstance(sources, list)
        ):
            raise ValueError("unsupported manifest format")
        if settings.get("trigger") != draft.trigger.strip():
            manifest_error = "The prepared captions use a different identity phrase."
        elif settings.get("size") != draft.preset.image_size:
            manifest_error = (
                "The prepared photo size does not match this training plan."
            )
        elif draft.source_dir is not None and draft.source_dir.is_dir():
            for entry in sources:
                if not isinstance(entry, dict) or not isinstance(
                    entry.get("path"), str
                ):
                    raise TypeError("invalid source entry")
                relative = Path(entry["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("unsafe source entry")
                source = draft.source_dir / relative
                if not source.is_file():
                    manifest_error = (
                        "The source photo set changed after preparation; rebuild it."
                    )
                    break
                stat = source.stat()
                if stat.st_size != entry.get("size") or stat.st_mtime_ns != entry.get(
                    "mtime_ns"
                ):
                    manifest_error = (
                        "The source photo set changed after preparation; rebuild it."
                    )
                    break
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        manifest_error = f"The preparation record is missing or damaged: {exc}"

    return PreparedDatasetSummary(
        folder=folder,
        inventory=inventory,
        empty_caption_files=tuple(empty_captions),
        missing_trigger_caption_files=tuple(missing_trigger_captions),
        unreadable_image_files=tuple(unreadable_images),
        manifest_error=manifest_error,
    )


def _source_readiness_item(source: SourceSummary) -> ReadinessItem:
    if source.usable_count == 0:
        return ReadinessItem(
            "photos",
            "Training photos",
            "error",
            "No readable JPG, PNG, WebP, BMP, or TIFF images were found.",
            "Choose another folder",
        )
    if source.duplicate_count:
        return ReadinessItem(
            "photos",
            "Training photos",
            "warning",
            f"{source.unique_usable_count} unique usable photo(s); "
            f"{source.duplicate_count} exact copy or copies will be ignored "
            "automatically. Original files stay untouched.",
            "Review photo checks",
        )
    if source.unreadable_count:
        return ReadinessItem(
            "photos",
            "Training photos",
            "warning",
            f"{source.usable_count} usable; "
            f"{source.unreadable_count} unreadable file(s) will be skipped.",
            "Review photo checks",
        )
    if source.usable_count < 10:
        return ReadinessItem(
            "photos",
            "Training photos",
            "warning",
            f"{source.usable_count} usable photo(s). "
            "15–40 varied photos usually gives a better identity.",
            "Review photo checks" if source.issues else "Add more varied photos",
        )
    if source.usable_count > 100:
        return ReadinessItem(
            "photos",
            "Training photos",
            "error",
            f"{source.usable_count} photos were found recursively. This looks "
            "like a broad photo library rather than one focused identity dataset.",
            "Choose a folder with 15–40 varied photos",
        )
    if source.usable_count > 60:
        return ReadinessItem(
            "photos",
            "Training photos",
            "warning",
            f"{source.usable_count} photos were found. A focused set of 15–40 "
            "varied images is usually easier to train well.",
            "Review photo checks",
        )
    if source.blurry_count or source.near_duplicate_count:
        concerns: list[str] = []
        if source.blurry_count:
            concerns.append(f"{source.blurry_count} may be blurry")
        if source.near_duplicate_count:
            concerns.append(f"{source.near_duplicate_count} look very similar")
        return ReadinessItem(
            "photos",
            "Training photos",
            "warning",
            f"{source.usable_count} usable photo(s); " + " and ".join(concerns) + ".",
            "Review photo checks",
        )
    note = f"{source.usable_count} usable photo(s)" + (
        f"; {source.small_count} will be enlarged." if source.small_count else "."
    )
    return ReadinessItem(
        "photos",
        "Training photos",
        "ready",
        note,
        "Review photo checks" if source.issues else "",
    )


def _caption_readiness_item(draft: ProjectDraft) -> ReadinessItem:
    if not draft.use_vision_captions:
        return ReadinessItem(
            "captions",
            "Photo descriptions",
            "ready",
            "Using the built-in local caption template.",
        )
    if _module_available("torch") and _module_available("transformers"):
        return ReadinessItem(
            "captions",
            "Automatic photo descriptions",
            "ready",
            "The local vision-captioning components are installed.",
        )
    return ReadinessItem(
        "captions",
        "Automatic photo descriptions",
        "error",
        "The optional PyTorch and Transformers components are not installed.",
        "Turn off automatic descriptions",
    )


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


def verified_missing_lora_directory(
    settings: AppSettings,
    discovery: ComfyDesktopDiscovery | None,
) -> Path | None:
    """Return only an exact, selected Desktop LoRA directory safe to create."""

    target = settings.comfy_lora_dir
    if discovery is None or target is None or target.exists():
        return None
    selected = discovery.selected_installation
    if (
        selected is None
        or not selected.is_local
        or selected.id != settings.comfy_installation_id
    ):
        return None
    target_resolved = target.resolve()
    for root in discovery.model_roots:
        candidate = root.loras
        if (
            candidate.resolve() == target_resolved
            and root.path.is_dir()
            and candidate.parent.is_dir()
        ):
            return candidate
    return None


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

    def outcome_path(self, settings: AppSettings, project_name: str) -> Path:
        config.validate_project_name(project_name)
        return (
            self.project_path(settings.run_configs_dir) / f"{project_name}-result.json"
        )

    def read_pipeline_outcome(
        self,
        settings: AppSettings,
        project_name: str,
    ) -> PipelineOutcome | None:
        path = self.outcome_path(settings, project_name)
        try:
            if path.stat().st_size > 64 * 1024:
                return None
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            return None
        mode = value.get("mode")
        fields = {
            key: value.get(key)
            for key in (
                "project_name",
                "result_path",
                "trigger",
                "preset",
                "checkpoint",
                "lora_name",
                "completed_at",
            )
        }
        if (
            mode not in {"prepare", "full", "validate"}
            or fields["project_name"] != project_name
            or not all(isinstance(field, str) for field in fields.values())
        ):
            return None
        return PipelineOutcome(
            project_name=project_name,
            mode=mode,
            result_path=Path(cast(str, fields["result_path"])),
            trigger=cast(str, fields["trigger"]),
            preset=cast(str, fields["preset"]),
            checkpoint=cast(str, fields["checkpoint"]),
            lora_name=cast(str, fields["lora_name"]),
            completed_at=cast(str, fields["completed_at"]),
        )

    def recovery_path(self, settings: AppSettings, project_name: str) -> Path:
        """Return the app-owned validation recovery record for one project."""

        config.validate_project_name(project_name)
        return (
            self.project_path(settings.run_configs_dir)
            / f"{project_name}-recovery.json"
        )

    def read_validation_recovery(
        self,
        settings: AppSettings,
        project_name: str,
    ) -> ValidationRecovery | None:
        """Read a safe, locally verified post-install validation recovery record."""

        path = self.recovery_path(settings, project_name)
        try:
            if path.stat().st_size > 64 * 1024:
                return None
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            return None
        string_fields = {
            key: value.get(key)
            for key in (
                "project_name",
                "state",
                "lora_name",
                "installed_lora_path",
                "expected_grid_path",
                "trigger",
                "preset",
                "checkpoint",
                "updated_at",
            )
        }
        if (
            string_fields["project_name"] != project_name
            or not all(
                isinstance(field, str) and field for field in string_fields.values()
            )
            or string_fields["state"] not in {"validation_pending", "complete"}
        ):
            return None
        lora_name = cast(str, string_fields["lora_name"])
        if (
            Path(lora_name).name != lora_name
            or Path(lora_name).suffix.casefold() != ".safetensors"
        ):
            return None
        try:
            preset = TrainingPreset(cast(str, string_fields["preset"]))
        except ValueError:
            return None
        installed_lora = Path(cast(str, string_fields["installed_lora_path"]))
        expected_grid = Path(cast(str, string_fields["expected_grid_path"]))
        if not installed_lora.is_absolute() or not expected_grid.is_absolute():
            return None
        configured_grid = (
            self.project_path(settings.grids_dir) / f"{project_name}_grid.png"
        ).resolve()
        if expected_grid.resolve() != configured_grid:
            return None
        configured_lora_dir = settings.comfy_lora_dir
        if configured_lora_dir is None:
            return None
        configured_lora = (self.project_path(configured_lora_dir) / lora_name).resolve()
        if installed_lora.resolve() != configured_lora or not installed_lora.is_file():
            return None
        return ValidationRecovery(
            project_name=project_name,
            state=cast(
                Literal["validation_pending", "complete"],
                string_fields["state"],
            ),
            lora_name=lora_name,
            installed_lora_path=installed_lora,
            expected_grid_path=expected_grid,
            trigger=cast(str, string_fields["trigger"]),
            preset=preset,
            checkpoint=cast(str, string_fields["checkpoint"]),
            updated_at=cast(str, string_fields["updated_at"]),
        )

    def resolve_checkpoint_path(
        self,
        discovery: ComfyDesktopDiscovery | None,
        checkpoint: str,
    ) -> Path | None:
        """Resolve an unambiguous API checkpoint inside configured model roots."""

        candidates = self.resolve_checkpoint_paths(discovery, checkpoint)
        return candidates[0] if len(candidates) == 1 else None

    def resolve_checkpoint_paths(
        self,
        discovery: ComfyDesktopDiscovery | None,
        checkpoint: str,
    ) -> tuple[Path, ...]:
        """Return every safe local match in ComfyUI's effective search order."""

        if discovery is None or not checkpoint.strip():
            return ()
        relative = Path(checkpoint.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            return ()
        matches: list[Path] = []
        for model_root in discovery.model_roots:
            root = model_root.checkpoints.resolve()
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            if candidate.is_file():
                matches.append(candidate)
        return tuple(dict.fromkeys(matches))

    def apply_discovery(
        self,
        settings: AppSettings,
        discovery: ComfyDesktopDiscovery | None = None,
    ) -> AppSettings:
        """Fill only missing/default setup values from read-only discovery."""

        applied, _discovery, _status = self.setup_snapshot(settings, discovery)
        return applied

    def setup_snapshot(
        self,
        settings: AppSettings,
        discovery: ComfyDesktopDiscovery | None = None,
    ) -> tuple[AppSettings, ComfyDesktopDiscovery, ComfyApiStatus]:
        """Bind one Desktop instance, its paths, and the live API atomically."""

        found = discovery or self.discover_comfy()
        local_installations = tuple(
            installation
            for installation in found.installations
            if installation.is_local
        )
        selected_id = settings.comfy_installation_id
        if not any(item.id == selected_id for item in local_installations):
            if found.selected_installation is not None:
                selected_id = found.selected_installation.id
            elif len(local_installations) == 1:
                selected_id = local_installations[0].id
            else:
                selected_id = ""
        if selected_id:
            found = select_comfy_installation(found, selected_id)

        host_candidates = tuple(found.host_candidates)
        requested_host = _normalise_host(settings.comfy_host)
        if found.selected_installation is not None:
            # Identity-bearing port locks are the only safe way to associate a
            # live API with a selected Desktop installation.  Never continue
            # probing generic ports that may belong to another installation.
            ordered_hosts = host_candidates
        else:
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
        effective_lora_dirs = tuple(root.loras for root in found.model_roots)
        if lora_dir is None or (
            found.selected_installation is not None
            and not any(
                lora_dir.resolve() == candidate.resolve()
                for candidate in effective_lora_dirs
            )
        ):
            lora_dir = None
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
        else:
            local_names: list[str] = []
            for model_root in found.model_roots:
                if model_root.checkpoints.is_dir():
                    local_names.extend(
                        str(path.relative_to(model_root.checkpoints)).replace("\\", "/")
                        for path in model_root.checkpoints.rglob("*")
                        if path.is_file()
                        and path.suffix.lower() in {".safetensors", ".ckpt", ".pt"}
                    )
            unique_local_names = tuple(sorted(set(local_names), key=str.casefold))
            if unique_local_names and (
                checkpoint not in unique_local_names
                or not self.resolve_checkpoint_paths(found, checkpoint)
            ):
                checkpoint = next(
                    (
                        candidate
                        for candidate in unique_local_names
                        if "xl" in candidate.casefold()
                    ),
                    unique_local_names[0],
                )

        checkpoint_paths = self.resolve_checkpoint_paths(found, checkpoint)
        previous_base = settings.training_base_model
        if previous_base is not None and any(
            previous_base.resolve() == candidate.resolve()
            for candidate in checkpoint_paths
        ):
            training_base_model = previous_base
        elif len(checkpoint_paths) == 1:
            training_base_model = checkpoint_paths[0]
        else:
            training_base_model = None

        toolkit_dir = settings.ai_toolkit_dir
        toolkit_python = settings.ai_toolkit_python
        if toolkit_dir is None:
            toolkit_dir, toolkit_python = discover_ai_toolkit(self.root)
        elif (toolkit_dir / "run.py").is_file():
            if toolkit_python is None or not toolkit_python.is_file():
                toolkit_python = _trainer_python(toolkit_dir)
        else:
            toolkit_python = None

        return (
            replace(
                settings,
                comfy_host=host or "127.0.0.1:8188",
                comfy_installation_id=selected_id,
                comfy_lora_dir=lora_dir,
                comfy_checkpoint=checkpoint,
                training_base_model=training_base_model,
                ai_toolkit_dir=toolkit_dir,
                ai_toolkit_python=toolkit_python,
            ),
            found,
            live,
        )

    def project_artifacts(
        self,
        settings: AppSettings,
        project_name: str,
    ) -> ProjectArtifactSummary:
        """Return every output namespace that a new run could otherwise reuse."""

        project_name = config.validate_project_name(project_name)
        dataset_path = self.project_path(settings.datasets_dir) / project_name
        checkpoints_dir = self.project_path(settings.checkpoints_dir)
        grids_dir = self.project_path(settings.grids_dir)
        configs_dir = self.project_path(settings.run_configs_dir)
        validation_grid = grids_dir / f"{project_name}_grid.png"
        validation_work_dir = grids_dir / project_name
        run_config = configs_dir / f"{project_name}.yaml"
        result_record = configs_dir / f"{project_name}-result.json"
        return ProjectArtifactSummary(
            dataset=dataset_path if dataset_path.exists() else None,
            checkpoints=_project_model_files(checkpoints_dir, project_name),
            validation_grid=validation_grid if validation_grid.exists() else None,
            validation_work_dir=(
                validation_work_dir if validation_work_dir.exists() else None
            ),
            run_config=run_config if run_config.exists() else None,
            result_record=result_record if result_record.exists() else None,
            installed_loras=_project_model_files(
                (
                    self.project_path(settings.comfy_lora_dir)
                    if settings.comfy_lora_dir is not None
                    else None
                ),
                project_name,
            ),
        )

    def readiness(
        self,
        settings: AppSettings,
        draft: ProjectDraft,
        discovery: ComfyDesktopDiscovery | None = None,
    ) -> ReadinessReport:
        items: list[ReadinessItem] = []
        source: SourceSummary | None = None
        prepared: PreparedDatasetSummary | None = None
        reuse_prepared = False
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
            artifacts = self.project_artifacts(settings, draft.project_name)
            existing_dataset = artifacts.dataset
            artifacts_for_collision = artifacts
            invalid_prepared_reason = ""
            if existing_dataset is not None and not draft.overwrite_existing:
                prepared = inspect_prepared_dataset(existing_dataset, draft)
                reuse_prepared = prepared.ready
                if not reuse_prepared:
                    invalid_prepared_reason = prepared.problem_detail

                # A prepare-only worker writes this record after atomically completing
                # the dataset. It is resume metadata, not a prior training output.
                outcome = self.read_pipeline_outcome(settings, draft.project_name)
                if (
                    reuse_prepared
                    and outcome is not None
                    and outcome.mode == "prepare"
                    and outcome.result_path.resolve() == existing_dataset.resolve()
                ):
                    artifacts_for_collision = replace(
                        artifacts_for_collision,
                        result_record=None,
                    )

            output_description = artifacts_for_collision.describe_non_dataset_outputs()
            if draft.overwrite_existing and (
                existing_dataset is not None
                or artifacts_for_collision.has_non_dataset_outputs
            ):
                existing_descriptions = []
                if existing_dataset is not None:
                    existing_descriptions.append("prepared photos")
                if output_description:
                    existing_descriptions.append(output_description)
                items.append(
                    ReadinessItem(
                        "collision",
                        "Existing project data",
                        "warning",
                        "Rebuild is explicitly enabled. This run may replace "
                        + ", ".join(existing_descriptions)
                        + ".",
                        "Turn off rebuild to preserve existing work",
                    )
                )
            elif artifacts_for_collision.has_non_dataset_outputs:
                prepared_detail = (
                    f"{prepared.image_count} prepared photo/caption pairs can be "
                    "reused, but "
                    if reuse_prepared and prepared is not None
                    else ""
                )
                items.append(
                    ReadinessItem(
                        "collision",
                        "Existing project outputs",
                        "error",
                        f"{prepared_detail}{output_description} already use this "
                        "project name. A new run is blocked so files cannot be mixed "
                        "or replaced silently.",
                        "Use a new version name or explicitly allow rebuilding",
                    )
                )
            elif (
                existing_dataset is not None and reuse_prepared and prepared is not None
            ):
                items.append(
                    ReadinessItem(
                        "collision",
                        "Prepared project",
                        "ready",
                        f"{prepared.image_count} prepared photo/caption pairs "
                        "match this project and will be reused.",
                        "Review prepared captions",
                    )
                )
            elif existing_dataset is not None and not draft.overwrite_existing:
                repair_action = (
                    "Review prepared files"
                    if prepared is not None and prepared.file_problem_messages
                    else "Allow rebuilding or use a new version name"
                )
                items.append(
                    ReadinessItem(
                        "collision",
                        "Existing project data",
                        "error",
                        "The existing prepared data cannot be reused. "
                        f"{invalid_prepared_reason}",
                        repair_action,
                    )
                )
            elif existing_dataset is not None:
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

        if reuse_prepared and prepared is not None:
            items.append(
                ReadinessItem(
                    "photos",
                    "Prepared training photos",
                    "ready",
                    f"{prepared.image_count} prepared photo(s) will be reused; "
                    "the source folder will not be processed again.",
                    "Review prepared captions",
                )
            )
            items.append(
                ReadinessItem(
                    "captions",
                    "Prepared photo descriptions",
                    "ready",
                    f"All {prepared.caption_count} prepared photo(s) have editable "
                    "captions.",
                    "Review prepared captions",
                )
            )
        else:
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
                items.append(_source_readiness_item(source))
            items.append(_caption_readiness_item(draft))

        host = _normalise_host(settings.comfy_host)
        selected_for_probe = (
            next(
                (
                    installation
                    for installation in discovery.installations
                    if installation.is_local
                    and installation.id == settings.comfy_installation_id
                ),
                None,
            )
            if discovery is not None
            else None
        )
        matching_hosts = (
            tuple(
                hint.host
                for hint in discovery.runtime_hints
                if selected_for_probe is not None
                and hint.installation_name.casefold()
                == selected_for_probe.name.casefold()
            )
            if discovery is not None
            else ()
        )
        probe_hosts = (
            (host,) if selected_for_probe is None or host in matching_hosts else ()
        )
        comfy = probe_comfy_api(probe_hosts, timeout=0.75)
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

        local_installations = (
            tuple(
                installation
                for installation in discovery.installations
                if installation.is_local
            )
            if discovery is not None
            else ()
        )
        selected_installation = next(
            (
                installation
                for installation in local_installations
                if installation.id == settings.comfy_installation_id
            ),
            None,
        )
        if not local_installations:
            items.append(
                ReadinessItem(
                    "connection",
                    "Desktop instance binding",
                    "ready",
                    "The local ComfyUI connection is unambiguous.",
                )
            )
        else:
            runtime_hints = discovery.runtime_hints if discovery is not None else ()
            matching_runtime = next(
                (hint for hint in runtime_hints if hint.host == comfy.host),
                None,
            )
            if selected_installation is None:
                items.append(
                    ReadinessItem(
                        "connection",
                        "Desktop instance binding",
                        "error",
                        "More than one local Desktop instance exists and none is selected.",
                        "Choose a Desktop instance",
                    )
                )
            elif matching_runtime is None:
                items.append(
                    ReadinessItem(
                        "connection",
                        "Desktop instance binding",
                        "error",
                        "The running API could not be matched to the selected Desktop "
                        "instance. This run is blocked to protect the wrong model library.",
                        "Launch the selected instance and check again",
                    )
                )
            elif (
                matching_runtime.installation_name.casefold()
                != selected_installation.name.casefold()
            ):
                items.append(
                    ReadinessItem(
                        "connection",
                        "Desktop instance binding",
                        "error",
                        f"The API at {comfy.host} belongs to "
                        f"“{matching_runtime.installation_name}”, not the selected "
                        f"“{selected_installation.name}”.",
                        "Choose the running Desktop instance",
                    )
                )
            else:
                items.append(
                    ReadinessItem(
                        "connection",
                        "Desktop instance binding",
                        "ready",
                        f"{selected_installation.name}, {comfy.host}, and its model "
                        "folders are bound together.",
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
        elif (
            settings.confirmed_sdxl_checkpoint.strip().casefold()
            != selected_checkpoint.casefold()
        ):
            items.append(
                ReadinessItem(
                    "checkpoint",
                    "SDXL base model",
                    "error",
                    f"ComfyUI reports “{selected_checkpoint}”, but does not expose "
                    "enough metadata to prove that it is an SDXL base model. "
                    "Confirm it before spending time on training.",
                    "Confirm this is an SDXL base checkpoint",
                )
            )
        else:
            items.append(
                ReadinessItem(
                    "checkpoint",
                    "SDXL base model",
                    "ready",
                    f"Confirmed SDXL base model: {selected_checkpoint}",
                )
            )

        base_model = settings.training_base_model
        checkpoint_paths = self.resolve_checkpoint_paths(
            discovery,
            selected_checkpoint,
        )
        if base_model is None:
            ambiguity = (
                f" {len(checkpoint_paths)} matching files were found; choose the "
                "one used by this Desktop instance."
                if len(checkpoint_paths) > 1
                else ""
            )
            items.append(
                ReadinessItem(
                    "base_model",
                    "Training base model",
                    "error",
                    "The selected ComfyUI checkpoint file could not be located for "
                    f"the trainer.{ambiguity}",
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
        elif not checkpoint_paths:
            items.append(
                ReadinessItem(
                    "base_model",
                    "Training base model",
                    "error",
                    "The selected training file could not be resolved inside the "
                    "selected Desktop instance's effective model folders.",
                    "Choose the checkpoint from the selected Desktop instance",
                )
            )
        elif not any(
            base_model.resolve() == candidate.resolve()
            for candidate in checkpoint_paths
        ):
            items.append(
                ReadinessItem(
                    "base_model",
                    "Training base model",
                    "error",
                    "The selected training file is not the checkpoint selected in "
                    "ComfyUI. Training and validation must use the same base model.",
                    "Choose the matching local SDXL checkpoint",
                )
            )
        else:
            items.append(
                ReadinessItem(
                    "base_model",
                    "Training base model",
                    "ready",
                    f"Training and validation are bound to: {base_model}",
                )
            )

        lora_dir = settings.comfy_lora_dir
        effective_lora_dirs = (
            tuple(root.loras for root in discovery.model_roots)
            if discovery is not None and selected_installation is not None
            else ()
        )
        creatable_lora_dir = verified_missing_lora_directory(settings, discovery)
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
        elif selected_installation is not None and (
            not effective_lora_dirs
            or not any(
                lora_dir.resolve() == candidate.resolve()
                for candidate in effective_lora_dirs
            )
        ):
            items.append(
                ReadinessItem(
                    "lora_dir",
                    "ComfyUI LoRA library",
                    "error",
                    "The selected LoRA folder is not part of the selected Desktop "
                    "instance's effective model folders.",
                    "Use the selected Desktop instance's LoRA folder",
                )
            )
        elif not lora_dir.is_dir():
            items.append(
                ReadinessItem(
                    "lora_dir",
                    "ComfyUI LoRA library",
                    "error",
                    f"The folder does not exist: {lora_dir}",
                    (
                        "Create LoRA folder"
                        if creatable_lora_dir is not None
                        else "Choose another folder"
                    ),
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
            trainer_ready, trainer_detail = probe_trainer_runtime(toolkit_python)
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
            "connection",
            "checkpoint",
            "lora_dir",
            "trainer",
            "training_storage",
        }
        full_codes.add("base_model")
        can_prepare = (
            all(
                by_code.get(code) is not None and by_code[code].state != "error"
                for code in prepare_codes
            )
            and not reuse_prepared
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
            prepared=prepared,
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
        artifacts = self.project_artifacts(settings, draft.project_name)
        dataset = self.project_path(settings.datasets_dir) / draft.project_name
        prepared = (
            inspect_prepared_dataset(dataset, draft)
            if dataset.exists() and not draft.overwrite_existing
            else None
        )
        collision_artifacts = artifacts
        outcome = self.read_pipeline_outcome(settings, draft.project_name)
        if (
            prepared is not None
            and prepared.ready
            and outcome is not None
            and outcome.mode == "prepare"
            and outcome.result_path.resolve() == dataset.resolve()
        ):
            collision_artifacts = replace(collision_artifacts, result_record=None)
        if collision_artifacts.has_non_dataset_outputs and not draft.overwrite_existing:
            raise ValueError(
                "Existing project outputs were found: "
                f"{collision_artifacts.describe_non_dataset_outputs()}. Use a new "
                "version name or explicitly allow rebuilding before starting."
            )
        if (
            dataset.exists()
            and prepared is not None
            and not prepared.ready
            and not draft.overwrite_existing
        ):
            raise ValueError(
                "The existing prepared project is incomplete or no longer matches "
                f"these settings. {prepared.problem_detail} Explicitly allow "
                "rebuilding or use a new version name."
            )
        reuse_prepared = mode == "full" and prepared is not None and prepared.ready
        if not reuse_prepared and (
            draft.source_dir is None or not draft.source_dir.is_dir()
        ):
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
        if reuse_prepared:
            args.append("--skip-prep")
        else:
            args.extend(("--src", str(draft.source_dir)))
        if draft.use_vision_captions:
            args.append("--vision")
        if draft.overwrite_existing:
            args.append("--overwrite")
        if settings.training_base_model is not None:
            args.extend(("--base-model", str(settings.training_base_model)))
        if mode == "prepare":
            if prepared is not None and prepared.ready and not draft.overwrite_existing:
                raise ValueError(
                    "These photos are already prepared. Review them or start the full "
                    "LoRA run instead."
                )
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
        *,
        checkpoint: str | None = None,
    ) -> PipelineCommand:
        """Build a validation-only retry after training already installed a LoRA."""

        config.validate_project_name(draft.project_name)
        cleaned_lora = Path(lora_name).name
        if (
            cleaned_lora != lora_name
            or Path(cleaned_lora).suffix.casefold() != ".safetensors"
        ):
            raise ValueError(f"Invalid installed LoRA filename: {lora_name}")
        checkpoint_name = (
            checkpoint if checkpoint is not None else settings.comfy_checkpoint
        ).strip()
        checkpoint_path = Path(checkpoint_name.replace("\\", "/"))
        if (
            not checkpoint_name
            or checkpoint_path.is_absolute()
            or checkpoint_path.name in {"", ".", ".."}
            or ".." in checkpoint_path.parts
        ):
            raise ValueError(f"Invalid ComfyUI checkpoint name: {checkpoint_name!r}")
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
        # A durable retry must use the checkpoint saved when the LoRA was
        # installed, even if the current setup selection changed afterward.
        environment["COMFY_SDXL_CKPT"] = checkpoint_name
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
