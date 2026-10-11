"""Read-only integration helpers for ComfyUI Desktop on Windows.

The desktop application has used two configuration layouts:

* Comfy Desktop 2 stores ``settings.json``, ``installations.json`` and
  ``last-session.json`` below ``%APPDATA%/Comfy Desktop``.
* The legacy desktop application stores ``config.json`` below
  ``%APPDATA%/ComfyUI`` and records its data directory as ``basePath``.

This module intentionally treats both formats as best-effort input.  A damaged
or newer configuration file must not prevent this application from starting.
It never changes ComfyUI state and does not search drives or arbitrary ports.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

import yaml

from . import __version__

COMFY_DESKTOP_APP_ID = "com.todesktop.241012ess7yxs0e"
"""Windows Start-menu AppID used by the current Comfy Desktop release."""

_CONFIG_SIZE_LIMIT = 4 * 1024 * 1024
_DEFAULT_LOCAL_PORTS = (8188, 8000, 8001)
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_HOST_KEYS = frozenset(
    {
        "address",
        "host",
        "listen",
        "localurl",
        "remoteurl",
        "serverhost",
        "serverurl",
        "url",
    }
)
_PORT_KEYS = frozenset({"comfyport", "lastport", "listenport", "port", "serverport"})
_MODEL_FILE_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".pkl", ".pt", ".pt2", ".pth", ".safetensors", ".sft"}
)
_PORT_LOCK_NAME = re.compile(r"^port-(\d{1,5})\.json$", re.IGNORECASE)
_DESKTOP_DATA_LOCATION_FILE = "data-location.json"


@dataclass(frozen=True, slots=True)
class ComfyInstallation:
    """One installation known to Comfy Desktop 2."""

    id: str
    name: str
    source_id: str
    install_path: Path | None
    comfyui_path: Path | None
    version: str | None
    status: str | None
    last_launched_at: int | None
    use_shared_models: bool = True
    use_shared_input_output: bool = True
    model_dirs: tuple[Path, ...] = ()
    model_dirs_primary: Path | None = None
    input_dir: Path | None = None
    output_dir: Path | None = None
    launch_args: str = ""
    adopted: bool = False
    adopted_base_dir: Path | None = None

    @property
    def is_local(self) -> bool:
        """Return whether this record represents a locally installed server."""

        return self.install_path is not None and self.source_id.casefold() != "cloud"


@dataclass(frozen=True, slots=True)
class ComfyModelRoot:
    """A configured ComfyUI model root and its pipeline-relevant folders."""

    path: Path
    loras: Path
    checkpoints: Path
    source: str
    exists: bool


@dataclass(frozen=True, slots=True)
class _ConfiguredModelDirectory:
    """One category-specific model directory from an ordered ComfyUI config."""

    path: Path
    source: str
    is_default: bool = False


@dataclass(frozen=True, slots=True)
class _ExtraModelLayout:
    """Parsed category paths and compatibility roots from one extra-path file."""

    roots: tuple[ComfyModelRoot, ...]
    checkpoints: tuple[_ConfiguredModelDirectory, ...]
    loras: tuple[_ConfiguredModelDirectory, ...]


@dataclass(frozen=True, slots=True)
class _EffectiveModelLayout:
    """Compatibility roots plus ComfyUI-ordered category directories."""

    roots: tuple[ComfyModelRoot, ...]
    checkpoint_dirs: tuple[Path, ...]
    lora_dirs: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _LaunchPathConfiguration:
    """Path-affecting arguments Comfy Desktop will pass to one installation."""

    models_dir: Path | None
    input_dir: Path | None
    output_dir: Path | None
    extra_model_configs: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ComfyLauncher:
    """Metadata a Windows UI can use to launch the Start-menu application."""

    display_name: str
    app_id: str
    shell_target: str
    command: tuple[str, ...]
    supported: bool


@dataclass(frozen=True, slots=True)
class ComfyRuntimeHint:
    """Identity-bearing runtime hint written by current Comfy Desktop."""

    host: str
    installation_name: str
    pid: int
    timestamp: int


@dataclass(frozen=True, slots=True)
class ComfyDesktopDiscovery:
    """Immutable snapshot of locally configured Comfy Desktop state."""

    config_dir: Path | None
    legacy_config_dir: Path | None
    installations: tuple[ComfyInstallation, ...]
    selected_installation: ComfyInstallation | None
    model_roots: tuple[ComfyModelRoot, ...]
    input_dir: Path | None
    output_dir: Path | None
    host_candidates: tuple[str, ...]
    launcher: ComfyLauncher
    warnings: tuple[str, ...]
    runtime_hints: tuple[ComfyRuntimeHint, ...] = ()
    shared_model_roots: tuple[ComfyModelRoot, ...] = ()
    legacy_model_roots: tuple[ComfyModelRoot, ...] = ()
    checkpoint_dirs: tuple[Path, ...] = ()
    lora_dirs: tuple[Path, ...] = ()
    shared_input_dir: Path | None = None
    shared_output_dir: Path | None = None
    desktop_executable: Path | None = None

    @property
    def detected(self) -> bool:
        """Return whether either desktop configuration layout was found."""

        return (
            self.config_dir is not None
            or self.legacy_config_dir is not None
            or self.desktop_executable is not None
        )

    @property
    def preferred_model_root(self) -> ComfyModelRoot | None:
        """Return the first configured model root, preferring one that exists."""

        return next(
            (root for root in self.model_roots if root.exists),
            self.model_roots[0] if self.model_roots else None,
        )

    @property
    def effective_checkpoint_dirs(self) -> tuple[Path, ...]:
        """Return checkpoint search paths in the order ComfyUI resolves names."""

        return self.checkpoint_dirs or _deduplicate_paths(
            root.checkpoints for root in self.model_roots
        )

    @property
    def effective_lora_dirs(self) -> tuple[Path, ...]:
        """Return LoRA search paths in the order ComfyUI resolves names."""

        return self.lora_dirs or _deduplicate_paths(
            root.loras for root in self.model_roots
        )

    def checkpoint_candidates(self, name: str) -> tuple[Path, ...]:
        """Resolve every local checkpoint match without losing search priority."""

        return model_file_candidates(self.effective_checkpoint_dirs, name)

    def lora_candidates(self, name: str) -> tuple[Path, ...]:
        """Resolve every local LoRA match without losing search priority."""

        return model_file_candidates(self.effective_lora_dirs, name)


@dataclass(frozen=True, slots=True)
class ComfyApiStatus:
    """Result of probing a local ComfyUI HTTP API."""

    reachable: bool
    host: str | None
    checkpoints: tuple[str, ...]
    loras: tuple[str, ...]
    error: str | None


def comfy_start_instruction(installation: ComfyInstallation | None) -> str:
    """Name the action that starts a server in Desktop's instance dashboard."""
    target = f'"{installation.name}"' if installation is not None else "selected"
    return (
        f"In Comfy Desktop, open the {target} local instance card to start ComfyUI. "
        "Opening the Desktop dashboard alone does not start that instance."
    )


def _launcher() -> ComfyLauncher:
    shell_target = rf"shell:AppsFolder\{COMFY_DESKTOP_APP_ID}"
    return ComfyLauncher(
        display_name="Comfy Desktop",
        app_id=COMFY_DESKTOP_APP_ID,
        shell_target=shell_target,
        command=("explorer.exe", shell_target),
        supported=sys.platform == "win32",
    )


def _read_json(path: Path, warnings: list[str]) -> Any | None:
    """Read a small JSON state file without allowing it to disrupt discovery."""

    if not _path_is_file(path):
        return None
    try:
        if path.stat().st_size > _CONFIG_SIZE_LIMIT:
            warnings.append(f"Ignored unusually large Comfy config: {path.name}")
            return None
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        warnings.append(f"Could not read {path.name}: {exc}")
        return None


def _string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None


def _configured_path(value: object, config_dir: Path) -> Path | None:
    raw = _string(value)
    if raw is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(raw))
    path = Path(expanded)
    # Preserve drive-letter and UNC paths when Windows state is inspected by a
    # cross-platform support tool or test runner.
    return (
        path
        if path.is_absolute() or PureWindowsPath(expanded).is_absolute()
        else config_dir / path
    )


def _desktop_launch_tokens(value: str) -> tuple[str, ...]:
    """Split startup arguments with the same quote rules as Comfy Desktop."""

    tokens: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for character in value:
        if quote is not None:
            if character == quote:
                quote = None
            else:
                current.append(character)
        elif character in {'"', "'"}:
            quote = character
        elif character.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(character)
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def _launch_argument_values(
    tokens: Sequence[str],
    option: str,
    *,
    consume_many: bool = False,
) -> tuple[str, ...]:
    """Return supplied values for one long option in command-line order."""

    values: list[str] = []
    prefix = f"{option}="
    index = 0
    while index < len(tokens):
        token = tokens[index]
        inline: str | None = None
        if token.startswith(prefix):
            inline = token[len(prefix) :]
        elif token != option:
            index += 1
            continue

        if inline:
            values.append(inline)
        cursor = index + 1
        if inline is None and not consume_many:
            if cursor < len(tokens) and not tokens[cursor].startswith("--"):
                values.append(tokens[cursor])
                cursor += 1
        elif consume_many:
            while cursor < len(tokens) and not tokens[cursor].startswith("--"):
                values.append(tokens[cursor])
                cursor += 1
        index = max(cursor, index + 1)
    return tuple(values)


def _installation_launch_cwd(installation: ComfyInstallation) -> Path | None:
    """Return the working directory used by the Desktop source launcher."""

    if installation.source_id.casefold() == "git":
        return installation.comfyui_path or installation.install_path
    return installation.install_path or installation.comfyui_path


def _launch_path_configuration(
    installation: ComfyInstallation,
    *,
    shared_input_dir: Path | None,
    shared_output_dir: Path | None,
    warnings: list[str] | None = None,
) -> _LaunchPathConfiguration:
    """Resolve the path arguments Desktop passes after source-specific defaults."""

    def warn(message: str) -> None:
        if warnings is not None and message not in warnings:
            warnings.append(message)

    launch_cwd = _installation_launch_cwd(installation)
    tokens = _desktop_launch_tokens(installation.launch_args)
    base_values = _launch_argument_values(tokens, "--base-directory")
    models_values = _launch_argument_values(tokens, "--models-directory")
    input_values = _launch_argument_values(tokens, "--input-directory")
    output_values = _launch_argument_values(tokens, "--output-directory")
    config_values = _launch_argument_values(
        tokens,
        "--extra-model-paths-config",
        consume_many=True,
    )

    base_dir = (
        installation.adopted_base_dir
        if installation.adopted and installation.adopted_base_dir is not None
        else installation.comfyui_path
    )
    if installation.adopted and installation.adopted_base_dir is None:
        warn(
            f"{installation.name} is marked as an adopted Comfy installation, "
            "but its original data folder is missing from Desktop's settings."
        )
    if base_values and launch_cwd is not None:
        base_dir = _configured_path(base_values[-1], launch_cwd) or base_dir

    models_dir = base_dir / "models" if base_dir is not None else None
    if models_values and launch_cwd is not None:
        models_dir = _configured_path(models_values[-1], launch_cwd) or models_dir

    user_input = base_dir / "input" if base_dir is not None else None
    user_output = base_dir / "output" if base_dir is not None else None
    if input_values and launch_cwd is not None:
        user_input = _configured_path(input_values[-1], launch_cwd) or user_input
    if output_values and launch_cwd is not None:
        user_output = _configured_path(output_values[-1], launch_cwd) or user_output

    if installation.use_shared_input_output:
        # Desktop appends its shared values after launchArgs, so they win. A
        # missing raw setting means the launcher's computed default is unknown
        # to this read-only integration; do not claim the user argument wins.
        input_dir = shared_input_dir
        output_dir = shared_output_dir
    else:
        input_dir = installation.input_dir or user_input
        output_dir = installation.output_dir or user_output

    extra_configs: list[Path] = []
    if launch_cwd is not None:
        for value in config_values:
            path = _configured_path(value, launch_cwd)
            if path is not None:
                extra_configs.append(path)
    elif config_values:
        warn(
            f"Could not resolve {installation.name}'s relative extra model-path "
            "configuration because its install folder is unavailable."
        )

    return _LaunchPathConfiguration(
        models_dir=models_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        extra_model_configs=_deduplicate_paths(extra_configs),
    )


def _path_exists(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _path_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _installation_from_json(
    value: object, config_dir: Path
) -> ComfyInstallation | None:
    if not isinstance(value, Mapping):
        return None
    installation_id = _string(value.get("id"))
    if installation_id is None:
        return None
    source_id = _string(value.get("sourceId")) or "unknown"
    install_path = _configured_path(value.get("installPath"), config_dir)
    if source_id.casefold() == "cloud":
        install_path = None
    comfyui_path: Path | None = None
    if install_path is not None:
        nested = install_path / "ComfyUI"
        if _path_exists(nested / "models") or _path_is_file(nested / "main.py"):
            comfyui_path = nested
        elif _path_exists(install_path / "models") or _path_is_file(
            install_path / "main.py"
        ):
            comfyui_path = install_path
        else:
            comfyui_path = nested

    raw_model_dirs = value.get("modelDirs")
    model_dirs: list[Path] = []
    if isinstance(raw_model_dirs, Sequence) and not isinstance(raw_model_dirs, str):
        for item in raw_model_dirs:
            path = _configured_path(item, config_dir)
            if path is not None:
                model_dirs.append(path)
    model_dirs_primary = _configured_path(value.get("modelDirsPrimary"), config_dir)

    version = (
        _string(value.get("comfyVersionTag"))
        or _string(value.get("version"))
        or _nested_version(value.get("comfyVersion"))
    )
    return ComfyInstallation(
        id=installation_id,
        name=_string(value.get("name")) or "ComfyUI",
        source_id=source_id,
        install_path=install_path,
        comfyui_path=comfyui_path,
        version=version,
        status=_string(value.get("status")),
        last_launched_at=_integer(value.get("lastLaunchedAt")),
        use_shared_models=value.get("useSharedModels") is not False,
        use_shared_input_output=value.get("useSharedInputOutput") is not False,
        model_dirs=tuple(model_dirs),
        model_dirs_primary=model_dirs_primary,
        input_dir=_configured_path(value.get("inputDir"), config_dir),
        output_dir=_configured_path(value.get("outputDir"), config_dir),
        launch_args=_string(value.get("launchArgs")) or "",
        adopted=value.get("adopted") is True,
        adopted_base_dir=_configured_path(value.get("adoptedBaseDir"), config_dir),
    )


def _nested_version(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    return _string(value.get("baseTag")) or _string(value.get("commit"))


def _selected_local_installation(
    installations: Sequence[ComfyInstallation],
    settings: Mapping[str, Any],
    last_session: Mapping[str, Any],
) -> ComfyInstallation | None:
    local = tuple(
        installation for installation in installations if installation.is_local
    )
    if not local:
        return None

    selected_id = (
        _string(last_session.get("installationId"))
        or _string(settings.get("selectedInstallationId"))
        or _string(settings.get("installationId"))
    )
    if selected_id is not None:
        selected = next(
            (installation for installation in local if installation.id == selected_id),
            None,
        )
        if selected is not None:
            return selected

    return max(
        local,
        key=lambda installation: (
            installation.last_launched_at
            if installation.last_launched_at is not None
            else -1
        ),
    )


def _model_paths_from_settings(
    settings: Mapping[str, Any], config_dir: Path
) -> tuple[Path, ...]:
    raw_dirs = settings.get("modelsDirs")
    if isinstance(raw_dirs, str) or not isinstance(raw_dirs, Sequence):
        raw_dirs = ()

    paths: list[Path] = []
    for value in raw_dirs:
        raw_path: object = value
        if isinstance(value, Mapping):
            if value.get("enabled") is False:
                continue
            raw_path = (
                value.get("path") or value.get("directory") or value.get("modelsDir")
            )
        path = _configured_path(raw_path, config_dir)
        if path is not None:
            paths.append(path)
    return tuple(paths)


def _windows_drive(path: Path | str) -> str:
    return PureWindowsPath(str(path)).drive.casefold()


def _display_icon_executable(value: object) -> Path | None:
    """Extract an executable path from a Windows DisplayIcon registry value."""

    raw = _string(value)
    if raw is None:
        return None
    if raw.startswith('"'):
        closing_quote = raw.find('"', 1)
        executable = raw[1:closing_quote] if closing_quote > 1 else ""
    else:
        executable = re.sub(r",\s*-?\d+\s*$", "", raw).strip()
    path = Path(executable) if executable else None
    return path if path is not None and path.suffix.casefold() == ".exe" else None


def _registered_comfy_desktop_executable(
    warnings: list[str],
) -> Path | None:
    """Return the installed Desktop executable recorded by Windows.

    Electron Builder's NSIS installer records ``DisplayIcon`` in the Windows
    uninstall registry. Reading that value lets this integration reproduce
    Desktop's redirected-drive storage choice without searching disks.
    """

    if sys.platform != "win32":
        return None
    try:
        import winreg
    except ImportError:
        return None

    locations = (
        (winreg.HKEY_CURRENT_USER, 0),
        (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_64KEY),
        (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_32KEY),
    )
    subkey = r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
    candidates: list[Path] = []
    for hive, view in locations:
        try:
            root = winreg.OpenKey(hive, subkey, 0, winreg.KEY_READ | view)
        except OSError:
            continue
        with root:
            try:
                subkey_count = winreg.QueryInfoKey(root)[0]
            except OSError:
                continue
            for index in range(subkey_count):
                try:
                    child_name = winreg.EnumKey(root, index)
                    child = winreg.OpenKey(root, child_name)
                except OSError:
                    continue
                with child:
                    try:
                        display_name = winreg.QueryValueEx(child, "DisplayName")[0]
                    except OSError:
                        continue
                    if not (
                        isinstance(display_name, str)
                        and display_name.strip().casefold().startswith("comfy desktop")
                    ):
                        continue
                    try:
                        display_icon = winreg.QueryValueEx(child, "DisplayIcon")[0]
                    except OSError:
                        display_icon = None
                    executable = _display_icon_executable(display_icon)
                    if executable is None:
                        try:
                            install_location = winreg.QueryValueEx(
                                child, "InstallLocation"
                            )[0]
                        except OSError:
                            install_location = None
                        install_path = _string(install_location)
                        if install_path is not None:
                            executable = Path(install_path) / "Comfy Desktop.exe"
                    if executable is not None and _path_is_file(executable):
                        candidates.append(executable)

    candidates = list(_deduplicate_paths(candidates))
    drives = {_windows_drive(path) for path in candidates if _windows_drive(path)}
    if len(drives) > 1:
        warnings.append(
            "Multiple registered Comfy Desktop executables were found on "
            "different drives; the default shared folders were not guessed."
        )
        return None
    return candidates[0] if candidates else None


def _desktop_default_data_root(
    desktop_dir: Path,
    settings: Mapping[str, Any],
    installations: Sequence[ComfyInstallation],
    *,
    desktop_executable: Path | None,
    home_dir: Path,
    local_appdata: Path,
    system_drive: str,
    warnings: list[str],
) -> Path | None:
    """Mirror Desktop's persisted Windows default-data-root decision.

    Current Desktop records the system-drive choice in ``data-location.json``.
    A redirected Desktop executable deliberately has no marker, so only the
    launcher's exact default installation shape is accepted as evidence of that
    drive. Ambiguous state fails closed instead of inventing a model library.
    """

    system = _windows_drive(system_drive)
    executable_drive = (
        _windows_drive(desktop_executable) if desktop_executable is not None else ""
    )
    executable_drive_attested = bool(executable_drive and system)
    if executable_drive_attested and executable_drive != system:
        return Path(f"{executable_drive}\\Comfy-Desktop")

    if not executable_drive_attested:
        explicit_install_dir = _configured_path(settings.get("installDir"), desktop_dir)
        redirected_roots: set[str] = set()
        ambiguous_external_install = False
        for installation in installations:
            install_path = installation.install_path
            if install_path is None:
                continue
            windows_path = PureWindowsPath(str(install_path))
            drive = windows_path.drive.casefold()
            if not drive or not system or drive == system:
                continue
            exact_default_shape = (
                installation.source_id.casefold() == "standalone"
                and not installation.adopted
                and windows_path.parent.name.casefold() == "comfyui-installs"
                and windows_path.parent.parent.name.casefold() == "comfy-desktop"
                and windows_path.parent.parent.parent == PureWindowsPath(f"{drive}\\")
            )
            if explicit_install_dir is None and exact_default_shape:
                redirected_roots.add(str(windows_path.parent.parent))
            else:
                ambiguous_external_install = True
        if len(redirected_roots) == 1 and not ambiguous_external_install:
            return Path(next(iter(redirected_roots)))
        if redirected_roots or ambiguous_external_install:
            warnings.append(
                "Comfy Desktop's default shared-folder drive could not be "
                "proven from its read-only state. Launch Desktop once so it "
                "can record or generate the active model paths."
            )
            return None

    marker_path = desktop_dir / _DESKTOP_DATA_LOCATION_FILE
    marker_exists = _path_is_file(marker_path)
    marker = _read_json(marker_path, warnings)
    if isinstance(marker, Mapping):
        mode = marker.get("mode")
        if mode == "legacy-home":
            return home_dir
        if mode == "local-appdata":
            return local_appdata / "Comfy-Desktop"
        warnings.append(
            "Comfy Desktop has an unrecognized data-location marker; MIVIM "
            "reconstructed Desktop's safe system-drive default."
        )
    elif marker is not None:
        warnings.append(
            "Comfy Desktop has an invalid data-location marker; MIVIM "
            "reconstructed Desktop's safe system-drive default."
        )
    elif marker_exists:
        warnings.append(
            "Comfy Desktop's data-location marker could not be read; MIVIM "
            "reconstructed Desktop's safe system-drive default."
        )

    if _path_exists(home_dir / "ComfyUI-Installs") or _path_exists(
        home_dir / "ComfyUI-Shared"
    ):
        return home_dir
    return local_appdata / "Comfy-Desktop"


def _path_key(path: Path) -> str:
    # These paths describe Windows state even when discovery is tested elsewhere.
    return str(path).replace("/", "\\").rstrip("\\").casefold()


def _deduplicate_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    """Preserve the first occurrence of each Windows model directory."""

    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = _path_key(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return tuple(result)


def model_file_candidates(
    directories: Iterable[Path],
    name: str,
) -> tuple[Path, ...]:
    """Return safe existing matches in ComfyUI's supplied search order."""

    if not isinstance(name, str) or not name.strip():
        return ()
    relative = Path(name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        return ()

    matches: list[Path] = []
    seen: set[str] = set()
    for directory in directories:
        try:
            # Keep lexical containment while allowing a configured model folder
            # to contain Windows junctions/symlinks, as ComfyUI itself does.
            root = directory.absolute()
            candidate = (root / relative).absolute()
            candidate.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            continue
        key = _path_key(candidate)
        if key in seen:
            continue
        try:
            is_file = candidate.is_file()
        except OSError:
            is_file = False
        if is_file:
            seen.add(key)
            matches.append(candidate)
    return tuple(matches)


def _make_model_roots(
    configured: Iterable[tuple[Path, str]],
) -> tuple[ComfyModelRoot, ...]:
    roots: list[ComfyModelRoot] = []
    seen: set[str] = set()
    for path, source in configured:
        key = _path_key(path)
        if key in seen:
            continue
        seen.add(key)
        roots.append(
            ComfyModelRoot(
                path=path,
                loras=path / "loras",
                checkpoints=path / "checkpoints",
                source=source,
                exists=_path_exists(path),
            )
        )
    return tuple(roots)


def _extra_model_layout_file(
    config_path: Path | None,
    source_name: str,
    warnings: list[str] | None = None,
) -> _ExtraModelLayout:
    """Parse one config in the same scalar-path form accepted by ComfyUI."""

    empty = _ExtraModelLayout(roots=(), checkpoints=(), loras=())
    if config_path is None or not _path_is_file(config_path):
        return empty

    def warn(message: str) -> None:
        if warnings is not None and message not in warnings:
            warnings.append(message)

    try:
        if config_path.stat().st_size > _CONFIG_SIZE_LIMIT:
            warn(f"Ignored unusually large Comfy model-path config: {config_path}")
            return empty
        value = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        warn(f"Could not read Comfy model-path config {config_path.name}: {exc}")
        return empty
    if not isinstance(value, Mapping):
        warn(f"Comfy model-path config {config_path.name} is not a YAML mapping.")
        return empty

    def category_paths(
        section: Mapping[object, object],
        key: str,
        base_path: Path,
    ) -> tuple[Path, ...]:
        raw = section.get(key)
        if raw is None:
            return ()
        if not isinstance(raw, str):
            warn(
                f"{config_path.name} uses a non-text {key} value; ComfyUI "
                "accepts one path or a multiline text block."
            )
            return ()
        paths: list[Path] = []
        for item in (line for line in raw.splitlines() if line):
            path = _configured_path(item, base_path)
            if path is not None:
                paths.append(path)
        return tuple(paths)

    roots: list[ComfyModelRoot] = []
    checkpoint_dirs: list[_ConfiguredModelDirectory] = []
    lora_dirs: list[_ConfiguredModelDirectory] = []
    seen: set[tuple[str, str]] = set()
    for section_name, raw_section in value.items():
        if not isinstance(raw_section, Mapping):
            warn(f"Ignored invalid section {section_name!s} in {config_path.name}.")
            continue
        base_path = (
            _configured_path(raw_section.get("base_path"), config_path.parent)
            or config_path.parent
        )
        checkpoints = category_paths(raw_section, "checkpoints", base_path)
        loras = category_paths(raw_section, "loras", base_path)
        source = f"{source_name} ({section_name})"
        raw_default = raw_section.get("is_default", False)
        is_default = bool(raw_default)
        if (
            raw_default is not True
            and raw_default is not False
            and raw_default is not None
        ):
            warn(
                f"{config_path.name} has a non-boolean is_default value in "
                f"section {section_name!s}; ComfyUI treats it according to "
                "YAML runtime truthiness."
            )
        checkpoint_dirs.extend(
            _ConfiguredModelDirectory(
                path=path,
                source=source,
                is_default=is_default,
            )
            for path in checkpoints
        )
        lora_dirs.extend(
            _ConfiguredModelDirectory(
                path=path,
                source=source,
                is_default=is_default,
            )
            for path in loras
        )
        for checkpoint_dir in checkpoints:
            for lora_dir in loras:
                key = (_path_key(checkpoint_dir), _path_key(lora_dir))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    common_path = Path(
                        os.path.commonpath((str(checkpoint_dir), str(lora_dir)))
                    )
                except ValueError:
                    common_path = base_path
                roots.append(
                    ComfyModelRoot(
                        path=common_path,
                        loras=lora_dir,
                        checkpoints=checkpoint_dir,
                        source=source,
                        exists=_path_exists(checkpoint_dir) or _path_exists(lora_dir),
                    )
                )
    return _ExtraModelLayout(
        roots=tuple(roots),
        checkpoints=tuple(checkpoint_dirs),
        loras=tuple(lora_dirs),
    )


def _extra_model_layout(
    comfyui_path: Path | None,
    installation_name: str,
    warnings: list[str] | None = None,
) -> _ExtraModelLayout:
    """Read an install-local ``extra_model_paths.yaml`` if it exists."""

    config_path = (
        comfyui_path / "extra_model_paths.yaml" if comfyui_path is not None else None
    )
    return _extra_model_layout_file(
        config_path,
        f"{installation_name} extra_model_paths.yaml",
        warnings,
    )


def _extra_model_paths(
    comfyui_path: Path | None,
    installation_name: str,
) -> tuple[ComfyModelRoot, ...]:
    """Return paired extra roots for compatibility with pre-0.6 callers."""

    return _extra_model_layout(comfyui_path, installation_name).roots


def _ordered_model_directories(
    base: Iterable[Path],
    configured: Iterable[_ConfiguredModelDirectory],
) -> tuple[Path, ...]:
    """Apply ComfyUI's add-model-path prepend/append ordering semantics."""

    ordered = list(_deduplicate_paths(base))
    for item in configured:
        key = _path_key(item.path)
        existing_index = next(
            (index for index, path in enumerate(ordered) if _path_key(path) == key),
            None,
        )
        if existing_index is not None:
            if item.is_default and existing_index != 0:
                ordered.insert(0, ordered.pop(existing_index))
            continue
        if item.is_default:
            ordered.insert(0, item.path)
        else:
            ordered.append(item.path)
    return tuple(ordered)


def _generated_layout_matches_roots(
    layout: _ExtraModelLayout,
    roots: Sequence[ComfyModelRoot],
) -> bool:
    """Return whether Desktop's generated YAML matches current settings."""

    expected_checkpoints = tuple(root.checkpoints for root in roots)
    expected_loras = tuple(root.loras for root in roots)
    actual_checkpoints = _ordered_model_directories((), layout.checkpoints)
    actual_loras = _ordered_model_directories((), layout.loras)
    return tuple(_path_key(path) for path in actual_checkpoints) == tuple(
        _path_key(path) for path in expected_checkpoints
    ) and tuple(_path_key(path) for path in actual_loras) == tuple(
        _path_key(path) for path in expected_loras
    )


def _effective_model_layout(
    installation: ComfyInstallation | None,
    shared_roots: tuple[ComfyModelRoot, ...],
    legacy_roots: tuple[ComfyModelRoot, ...],
    *,
    desktop_config_dir: Path | None = None,
    legacy_config_dir: Path | None = None,
    output_dir: Path | None = None,
    launch_paths: _LaunchPathConfiguration | None = None,
    running_since_ms: int | None = None,
    warnings: list[str] | None = None,
) -> _EffectiveModelLayout:
    if installation is None:
        roots = shared_roots or legacy_roots
        checkpoint_dirs = _deduplicate_paths(root.checkpoints for root in roots)
        lora_dirs = _deduplicate_paths(root.loras for root in roots)
        legacy_extra = _ExtraModelLayout(roots=(), checkpoints=(), loras=())
        if not shared_roots and legacy_config_dir is not None:
            legacy_extra = _extra_model_layout_file(
                legacy_config_dir / "extra_models_config.yaml",
                "Legacy Comfy Desktop extra_models_config.yaml",
                warnings,
            )
            checkpoint_dirs = _ordered_model_directories(
                checkpoint_dirs,
                legacy_extra.checkpoints,
            )
            lora_dirs = _ordered_model_directories(lora_dirs, legacy_extra.loras)
        if output_dir is not None:
            checkpoint_dirs = _deduplicate_paths(
                (*checkpoint_dirs, output_dir / "checkpoints")
            )
            lora_dirs = _deduplicate_paths((*lora_dirs, output_dir / "loras"))
        unique_roots: list[ComfyModelRoot] = []
        seen_roots: set[tuple[str, str]] = set()
        for root in (*roots, *legacy_extra.roots):
            key = (_path_key(root.checkpoints), _path_key(root.loras))
            if key not in seen_roots:
                seen_roots.add(key)
                unique_roots.append(root)
        return _EffectiveModelLayout(
            roots=tuple(unique_roots),
            checkpoint_dirs=checkpoint_dirs,
            lora_dirs=lora_dirs,
        )

    resolved_launch_paths = launch_paths or _launch_path_configuration(
        installation,
        shared_input_dir=None,
        shared_output_dir=output_dir,
        warnings=warnings,
    )
    own = (
        _make_model_roots(((resolved_launch_paths.models_dir, installation.name),))
        if resolved_launch_paths.models_dir is not None
        else ()
    )
    local_extra = _extra_model_layout(
        installation.comfyui_path,
        installation.name,
        warnings,
    )
    launch_extras: list[_ExtraModelLayout] = []
    for config_path in resolved_launch_paths.extra_model_configs:
        if not _path_is_file(config_path):
            message = (
                f"{installation.name} starts with a missing extra model-path "
                f"configuration: {config_path}"
            )
            if warnings is not None and message not in warnings:
                warnings.append(message)
            continue
        launch_extras.append(
            _extra_model_layout_file(
                config_path,
                f"{installation.name} startup arguments ({config_path.name})",
                warnings,
            )
        )
    if installation.use_shared_models:
        launcher_roots = shared_roots
        preferred_roots = (*shared_roots, *own)
        launcher_primary = shared_roots[0].path if shared_roots else None
        generated_name = "shared_model_paths.yaml"
    else:
        launcher_roots = _make_model_roots(
            (path, f"{installation.name} extra models")
            for path in installation.model_dirs
        )
        launcher_primary = None
        if installation.model_dirs_primary is not None:
            primary_key = _path_key(installation.model_dirs_primary)
            primary = tuple(
                root for root in launcher_roots if _path_key(root.path) == primary_key
            )
            remainder = tuple(root for root in launcher_roots if root not in primary)
            if primary:
                launcher_primary = primary[0].path
                preferred_roots = (*primary, *own, *remainder)
            else:
                preferred_roots = (*own, *launcher_roots)
        else:
            preferred_roots = (*own, *launcher_roots)
        generated_name = f"instance-model-paths/{installation.id}.yaml"

    generated_path = (
        desktop_config_dir / Path(generated_name)
        if desktop_config_dir is not None
        else None
    )
    if generated_path is not None and _path_is_file(generated_path):
        generated_extra = _extra_model_layout_file(
            generated_path,
            f"Comfy Desktop {generated_path.name}",
            warnings,
        )
        generated_matches_settings = _generated_layout_matches_roots(
            generated_extra,
            launcher_roots,
        )
        if running_since_ms is not None:
            try:
                generated_modified_ms = generated_path.stat().st_mtime_ns // 1_000_000
            except OSError as exc:
                if warnings is not None:
                    warnings.append(
                        "MIVIM could not verify the running ComfyUI model-path "
                        f"snapshot: {exc}. Restart this instance in Comfy Desktop."
                    )
                return _EffectiveModelLayout(
                    roots=(),
                    checkpoint_dirs=(),
                    lora_dirs=(),
                )
            if generated_modified_ms > running_since_ms:
                if warnings is not None:
                    warnings.append(
                        "Comfy Desktop rewrote model paths after this ComfyUI "
                        "instance started. Restart the selected instance so MIVIM "
                        "can verify exactly which model library is active."
                    )
                return _EffectiveModelLayout(
                    roots=(),
                    checkpoint_dirs=(),
                    lora_dirs=(),
                )
            if (
                launcher_roots
                and not generated_extra.checkpoints
                and not generated_extra.loras
            ):
                if warnings is not None:
                    warnings.append(
                        "The running ComfyUI model-path snapshot is incomplete. "
                        "Restart the selected instance in Comfy Desktop."
                    )
                return _EffectiveModelLayout(
                    roots=(),
                    checkpoint_dirs=(),
                    lora_dirs=(),
                )
            desktop_extra = generated_extra
            if not generated_matches_settings:
                preferred_roots = (*generated_extra.roots, *own)
                if warnings is not None:
                    warnings.append(
                        "The running ComfyUI instance still uses its launch-time "
                        "model folders. Restart it in Comfy Desktop to apply the "
                        "new storage settings."
                    )
        elif generated_matches_settings:
            desktop_extra = generated_extra
        else:
            if warnings is not None:
                warnings.append(
                    f"Ignored stale Comfy Desktop model-path file "
                    f"{generated_path.name}; current Desktop storage settings "
                    "take effect when the instance starts."
                )
            desktop_extra = _ExtraModelLayout(roots=(), checkpoints=(), loras=())
    else:
        if running_since_ms is not None and launcher_roots:
            if warnings is not None:
                warnings.append(
                    "MIVIM could not find the model-path snapshot used by the "
                    "running ComfyUI instance. Restart it in Comfy Desktop."
                )
            return _EffectiveModelLayout(
                roots=(),
                checkpoint_dirs=(),
                lora_dirs=(),
            )
        desktop_extra = _ExtraModelLayout(
            roots=(),
            checkpoints=(),
            loras=(),
        )
    if not desktop_extra.checkpoints and not desktop_extra.loras:
        desktop_extra = _ExtraModelLayout(
            roots=(),
            checkpoints=tuple(
                _ConfiguredModelDirectory(
                    root.checkpoints,
                    f"{installation.name} Desktop model library",
                    launcher_primary is not None
                    and _path_key(root.path) == _path_key(launcher_primary),
                )
                for root in launcher_roots
            ),
            loras=tuple(
                _ConfiguredModelDirectory(
                    root.loras,
                    f"{installation.name} Desktop model library",
                    launcher_primary is not None
                    and _path_key(root.path) == _path_key(launcher_primary),
                )
                for root in launcher_roots
            ),
        )

    # ComfyUI initializes the install's built-in models directory, then loads
    # install-local YAML, then Desktop's generated CLI YAML, and finally adds
    # category folders under the configured output directory.
    checkpoint_dirs = _ordered_model_directories(
        (root.checkpoints for root in own),
        local_extra.checkpoints,
    )
    for extra in launch_extras:
        checkpoint_dirs = _ordered_model_directories(
            checkpoint_dirs,
            extra.checkpoints,
        )
    checkpoint_dirs = _ordered_model_directories(
        checkpoint_dirs,
        desktop_extra.checkpoints,
    )
    lora_dirs = _ordered_model_directories(
        (root.loras for root in own),
        local_extra.loras,
    )
    for extra in launch_extras:
        lora_dirs = _ordered_model_directories(lora_dirs, extra.loras)
    lora_dirs = _ordered_model_directories(
        lora_dirs,
        desktop_extra.loras,
    )
    effective_output_dir = resolved_launch_paths.output_dir
    if effective_output_dir is not None:
        checkpoint_dirs = _deduplicate_paths(
            (*checkpoint_dirs, effective_output_dir / "checkpoints")
        )
        lora_dirs = _deduplicate_paths((*lora_dirs, effective_output_dir / "loras"))

    unique: list[ComfyModelRoot] = []
    seen: set[tuple[str, str]] = set()
    for root in (
        *preferred_roots,
        *local_extra.roots,
        *(root for extra in launch_extras for root in extra.roots),
        *desktop_extra.roots,
    ):
        key = (_path_key(root.checkpoints), _path_key(root.loras))
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return _EffectiveModelLayout(
        roots=tuple(unique),
        checkpoint_dirs=checkpoint_dirs,
        lora_dirs=lora_dirs,
    )


def _effective_model_roots(
    installation: ComfyInstallation | None,
    shared_roots: tuple[ComfyModelRoot, ...],
    legacy_roots: tuple[ComfyModelRoot, ...],
) -> tuple[ComfyModelRoot, ...]:
    """Return paired roots for compatibility with pre-0.6 callers."""

    return _effective_model_layout(
        installation,
        shared_roots,
        legacy_roots,
    ).roots


def select_comfy_installation(
    discovery: ComfyDesktopDiscovery,
    installation_id: str,
) -> ComfyDesktopDiscovery:
    """Bind discovery paths and runtime hints to one explicit Desktop instance."""

    selected = next(
        (
            installation
            for installation in discovery.installations
            if installation.id == installation_id and installation.is_local
        ),
        None,
    )
    if selected is None:
        return replace(
            discovery,
            selected_installation=None,
            model_roots=(),
            input_dir=None,
            output_dir=None,
            host_candidates=(),
            checkpoint_dirs=(),
            lora_dirs=(),
        )

    same_named_installations = tuple(
        installation
        for installation in discovery.installations
        if installation.is_local
        and installation.name.casefold() == selected.name.casefold()
    )
    matched_runtime_hints = (
        tuple(
            hint
            for hint in discovery.runtime_hints
            if hint.installation_name.casefold() == selected.name.casefold()
        )
        if len(same_named_installations) == 1
        else ()
    )
    matched_hosts = tuple(hint.host for hint in matched_runtime_hints)
    warnings = list(discovery.warnings)
    if len(same_named_installations) > 1:
        duplicate_warning = (
            f'Multiple Comfy Desktop installations are named "{selected.name}". '
            "Their running ports cannot be distinguished safely; rename one "
            "installation in Comfy Desktop."
        )
        if duplicate_warning not in warnings:
            warnings.append(duplicate_warning)
    # A selected Desktop installation is an identity boundary.  Falling back
    # to another installation's live port can train into and validate against
    # different model libraries without any obvious error for a casual user.
    hosts = _deduplicate_hosts(matched_hosts)
    shared_input_dir = discovery.shared_input_dir
    shared_output_dir = discovery.shared_output_dir
    if (
        discovery.selected_installation is not None
        and discovery.selected_installation.use_shared_input_output
    ):
        shared_input_dir = shared_input_dir or discovery.input_dir
        shared_output_dir = shared_output_dir or discovery.output_dir
    launch_paths = _launch_path_configuration(
        selected,
        shared_input_dir=shared_input_dir,
        shared_output_dir=shared_output_dir,
        warnings=warnings,
    )
    layout = _effective_model_layout(
        selected,
        discovery.shared_model_roots,
        discovery.legacy_model_roots,
        desktop_config_dir=discovery.config_dir,
        legacy_config_dir=discovery.legacy_config_dir,
        output_dir=launch_paths.output_dir,
        launch_paths=launch_paths,
        running_since_ms=(
            min(hint.timestamp for hint in matched_runtime_hints)
            if matched_runtime_hints
            else None
        ),
        warnings=warnings,
    )
    if matched_runtime_hints and _desktop_settings_changed_after_start(
        discovery.config_dir,
        min(hint.timestamp for hint in matched_runtime_hints),
        warnings,
    ):
        layout = _EffectiveModelLayout(roots=(), checkpoint_dirs=(), lora_dirs=())
    paths_attested = (
        (not selected.use_shared_models or bool(discovery.shared_model_roots))
        and (
            not selected.use_shared_input_output or launch_paths.output_dir is not None
        )
        and bool(layout.checkpoint_dirs)
        and bool(layout.lora_dirs)
    )
    if not paths_attested:
        warning = (
            "The selected Comfy Desktop instance uses shared storage, but its "
            "effective folders could not be proven. Launch Desktop once, then "
            "refresh MIVIM before selecting a model."
        )
        if warning not in warnings:
            warnings.append(warning)
        layout = _EffectiveModelLayout(roots=(), checkpoint_dirs=(), lora_dirs=())
    return replace(
        discovery,
        selected_installation=selected,
        model_roots=layout.roots,
        input_dir=launch_paths.input_dir if paths_attested else None,
        output_dir=launch_paths.output_dir if paths_attested else None,
        host_candidates=hosts,
        warnings=tuple(warnings),
        checkpoint_dirs=layout.checkpoint_dirs,
        lora_dirs=layout.lora_dirs,
    )


def _normalise_local_host(value: object, default_port: int = 8188) -> str | None:
    raw = _string(value)
    if raw is None:
        return None
    candidate = raw if "://" in raw else f"http://{raw}"
    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname.casefold() if parsed.hostname else ""
        port = parsed.port or default_port
    except ValueError:
        return None
    if (
        parsed.scheme.casefold() != "http"
        or hostname not in _LOCAL_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or not 1 <= port <= 65535
    ):
        return None
    display_host = f"[{hostname}]" if ":" in hostname else hostname
    return f"{display_host}:{port}"


def _saved_host_candidates(value: object) -> tuple[str, ...]:
    candidates: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            sibling_port = next(
                (
                    _integer(child)
                    for key, child in item.items()
                    if str(key).replace("_", "").replace("-", "").casefold()
                    in _PORT_KEYS
                    and _integer(child) is not None
                ),
                None,
            )
            for key, child in item.items():
                normalised_key = str(key).replace("_", "").replace("-", "").casefold()
                if normalised_key in _HOST_KEYS:
                    candidate = _normalise_local_host(
                        child, sibling_port if sibling_port is not None else 8188
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                elif normalised_key in _PORT_KEYS:
                    port = _integer(child)
                    if port is not None and 1 <= port <= 65535:
                        candidates.append(f"127.0.0.1:{port}")
                if isinstance(child, (Mapping, list, tuple)):
                    visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(candidates)


def _deduplicate_hosts(hosts: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for host in hosts:
        normalised = _normalise_local_host(host)
        if normalised is None:
            continue
        key = normalised.casefold()
        if key not in seen:
            seen.add(key)
            result.append(normalised)
    return tuple(result)


def _port_lock_hints(
    desktop_dir: Path,
    warnings: list[str],
) -> tuple[ComfyRuntimeHint, ...]:
    """Read current Desktop runtime identity without treating it as control state."""

    lock_dir = desktop_dir / "port-locks"
    if not _path_exists(lock_dir):
        return ()
    try:
        candidates = sorted(
            (path for path in lock_dir.iterdir() if path.is_file()),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )[:64]
    except OSError as exc:
        warnings.append(f"Could not inspect Comfy Desktop port locks: {exc}")
        return ()

    hints: list[ComfyRuntimeHint] = []
    for path in candidates:
        match = _PORT_LOCK_NAME.fullmatch(path.name)
        if match is None:
            continue
        port = int(match.group(1))
        if not 1 <= port <= 65535:
            continue
        value = _read_json(path, warnings)
        if not isinstance(value, Mapping):
            warnings.append(f"Ignored malformed Comfy port lock: {path.name}")
            continue
        pid = _integer(value.get("pid"))
        installation_name = _string(value.get("installationName"))
        timestamp = _integer(value.get("timestamp"))
        if pid is None or pid <= 0 or installation_name is None or timestamp is None:
            warnings.append(f"Ignored incomplete Comfy port lock: {path.name}")
            continue
        if not _process_is_running(pid):
            warnings.append(f"Ignored stale Comfy port lock: {path.name}")
            continue
        hints.append(
            ComfyRuntimeHint(
                host=f"127.0.0.1:{port}",
                installation_name=installation_name,
                pid=pid,
                timestamp=timestamp,
            )
        )
    return tuple(hints)


def _desktop_settings_changed_after_start(
    desktop_dir: Path | None,
    started_at_ms: int,
    warnings: list[str],
) -> bool:
    """Fail closed when live input/output settings lack a launch snapshot."""

    if desktop_dir is None:
        return False
    settings_path = desktop_dir / "settings.json"
    if not _path_is_file(settings_path):
        return False
    try:
        changed = settings_path.stat().st_mtime_ns // 1_000_000 > started_at_ms
    except OSError as exc:
        warnings.append(
            "MIVIM could not compare the running ComfyUI instance with Desktop "
            f"storage settings: {exc}. Restart the selected instance."
        )
        return True
    if changed:
        warnings.append(
            "Comfy Desktop storage settings changed after the selected ComfyUI "
            "instance started. Restart that instance before training so MIVIM "
            "can verify its active input, output, and model folders."
        )
    return changed


def _process_is_running(pid: int) -> bool:
    """Return whether ``pid`` currently identifies a running local process."""

    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        # OpenProcess is read-only here; unlike os.kill(..., 0) on Windows it
        # cannot accidentally terminate the process.
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _port_lock_hosts(desktop_dir: Path, warnings: list[str]) -> tuple[str, ...]:
    """Compatibility wrapper returning hosts from identity-bearing lock files."""

    return tuple(hint.host for hint in _port_lock_hints(desktop_dir, warnings))


def discover_comfy_desktop(
    appdata: Path | None = None,
    *,
    desktop_executable: Path | None = None,
    home_dir: Path | None = None,
    local_appdata: Path | None = None,
    system_drive: str | None = None,
) -> ComfyDesktopDiscovery:
    """Discover Comfy Desktop state without modifying it.

    ``appdata`` is injectable for tests and portable tools.  When omitted, the
    Windows ``APPDATA`` environment variable is used.  On platforms without
    that variable discovery safely returns an empty snapshot.
    """

    warnings: list[str] = []
    injected_appdata = appdata is not None
    roaming = appdata
    if roaming is None:
        configured_appdata = os.environ.get("APPDATA")
        roaming = Path(configured_appdata) if configured_appdata else None

    if roaming is None:
        defaults = tuple(f"127.0.0.1:{port}" for port in _DEFAULT_LOCAL_PORTS)
        return ComfyDesktopDiscovery(
            config_dir=None,
            legacy_config_dir=None,
            installations=(),
            selected_installation=None,
            model_roots=(),
            input_dir=None,
            output_dir=None,
            host_candidates=defaults,
            launcher=_launcher(),
            warnings=("APPDATA is unavailable; Comfy Desktop was not auto-detected.",),
        )

    if home_dir is None or local_appdata is None:
        if injected_appdata:
            if (
                roaming.name.casefold() == "roaming"
                and roaming.parent.name.casefold() == "appdata"
            ):
                inferred_home = roaming.parent.parent
                inferred_local = roaming.parent / "Local"
            else:
                fixture_root = (
                    roaming.parent if roaming.name.casefold() == "roaming" else roaming
                )
                inferred_home = fixture_root
                inferred_local = fixture_root / "Local"
        else:
            inferred_home = Path.home()
            configured_local = os.environ.get("LOCALAPPDATA")
            inferred_local = (
                Path(configured_local)
                if configured_local
                else inferred_home / "AppData" / "Local"
            )
        home_dir = home_dir or inferred_home
        local_appdata = local_appdata or inferred_local
    assert home_dir is not None
    assert local_appdata is not None
    system_drive = system_drive or os.environ.get("SystemDrive", "")

    desktop_dir = roaming / "Comfy Desktop"
    legacy_dir = roaming / "ComfyUI"
    desktop_files_found = any(
        _path_is_file(desktop_dir / name)
        for name in (
            "settings.json",
            "installations.json",
            "last-session.json",
            _DESKTOP_DATA_LOCATION_FILE,
        )
    )
    legacy_file_found = _path_is_file(legacy_dir / "config.json")

    settings_value = _read_json(desktop_dir / "settings.json", warnings)
    installations_value = _read_json(desktop_dir / "installations.json", warnings)
    last_session_value = _read_json(desktop_dir / "last-session.json", warnings)
    legacy_value = _read_json(legacy_dir / "config.json", warnings)

    settings: Mapping[str, Any] = (
        settings_value if isinstance(settings_value, Mapping) else {}
    )
    last_session: Mapping[str, Any] = (
        last_session_value if isinstance(last_session_value, Mapping) else {}
    )
    legacy: Mapping[str, Any] = (
        legacy_value if isinstance(legacy_value, Mapping) else {}
    )

    if settings_value is not None and not isinstance(settings_value, Mapping):
        warnings.append("settings.json does not contain a JSON object.")
    if last_session_value is not None and not isinstance(last_session_value, Mapping):
        warnings.append("last-session.json does not contain a JSON object.")
    if legacy_value is not None and not isinstance(legacy_value, Mapping):
        warnings.append("Legacy config.json does not contain a JSON object.")

    installation_items: Sequence[object]
    if isinstance(installations_value, list):
        installation_items = installations_value
    elif installations_value is None:
        installation_items = ()
    else:
        installation_items = ()
        warnings.append("installations.json does not contain a JSON array.")
    installations = tuple(
        installation
        for item in installation_items
        if (installation := _installation_from_json(item, desktop_dir)) is not None
    )
    selected = _selected_local_installation(installations, settings, last_session)

    configured_model_paths = _model_paths_from_settings(settings, desktop_dir)
    configured_input_dir = _configured_path(settings.get("inputDir"), desktop_dir)
    configured_output_dir = _configured_path(settings.get("outputDir"), desktop_dir)
    needs_default_root = desktop_files_found and (
        not configured_model_paths
        or not any(_path_exists(path) for path in configured_model_paths)
        or configured_input_dir is None
        or not _path_exists(configured_input_dir)
        or configured_output_dir is None
        or not _path_exists(configured_output_dir)
    )
    resolved_desktop_executable = (
        desktop_executable
        if desktop_executable is not None
        else (
            _registered_comfy_desktop_executable(warnings)
            if not injected_appdata
            else None
        )
    )
    default_data_root = (
        _desktop_default_data_root(
            desktop_dir,
            settings,
            installations,
            desktop_executable=resolved_desktop_executable,
            home_dir=home_dir,
            local_appdata=local_appdata,
            system_drive=system_drive,
            warnings=warnings,
        )
        if needs_default_root
        else None
    )
    default_shared_root = (
        default_data_root / "ComfyUI-Shared" if default_data_root is not None else None
    )
    if not configured_model_paths or not any(
        _path_exists(path) for path in configured_model_paths
    ):
        configured_model_paths = (
            _deduplicate_paths(
                (
                    default_shared_root / "models",
                    *configured_model_paths,
                )
            )
            if default_shared_root is not None
            else ()
        )
    shared_roots = _make_model_roots(
        (path, "Comfy Desktop settings") for path in configured_model_paths
    )

    legacy_base = _configured_path(
        legacy.get("basePath") or legacy.get("base_path"), legacy_dir
    )
    legacy_roots = (
        _make_model_roots(((legacy_base / "models", "Legacy ComfyUI Desktop"),))
        if legacy_base is not None
        else ()
    )
    shared_input_dir = (
        configured_input_dir
        if configured_input_dir is not None and _path_exists(configured_input_dir)
        else (
            default_shared_root / "input" if default_shared_root is not None else None
        )
    )
    shared_output_dir = (
        configured_output_dir
        if configured_output_dir is not None and _path_exists(configured_output_dir)
        else (
            default_shared_root / "output" if default_shared_root is not None else None
        )
    )
    input_dir = shared_input_dir
    output_dir = shared_output_dir
    launch_paths: _LaunchPathConfiguration | None = None
    if selected is not None:
        launch_paths = _launch_path_configuration(
            selected,
            shared_input_dir=shared_input_dir,
            shared_output_dir=shared_output_dir,
            warnings=warnings,
        )
        input_dir = launch_paths.input_dir
        output_dir = launch_paths.output_dir
    elif legacy_base is not None:
        input_dir = input_dir or _configured_path(
            legacy.get("inputDirectory") or legacy.get("inputDir"), legacy_dir
        )
        output_dir = output_dir or _configured_path(
            legacy.get("outputDirectory") or legacy.get("outputDir"), legacy_dir
        )
        input_dir = input_dir or legacy_base / "input"
        output_dir = output_dir or legacy_base / "output"

    runtime_hints = _port_lock_hints(desktop_dir, warnings)
    selected_name_is_unique = bool(
        selected is not None
        and sum(
            1
            for installation in installations
            if installation.is_local
            and installation.name.casefold() == selected.name.casefold()
        )
        == 1
    )
    selected_runtime_hints = (
        tuple(
            hint
            for hint in runtime_hints
            if selected is not None
            and hint.installation_name.casefold() == selected.name.casefold()
        )
        if selected_name_is_unique
        else ()
    )
    running_since_ms = (
        min(hint.timestamp for hint in selected_runtime_hints)
        if selected_runtime_hints
        else None
    )
    effective_layout = _effective_model_layout(
        selected,
        shared_roots,
        legacy_roots,
        desktop_config_dir=desktop_dir,
        legacy_config_dir=legacy_dir,
        output_dir=output_dir,
        launch_paths=launch_paths,
        running_since_ms=running_since_ms,
        warnings=warnings,
    )
    if running_since_ms is not None and _desktop_settings_changed_after_start(
        desktop_dir,
        running_since_ms,
        warnings,
    ):
        effective_layout = _EffectiveModelLayout(
            roots=(),
            checkpoint_dirs=(),
            lora_dirs=(),
        )
    selected_paths_attested = selected is None or (
        (not selected.use_shared_models or bool(shared_roots))
        and (not selected.use_shared_input_output or shared_output_dir is not None)
        and bool(effective_layout.checkpoint_dirs)
        and bool(effective_layout.lora_dirs)
    )
    if not selected_paths_attested:
        warnings.append(
            "The selected Comfy Desktop instance uses shared storage, but its "
            "effective folders could not be proven. Launch Desktop once, then "
            "refresh MIVIM before selecting a model."
        )
        effective_layout = _EffectiveModelLayout(
            roots=(),
            checkpoint_dirs=(),
            lora_dirs=(),
        )
        input_dir = None
        output_dir = None

    saved_hosts: list[str] = [hint.host for hint in runtime_hints]
    for document in (settings, last_session, legacy):
        saved_hosts.extend(_saved_host_candidates(document))
    if selected is not None:
        selected_json = next(
            (
                item
                for item in installation_items
                if isinstance(item, Mapping) and item.get("id") == selected.id
            ),
            None,
        )
        saved_hosts.extend(_saved_host_candidates(selected_json))
    saved_hosts.extend(f"127.0.0.1:{port}" for port in _DEFAULT_LOCAL_PORTS)

    return ComfyDesktopDiscovery(
        config_dir=desktop_dir if desktop_files_found else None,
        legacy_config_dir=legacy_dir if legacy_file_found else None,
        installations=installations,
        selected_installation=selected,
        model_roots=effective_layout.roots,
        input_dir=input_dir,
        output_dir=output_dir,
        host_candidates=_deduplicate_hosts(saved_hosts),
        launcher=_launcher(),
        warnings=tuple(warnings),
        runtime_hints=runtime_hints,
        shared_model_roots=shared_roots,
        legacy_model_roots=legacy_roots,
        checkpoint_dirs=effective_layout.checkpoint_dirs,
        lora_dirs=effective_layout.lora_dirs,
        shared_input_dir=shared_input_dir,
        shared_output_dir=shared_output_dir,
        desktop_executable=resolved_desktop_executable,
    )


def _get_json(url: str, timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": f"MIVIM-LoRA-Studio/{__version__}",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return json.loads(payload)


def _get_api_json(base_url: str, path: str, timeout: float) -> Any:
    """Prefer the current ``/api`` namespace and retain older compatibility."""

    modern_path = f"/api{path}"
    try:
        return _get_json(f"{base_url}{modern_path}", timeout)
    except (OSError, HTTPError, URLError, UnicodeError, json.JSONDecodeError):
        return _get_json(f"{base_url}{path}", timeout)


def _model_names_from_object_info(
    value: object, node_name: str, field: str
) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    node = value.get(node_name)
    if not isinstance(node, Mapping):
        return ()
    inputs = node.get("input")
    if not isinstance(inputs, Mapping):
        return ()
    required = inputs.get("required")
    if not isinstance(required, Mapping):
        return ()
    definition = required.get(field)
    if (
        not isinstance(definition, Sequence)
        or isinstance(definition, str)
        or not definition
    ):
        return ()
    choices = definition[0]
    if not isinstance(choices, Sequence) or isinstance(choices, str):
        return ()
    return _clean_model_names(choices)


def _clean_model_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    names = {
        item.strip()
        for item in value
        if isinstance(item, str)
        and item.strip()
        and Path(item.strip()).suffix.casefold() in _MODEL_FILE_SUFFIXES
    }
    return tuple(sorted(names, key=str.casefold))


def _enumerate_model_type(
    base_url: str,
    folder: str,
    node_name: str,
    field: str,
    timeout: float,
) -> tuple[str, ...]:
    try:
        direct = _get_api_json(base_url, f"/models/{folder}", timeout)
    except (OSError, HTTPError, URLError, UnicodeError, json.JSONDecodeError):
        direct = None
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes, bytearray)):
        return _clean_model_names(direct)

    info = _get_api_json(base_url, f"/object_info/{node_name}", timeout)
    return _model_names_from_object_info(info, node_name, field)


def enumerate_live_models(
    host: str, timeout: float = 1.0
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return checkpoint and LoRA names exposed by a running local ComfyUI.

    Only loopback HTTP hosts are accepted.  The modern ``/models`` API is used
    first, with node metadata as a compatibility fallback.
    """

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    normalised = _normalise_local_host(host)
    if normalised is None:
        raise ValueError("ComfyUI host must be a loopback HTTP address.")
    base_url = f"http://{normalised}"
    checkpoints = _enumerate_model_type(
        base_url,
        "checkpoints",
        "CheckpointLoaderSimple",
        "ckpt_name",
        timeout,
    )
    loras = _enumerate_model_type(base_url, "loras", "LoraLoader", "lora_name", timeout)
    return checkpoints, loras


def probe_comfy_api(hosts: Iterable[str], timeout: float = 0.5) -> ComfyApiStatus:
    """Probe explicit local candidates and return the first running ComfyUI.

    The probe never expands beyond the supplied loopback addresses.  A server
    that answers ``/system_stats`` is reported as reachable even when an older
    version cannot enumerate its models.
    """

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    candidates = _deduplicate_hosts(hosts)
    if not candidates:
        return ComfyApiStatus(
            reachable=False,
            host=None,
            checkpoints=(),
            loras=(),
            error="No valid local ComfyUI addresses were provided.",
        )

    failures: list[str] = []
    for host in candidates:
        try:
            stats = _get_api_json(f"http://{host}", "/system_stats", timeout)
            if not isinstance(stats, Mapping):
                raise TypeError("system_stats returned non-object JSON")
        except (
            OSError,
            HTTPError,
            URLError,
            UnicodeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            failures.append(f"{host}: {exc}")
            continue

        try:
            checkpoints, loras = enumerate_live_models(host, timeout)
        except (
            OSError,
            HTTPError,
            URLError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            return ComfyApiStatus(
                reachable=True,
                host=host,
                checkpoints=(),
                loras=(),
                error=f"ComfyUI is running, but its models could not be listed: {exc}",
            )
        return ComfyApiStatus(
            reachable=True,
            host=host,
            checkpoints=checkpoints,
            loras=loras,
            error=None,
        )

    return ComfyApiStatus(
        reachable=False,
        host=None,
        checkpoints=(),
        loras=(),
        error="ComfyUI did not answer at the configured local addresses"
        + (f" ({'; '.join(failures)})." if failures else "."),
    )
