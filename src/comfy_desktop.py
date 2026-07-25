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

    @property
    def detected(self) -> bool:
        """Return whether either desktop configuration layout was found."""

        return self.config_dir is not None or self.legacy_config_dir is not None

    @property
    def preferred_model_root(self) -> ComfyModelRoot | None:
        """Return the first configured model root, preferring one that exists."""

        return next(
            (root for root in self.model_roots if root.exists),
            self.model_roots[0] if self.model_roots else None,
        )


@dataclass(frozen=True, slots=True)
class ComfyApiStatus:
    """Result of probing a local ComfyUI HTTP API."""

    reachable: bool
    host: str | None
    checkpoints: tuple[str, ...]
    loras: tuple[str, ...]
    error: str | None


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


def _path_key(path: Path) -> str:
    # These paths describe Windows state even when discovery is tested elsewhere.
    return str(path).replace("/", "\\").rstrip("\\").casefold()


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


def _extra_model_paths(
    comfyui_path: Path | None,
    installation_name: str,
) -> tuple[ComfyModelRoot, ...]:
    """Read model folders enabled by ComfyUI's ``extra_model_paths.yaml``.

    This is deliberately read-only and conservative: only entries that define
    both checkpoint and LoRA folders can become a pipeline destination.  An
    invalid or future configuration shape is ignored so Desktop discovery can
    still start normally.
    """

    if comfyui_path is None:
        return ()
    config_path = comfyui_path / "extra_model_paths.yaml"
    if not _path_is_file(config_path):
        return ()
    try:
        if config_path.stat().st_size > _CONFIG_SIZE_LIMIT:
            return ()
        value = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return ()
    if not isinstance(value, Mapping):
        return ()

    def category_paths(
        section: Mapping[object, object],
        key: str,
        base_path: Path,
    ) -> tuple[Path, ...]:
        raw = section.get(key)
        values: Sequence[object]
        if isinstance(raw, str):
            values = tuple(line.strip() for line in raw.splitlines() if line.strip())
        elif isinstance(raw, Sequence):
            values = raw
        else:
            values = ()
        paths: list[Path] = []
        for item in values:
            path = _configured_path(item, base_path)
            if path is not None:
                paths.append(path)
        return tuple(paths)

    roots: list[ComfyModelRoot] = []
    seen: set[tuple[str, str]] = set()
    for section_name, raw_section in value.items():
        if not isinstance(raw_section, Mapping):
            continue
        base_path = (
            _configured_path(raw_section.get("base_path"), config_path.parent)
            or config_path.parent
        )
        checkpoints = category_paths(raw_section, "checkpoints", base_path)
        loras = category_paths(raw_section, "loras", base_path)
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
                        source=(
                            f"{installation_name} extra_model_paths.yaml "
                            f"({section_name})"
                        ),
                        exists=_path_exists(checkpoint_dir) or _path_exists(lora_dir),
                    )
                )
    return tuple(roots)


def _effective_model_roots(
    installation: ComfyInstallation | None,
    shared_roots: tuple[ComfyModelRoot, ...],
    legacy_roots: tuple[ComfyModelRoot, ...],
) -> tuple[ComfyModelRoot, ...]:
    if installation is None:
        return shared_roots or legacy_roots

    own = (
        _make_model_roots(((installation.comfyui_path / "models", installation.name),))
        if installation.comfyui_path is not None
        else ()
    )
    extra = _extra_model_paths(installation.comfyui_path, installation.name)
    if installation.use_shared_models:
        visible_own = (
            own if any(root.exists for root in own) or not shared_roots else ()
        )
        candidates = (*shared_roots, *visible_own, *extra)
    else:
        external = _make_model_roots(
            (path, f"{installation.name} extra models")
            for path in installation.model_dirs
        )
        if installation.model_dirs_primary is not None:
            primary_key = _path_key(installation.model_dirs_primary)
            primary = tuple(
                root for root in external if _path_key(root.path) == primary_key
            )
            remainder = tuple(root for root in external if root not in primary)
            candidates = (*primary, *own, *remainder, *extra)
        else:
            candidates = (*own, *external, *extra)

    unique: list[ComfyModelRoot] = []
    seen: set[tuple[str, str]] = set()
    for root in candidates:
        key = (_path_key(root.checkpoints), _path_key(root.loras))
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return tuple(unique)


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
        )

    same_named_installations = tuple(
        installation
        for installation in discovery.installations
        if installation.is_local
        and installation.name.casefold() == selected.name.casefold()
    )
    matched_hosts = (
        tuple(
            hint.host
            for hint in discovery.runtime_hints
            if hint.installation_name.casefold() == selected.name.casefold()
        )
        if len(same_named_installations) == 1
        else ()
    )
    warnings = discovery.warnings
    if len(same_named_installations) > 1:
        warnings = (
            *warnings,
            (
                f'Multiple Comfy Desktop installations are named "{selected.name}". '
                "Their running ports cannot be distinguished safely; rename one "
                "installation in Comfy Desktop."
            ),
        )
    # A selected Desktop installation is an identity boundary.  Falling back
    # to another installation's live port can train into and validate against
    # different model libraries without any obvious error for a casual user.
    hosts = _deduplicate_hosts(matched_hosts)
    input_dir = discovery.input_dir
    output_dir = discovery.output_dir
    if not selected.use_shared_input_output:
        input_dir = selected.input_dir or (
            selected.comfyui_path / "input"
            if selected.comfyui_path is not None
            else input_dir
        )
        output_dir = selected.output_dir or (
            selected.comfyui_path / "output"
            if selected.comfyui_path is not None
            else output_dir
        )
    return replace(
        discovery,
        selected_installation=selected,
        model_roots=_effective_model_roots(
            selected,
            discovery.shared_model_roots,
            discovery.legacy_model_roots,
        ),
        input_dir=input_dir,
        output_dir=output_dir,
        host_candidates=hosts,
        warnings=warnings,
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


def discover_comfy_desktop(appdata: Path | None = None) -> ComfyDesktopDiscovery:
    """Discover Comfy Desktop state without modifying it.

    ``appdata`` is injectable for tests and portable tools.  When omitted, the
    Windows ``APPDATA`` environment variable is used.  On platforms without
    that variable discovery safely returns an empty snapshot.
    """

    warnings: list[str] = []
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

    desktop_dir = roaming / "Comfy Desktop"
    legacy_dir = roaming / "ComfyUI"
    desktop_files_found = any(
        _path_is_file(desktop_dir / name)
        for name in ("settings.json", "installations.json", "last-session.json")
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

    shared_roots = _make_model_roots(
        (path, "Comfy Desktop settings")
        for path in _model_paths_from_settings(settings, desktop_dir)
    )

    legacy_base = _configured_path(
        legacy.get("basePath") or legacy.get("base_path"), legacy_dir
    )
    legacy_roots = (
        _make_model_roots(((legacy_base / "models", "Legacy ComfyUI Desktop"),))
        if legacy_base is not None
        else ()
    )
    effective_roots = _effective_model_roots(selected, shared_roots, legacy_roots)

    input_dir = _configured_path(settings.get("inputDir"), desktop_dir)
    output_dir = _configured_path(settings.get("outputDir"), desktop_dir)
    if selected is not None and not selected.use_shared_input_output:
        input_dir = selected.input_dir or (
            selected.comfyui_path / "input"
            if selected.comfyui_path is not None
            else input_dir
        )
        output_dir = selected.output_dir or (
            selected.comfyui_path / "output"
            if selected.comfyui_path is not None
            else output_dir
        )
    if legacy_base is not None:
        input_dir = input_dir or _configured_path(
            legacy.get("inputDirectory") or legacy.get("inputDir"), legacy_dir
        )
        output_dir = output_dir or _configured_path(
            legacy.get("outputDirectory") or legacy.get("outputDir"), legacy_dir
        )
        input_dir = input_dir or legacy_base / "input"
        output_dir = output_dir or legacy_base / "output"

    runtime_hints = _port_lock_hints(desktop_dir, warnings)
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
        model_roots=effective_roots,
        input_dir=input_dir,
        output_dir=output_dir,
        host_candidates=_deduplicate_hosts(saved_hosts),
        launcher=_launcher(),
        warnings=tuple(warnings),
        runtime_hints=runtime_hints,
        shared_model_roots=shared_roots,
        legacy_model_roots=legacy_roots,
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
