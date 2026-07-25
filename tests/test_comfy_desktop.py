from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Self
from urllib.error import URLError

import pytest

from src import comfy_desktop


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_missing_desktop_settings_use_recorded_default_shared_storage(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "AppData" / "Roaming"
    desktop = appdata / "Comfy Desktop"
    home = tmp_path / "User"
    local_appdata = tmp_path / "AppData" / "Local"
    install = tmp_path / "install"
    (install / "ComfyUI").mkdir(parents=True)
    _write_json(desktop / "settings.json", {})
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "local",
                "name": "Local",
                "sourceId": "standalone",
                "installPath": str(install),
            }
        ],
    )
    _write_json(desktop / "data-location.json", {"mode": "local-appdata"})

    result = comfy_desktop.discover_comfy_desktop(
        appdata,
        desktop_executable=Path(r"C:\Program Files\Comfy Desktop\Comfy Desktop.exe"),
        home_dir=home,
        local_appdata=local_appdata,
        system_drive="C:",
    )

    shared = local_appdata / "Comfy-Desktop" / "ComfyUI-Shared"
    assert result.shared_model_roots[0].path == shared / "models"
    assert result.checkpoint_dirs[0] == shared / "models" / "checkpoints"
    assert result.lora_dirs[0] == shared / "models" / "loras"
    assert result.input_dir == shared / "input"
    assert result.output_dir == shared / "output"


def test_legacy_home_data_location_marker_restores_desktop_defaults(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "AppData" / "Roaming"
    desktop = appdata / "Comfy Desktop"
    home = tmp_path / "User"
    _write_json(desktop / "settings.json", {})
    _write_json(desktop / "data-location.json", {"mode": "legacy-home"})

    result = comfy_desktop.discover_comfy_desktop(
        appdata,
        desktop_executable=Path(r"C:\Program Files\Comfy Desktop\Comfy Desktop.exe"),
        home_dir=home,
        local_appdata=tmp_path / "AppData" / "Local",
        system_drive="C:",
    )

    shared = home / "ComfyUI-Shared"
    assert result.shared_model_roots[0].path == shared / "models"
    assert result.input_dir == shared / "input"
    assert result.output_dir == shared / "output"


def test_marker_only_desktop_state_is_detected_and_guides_first_run(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "AppData" / "Roaming"
    desktop = appdata / "Comfy Desktop"
    local_appdata = tmp_path / "AppData" / "Local"
    _write_json(desktop / "data-location.json", {"mode": "local-appdata"})

    result = comfy_desktop.discover_comfy_desktop(
        appdata,
        desktop_executable=Path(r"C:\Program Files\Comfy Desktop\Comfy Desktop.exe"),
        home_dir=tmp_path / "User",
        local_appdata=local_appdata,
        system_drive="C:",
    )

    assert result.detected
    assert result.config_dir == desktop
    assert result.shared_model_roots[0].path == (
        local_appdata / "Comfy-Desktop" / "ComfyUI-Shared" / "models"
    )


def test_registered_desktop_is_detected_before_its_first_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "Programs" / "Comfy Desktop" / "Comfy Desktop.exe"
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    monkeypatch.setenv("SystemDrive", "C:")
    monkeypatch.setattr(
        comfy_desktop,
        "_registered_comfy_desktop_executable",
        lambda _warnings: executable,
    )

    result = comfy_desktop.discover_comfy_desktop()

    assert result.detected
    assert result.desktop_executable == executable
    assert result.config_dir is None
    assert result.installations == ()


def test_invalid_data_location_marker_uses_desktop_home_footprint_fallback(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "AppData" / "Roaming"
    desktop = appdata / "Comfy Desktop"
    desktop.mkdir(parents=True)
    (desktop / "data-location.json").write_text("{not-json", encoding="utf-8")
    home = tmp_path / "User"
    (home / "ComfyUI-Shared").mkdir(parents=True)

    result = comfy_desktop.discover_comfy_desktop(
        appdata,
        desktop_executable=Path(r"C:\Program Files\Comfy Desktop\Comfy Desktop.exe"),
        home_dir=home,
        local_appdata=tmp_path / "AppData" / "Local",
        system_drive="C:",
    )

    assert result.shared_model_roots[0].path == home / "ComfyUI-Shared" / "models"
    assert any(
        "reconstructed Desktop's safe system-drive default" in warning
        for warning in result.warnings
    )


def test_missing_custom_storage_falls_back_to_default_and_preserves_choice(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "AppData" / "Roaming"
    desktop = appdata / "Comfy Desktop"
    local_appdata = tmp_path / "AppData" / "Local"
    missing_models = tmp_path / "offline-model-library"
    _write_json(
        desktop / "settings.json",
        {
            "modelsDirs": [str(missing_models)],
            "inputDir": str(tmp_path / "offline-input"),
            "outputDir": str(tmp_path / "offline-output"),
        },
    )
    _write_json(desktop / "data-location.json", {"mode": "local-appdata"})

    result = comfy_desktop.discover_comfy_desktop(
        appdata,
        desktop_executable=Path(r"C:\Program Files\Comfy Desktop\Comfy Desktop.exe"),
        home_dir=tmp_path / "User",
        local_appdata=local_appdata,
        system_drive="C:",
    )

    shared = local_appdata / "Comfy-Desktop" / "ComfyUI-Shared"
    assert tuple(root.path for root in result.shared_model_roots) == (
        shared / "models",
        missing_models,
    )
    assert result.input_dir == shared / "input"
    assert result.output_dir == shared / "output"


def test_ambiguous_external_installation_fails_closed_without_path_claims(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "AppData" / "Roaming"
    desktop = appdata / "Comfy Desktop"
    _write_json(desktop / "settings.json", {})
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "external",
                "name": "External",
                "sourceId": "standalone",
                "installPath": r"D:\ModelsAndApps\Comfy",
            }
        ],
    )

    result = comfy_desktop.discover_comfy_desktop(
        appdata,
        desktop_executable=Path("Comfy Desktop.exe"),
        home_dir=tmp_path / "User",
        local_appdata=tmp_path / "AppData" / "Local",
        system_drive="C:",
    )

    assert result.checkpoint_dirs == ()
    assert result.lora_dirs == ()
    assert result.input_dir is None
    assert result.output_dir is None
    assert any("could not be proven" in warning for warning in result.warnings)


def test_redirected_default_installation_attests_shared_drive_layout(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "AppData" / "Roaming"
    desktop = appdata / "Comfy Desktop"
    _write_json(desktop / "settings.json", {})
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "redirected",
                "name": "Redirected",
                "sourceId": "standalone",
                "installPath": (
                    r"D:\Comfy-Desktop\ComfyUI-Installs\redirected-instance"
                ),
            }
        ],
    )
    _write_json(desktop / "data-location.json", {"mode": "local-appdata"})

    result = comfy_desktop.discover_comfy_desktop(
        appdata,
        desktop_executable=Path(r"D:\Programs\Comfy Desktop\Comfy Desktop.exe"),
        home_dir=tmp_path / "User",
        local_appdata=tmp_path / "AppData" / "Local",
        system_drive="C:",
    )

    shared = Path(r"D:\Comfy-Desktop") / "ComfyUI-Shared"
    assert result.shared_model_roots[0].path == shared / "models"
    assert result.input_dir == shared / "input"
    assert result.output_dir == shared / "output"


def test_discovers_desktop_2_selected_install_and_shared_models(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "Roaming"
    desktop = appdata / "Comfy Desktop"
    shared_models = tmp_path / "ComfyUI-Shared" / "models"
    (shared_models / "loras").mkdir(parents=True)
    (shared_models / "checkpoints").mkdir()
    local_install = tmp_path / "ComfyUI-Installs" / "local"
    (local_install / "ComfyUI").mkdir(parents=True)

    _write_json(
        desktop / "settings.json",
        {
            "modelsDirs": [str(shared_models)],
            "inputDir": str(tmp_path / "ComfyUI-Shared" / "input"),
            "outputDir": str(tmp_path / "ComfyUI-Shared" / "output"),
            "server": {"host": "127.0.0.1", "port": 8765},
        },
    )
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "local-id",
                "name": "My Comfy",
                "sourceId": "standalone",
                "installPath": str(local_install),
                "status": "installed",
                "comfyVersionTag": "v0.28.2",
                "lastLaunchedAt": 20,
            },
            {
                "id": "cloud-id",
                "name": "Comfy Cloud",
                "sourceId": "cloud",
                "remoteUrl": "https://cloud.comfy.org/",
                "lastLaunchedAt": 30,
            },
        ],
    )
    _write_json(
        desktop / "last-session.json",
        {"kind": "instance", "installationId": "local-id"},
    )

    result = comfy_desktop.discover_comfy_desktop(appdata)

    assert result.detected
    assert result.config_dir == desktop
    assert result.selected_installation is not None
    assert result.selected_installation.id == "local-id"
    assert result.selected_installation.name == "My Comfy"
    assert result.selected_installation.comfyui_path == local_install / "ComfyUI"
    assert result.selected_installation.version == "v0.28.2"
    assert len(result.installations) == 2
    assert result.model_roots == (
        comfy_desktop.ComfyModelRoot(
            path=shared_models,
            loras=shared_models / "loras",
            checkpoints=shared_models / "checkpoints",
            source="Comfy Desktop settings",
            exists=True,
        ),
        comfy_desktop.ComfyModelRoot(
            path=local_install / "ComfyUI" / "models",
            loras=local_install / "ComfyUI" / "models" / "loras",
            checkpoints=local_install / "ComfyUI" / "models" / "checkpoints",
            source="My Comfy",
            exists=False,
        ),
    )
    assert result.preferred_model_root == result.model_roots[0]
    assert result.checkpoint_dirs == (
        shared_models / "checkpoints",
        local_install / "ComfyUI" / "models" / "checkpoints",
        tmp_path / "ComfyUI-Shared" / "output" / "checkpoints",
    )
    assert result.lora_dirs == (
        shared_models / "loras",
        local_install / "ComfyUI" / "models" / "loras",
        tmp_path / "ComfyUI-Shared" / "output" / "loras",
    )
    assert result.input_dir == tmp_path / "ComfyUI-Shared" / "input"
    assert result.output_dir == tmp_path / "ComfyUI-Shared" / "output"
    assert result.host_candidates[:4] == (
        "127.0.0.1:8765",
        "127.0.0.1:8188",
        "127.0.0.1:8000",
        "127.0.0.1:8001",
    )
    assert result.launcher.app_id == comfy_desktop.COMFY_DESKTOP_APP_ID
    assert result.launcher.command[0] == "explorer.exe"
    assert result.warnings == ()


def test_falls_back_to_most_recent_local_install_and_ignores_cloud_session(
    tmp_path: Path,
) -> None:
    desktop = tmp_path / "Comfy Desktop"
    _write_json(desktop / "settings.json", {})
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "older",
                "sourceId": "standalone",
                "installPath": str(tmp_path / "older"),
                "lastLaunchedAt": 10,
            },
            {
                "id": "newer",
                "sourceId": "git",
                "installPath": str(tmp_path / "newer"),
                "lastLaunchedAt": 99,
            },
            {
                "id": "cloud",
                "sourceId": "cloud",
                "remoteUrl": "https://cloud.comfy.org",
                "lastLaunchedAt": 100,
            },
        ],
    )
    _write_json(desktop / "last-session.json", {"installationId": "cloud"})

    result = comfy_desktop.discover_comfy_desktop(tmp_path)

    assert result.selected_installation is not None
    assert result.selected_installation.id == "newer"
    assert result.model_roots[0].path == (
        tmp_path / "Local" / "Comfy-Desktop" / "ComfyUI-Shared" / "models"
    )
    assert result.model_roots[1].path == tmp_path / "newer" / "ComfyUI" / "models"


def test_discovers_legacy_base_path_and_saved_port(tmp_path: Path) -> None:
    base = tmp_path / "LegacyData"
    (base / "models").mkdir(parents=True)
    _write_json(
        tmp_path / "ComfyUI" / "config.json",
        {"basePath": str(base), "serverPort": 8123},
    )

    result = comfy_desktop.discover_comfy_desktop(tmp_path)

    assert result.detected
    assert result.config_dir is None
    assert result.legacy_config_dir == tmp_path / "ComfyUI"
    assert result.model_roots[0].path == base / "models"
    assert result.model_roots[0].loras == base / "models" / "loras"
    assert result.input_dir == base / "input"
    assert result.output_dir == base / "output"
    assert result.host_candidates[0] == "127.0.0.1:8123"


def test_current_desktop_port_locks_are_prioritized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(comfy_desktop, "_process_is_running", lambda _pid: True)
    desktop = tmp_path / "Comfy Desktop"
    _write_json(desktop / "settings.json", {})
    _write_json(
        desktop / "port-locks" / "port-8234.json",
        {
            "pid": 1234,
            "installationName": "Local instance",
            "timestamp": 1_725_000_000_000,
        },
    )

    result = comfy_desktop.discover_comfy_desktop(tmp_path)

    assert result.host_candidates[0] == "127.0.0.1:8234"
    assert result.warnings == ()
    assert result.runtime_hints[0].installation_name == "Local instance"


def test_selecting_an_instance_binds_its_runtime_and_private_model_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(comfy_desktop, "_process_is_running", lambda _pid: True)
    appdata = tmp_path / "Roaming"
    desktop = appdata / "Comfy Desktop"
    install_a = tmp_path / "installs" / "a"
    install_b = tmp_path / "installs" / "b"
    private_a = tmp_path / "models-a"
    private_b = tmp_path / "models-b"
    for path in (
        install_a / "ComfyUI",
        install_b / "ComfyUI",
        private_a / "loras",
        private_a / "checkpoints",
        private_b / "loras",
        private_b / "checkpoints",
    ):
        path.mkdir(parents=True)
    _write_json(desktop / "settings.json", {"modelsDirs": []})
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "a",
                "name": "Portrait A",
                "sourceId": "standalone",
                "installPath": str(install_a),
                "useSharedModels": False,
                "useSharedInputOutput": False,
                "modelDirs": [str(private_a)],
            },
            {
                "id": "b",
                "name": "Portrait B",
                "sourceId": "standalone",
                "installPath": str(install_b),
                "useSharedModels": False,
                "useSharedInputOutput": False,
                "modelDirs": [str(private_b)],
                "modelDirsPrimary": str(private_b),
            },
        ],
    )
    for installation_id, model_root in (("a", private_a), ("b", private_b)):
        generated = desktop / "instance-model-paths" / f"{installation_id}.yaml"
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text(
            f"private:\n  base_path: {model_root.as_posix()}\n"
            "  is_default: true\n"
            "  checkpoints: checkpoints\n"
            "  loras: loras\n",
            encoding="utf-8",
        )
    running_timestamp = int(time.time() * 1000) + 5_000
    _write_json(
        desktop / "port-locks" / "port-8201.json",
        {
            "pid": 101,
            "installationName": "Portrait A",
            "timestamp": running_timestamp,
        },
    )
    _write_json(
        desktop / "port-locks" / "port-8202.json",
        {
            "pid": 102,
            "installationName": "Portrait B",
            "timestamp": running_timestamp,
        },
    )

    discovered = comfy_desktop.discover_comfy_desktop(appdata)
    selected = comfy_desktop.select_comfy_installation(discovered, "b")

    assert selected.selected_installation is not None
    assert selected.selected_installation.id == "b"
    assert selected.host_candidates == ("127.0.0.1:8202",)
    assert selected.model_roots[0].path == private_b
    assert selected.model_roots[0].loras == private_b / "loras"
    assert all(root.path != private_a for root in selected.model_roots)
    assert selected.checkpoint_dirs[:2] == (
        private_b / "checkpoints",
        install_b / "ComfyUI" / "models" / "checkpoints",
    )
    assert selected.lora_dirs[:2] == (
        private_b / "loras",
        install_b / "ComfyUI" / "models" / "loras",
    )


def test_duplicate_desktop_names_never_bind_an_ambiguous_runtime_port(
    tmp_path: Path,
) -> None:
    install_a = comfy_desktop.ComfyInstallation(
        id="a",
        name="ComfyUI",
        source_id="standalone",
        install_path=tmp_path / "a",
        comfyui_path=tmp_path / "a" / "ComfyUI",
        version=None,
        status="installed",
        last_launched_at=1,
    )
    install_b = comfy_desktop.ComfyInstallation(
        id="b",
        name="ComfyUI",
        source_id="standalone",
        install_path=tmp_path / "b",
        comfyui_path=tmp_path / "b" / "ComfyUI",
        version=None,
        status="installed",
        last_launched_at=2,
    )
    discovery = comfy_desktop.ComfyDesktopDiscovery(
        config_dir=tmp_path,
        legacy_config_dir=None,
        installations=(install_a, install_b),
        selected_installation=install_a,
        model_roots=(),
        input_dir=None,
        output_dir=None,
        host_candidates=("127.0.0.1:8201",),
        launcher=comfy_desktop.ComfyLauncher("", "", "", (), False),
        warnings=(),
        runtime_hints=(
            comfy_desktop.ComfyRuntimeHint(
                "127.0.0.1:8201",
                "ComfyUI",
                101,
                1000,
            ),
        ),
    )

    selected = comfy_desktop.select_comfy_installation(discovery, "b")

    assert selected.selected_installation == install_b
    assert selected.host_candidates == ()
    assert any("cannot be distinguished safely" in item for item in selected.warnings)


def test_unknown_desktop_id_never_falls_back_to_the_current_installation(
    tmp_path: Path,
) -> None:
    installation = comfy_desktop.ComfyInstallation(
        id="current",
        name="Current",
        source_id="standalone",
        install_path=tmp_path / "current",
        comfyui_path=tmp_path / "current" / "ComfyUI",
        version=None,
        status="installed",
        last_launched_at=1,
    )
    discovery = comfy_desktop.ComfyDesktopDiscovery(
        config_dir=tmp_path,
        legacy_config_dir=None,
        installations=(installation,),
        selected_installation=installation,
        model_roots=(),
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        host_candidates=("127.0.0.1:8188",),
        launcher=comfy_desktop.ComfyLauncher("", "", "", (), False),
        warnings=(),
    )

    selected = comfy_desktop.select_comfy_installation(discovery, "missing")

    assert selected.selected_installation is None
    assert selected.host_candidates == ()
    assert selected.model_roots == ()
    assert selected.input_dir is None
    assert selected.output_dir is None


def test_incomplete_desktop_port_lock_is_ignored(tmp_path: Path) -> None:
    desktop = tmp_path / "Comfy Desktop"
    _write_json(desktop / "settings.json", {})
    _write_json(desktop / "port-locks" / "port-8234.json", {"pid": 1234})

    result = comfy_desktop.discover_comfy_desktop(tmp_path)

    assert "127.0.0.1:8234" not in result.host_candidates
    assert any("incomplete" in warning for warning in result.warnings)


def test_stale_desktop_port_lock_is_never_a_runtime_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(comfy_desktop, "_process_is_running", lambda _pid: False)
    desktop = tmp_path / "Comfy Desktop"
    _write_json(desktop / "settings.json", {})
    _write_json(
        desktop / "port-locks" / "port-8234.json",
        {
            "pid": 1234,
            "installationName": "Closed instance",
            "timestamp": 1_725_000_000_000,
        },
    )

    result = comfy_desktop.discover_comfy_desktop(tmp_path)

    assert result.runtime_hints == ()
    assert "127.0.0.1:8234" not in result.host_candidates
    assert any("stale Comfy port lock" in warning for warning in result.warnings)


def test_discovers_current_extra_model_paths_for_selected_instance(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "Roaming"
    desktop = appdata / "Comfy Desktop"
    install = tmp_path / "install"
    comfyui = install / "ComfyUI"
    extra = tmp_path / "external-models"
    (comfyui / "models").mkdir(parents=True)
    (extra / "checkpoints").mkdir(parents=True)
    (extra / "loras").mkdir()
    _write_json(desktop / "settings.json", {"modelsDirs": []})
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "local",
                "name": "Local",
                "sourceId": "standalone",
                "installPath": str(install),
                "useSharedModels": False,
                "useSharedInputOutput": False,
            }
        ],
    )
    (comfyui / "extra_model_paths.yaml").write_text(
        "external:\n"
        f"  base_path: {extra.as_posix()}\n"
        "  checkpoints: checkpoints\n"
        "  loras: loras\n",
        encoding="utf-8",
    )

    result = comfy_desktop.discover_comfy_desktop(appdata)

    extra_root = next(
        root for root in result.model_roots if "extra_model_paths" in root.source
    )
    assert extra_root.checkpoints == extra / "checkpoints"
    assert extra_root.loras == extra / "loras"


def test_discovers_independent_multiline_extra_model_categories(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "Roaming"
    desktop = appdata / "Comfy Desktop"
    install = tmp_path / "install"
    comfyui = install / "ComfyUI"
    own_models = comfyui / "models"
    extra = tmp_path / "external-models"
    own_models.mkdir(parents=True)
    for name in (
        "checkpoints-a",
        "checkpoints-b",
        "loras-a",
        "loras-b",
    ):
        (extra / name).mkdir(parents=True)
    _write_json(desktop / "settings.json", {"modelsDirs": []})
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "local",
                "name": "Local",
                "sourceId": "standalone",
                "installPath": str(install),
                "useSharedModels": False,
                "useSharedInputOutput": False,
            }
        ],
    )
    (comfyui / "extra_model_paths.yaml").write_text(
        "checkpoint_only:\n"
        f"  base_path: {extra.as_posix()}\n"
        "  checkpoints: |-\n"
        "    checkpoints-a\n"
        "    checkpoints-b\n"
        "lora_only:\n"
        f"  base_path: {extra.as_posix()}\n"
        "  loras: |-\n"
        "    loras-a\n"
        "    loras-b\n",
        encoding="utf-8",
    )

    result = comfy_desktop.discover_comfy_desktop(appdata)

    assert result.checkpoint_dirs == (
        own_models / "checkpoints",
        extra / "checkpoints-a",
        extra / "checkpoints-b",
        comfyui / "output" / "checkpoints",
    )
    assert result.lora_dirs == (
        own_models / "loras",
        extra / "loras-a",
        extra / "loras-b",
        comfyui / "output" / "loras",
    )
    assert not any("extra_model_paths" in root.source for root in result.model_roots)


def test_extra_model_is_default_uses_comfyui_prepend_order_and_deduplicates(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "Roaming"
    desktop = appdata / "Comfy Desktop"
    install = tmp_path / "install"
    comfyui = install / "ComfyUI"
    own_models = comfyui / "models"
    extra = tmp_path / "external-models"
    own_models.mkdir(parents=True)
    for name in ("ordinary", "preferred-a", "preferred-b"):
        (extra / name).mkdir(parents=True)
    _write_json(desktop / "settings.json", {"modelsDirs": []})
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "local",
                "name": "Local",
                "sourceId": "standalone",
                "installPath": str(install),
                "useSharedModels": False,
            }
        ],
    )
    (comfyui / "extra_model_paths.yaml").write_text(
        "ordinary:\n"
        f"  base_path: {extra.as_posix()}\n"
        "  checkpoints: ordinary\n"
        "  loras: ordinary\n"
        "preferred:\n"
        f"  base_path: {extra.as_posix()}\n"
        "  is_default: true\n"
        "  checkpoints: |-\n"
        "    preferred-a\n"
        "    preferred-b\n"
        "  loras: |-\n"
        "    preferred-a\n"
        "    preferred-b\n"
        "promote_existing:\n"
        f"  base_path: {extra.as_posix()}\n"
        "  is_default: true\n"
        "  checkpoints: ordinary\n"
        "  loras: ordinary\n",
        encoding="utf-8",
    )

    result = comfy_desktop.discover_comfy_desktop(appdata)

    # ComfyUI inserts each default path at index zero as it is encountered.
    assert result.checkpoint_dirs == (
        extra / "ordinary",
        extra / "preferred-b",
        extra / "preferred-a",
        own_models / "checkpoints",
        (
            tmp_path
            / "Local"
            / "Comfy-Desktop"
            / "ComfyUI-Shared"
            / "output"
            / "checkpoints"
        ),
    )
    assert result.lora_dirs == (
        extra / "ordinary",
        extra / "preferred-b",
        extra / "preferred-a",
        own_models / "loras",
        (tmp_path / "Local" / "Comfy-Desktop" / "ComfyUI-Shared" / "output" / "loras"),
    )


def test_non_boolean_is_default_matches_comfyui_runtime_truthiness(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "Roaming"
    desktop = appdata / "Comfy Desktop"
    install = tmp_path / "install"
    comfyui = install / "ComfyUI"
    own_models = comfyui / "models"
    extra = tmp_path / "extra"
    own_models.mkdir(parents=True)
    _write_json(desktop / "settings.json", {})
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "local",
                "name": "Local",
                "sourceId": "standalone",
                "installPath": str(install),
                "useSharedModels": False,
                "useSharedInputOutput": False,
            }
        ],
    )
    (comfyui / "extra_model_paths.yaml").write_text(
        "ordinary:\n"
        f"  base_path: {extra.as_posix()}\n"
        "  checkpoints: ordinary\n"
        "  loras: ordinary\n"
        "quoted_value:\n"
        f"  base_path: {extra.as_posix()}\n"
        '  is_default: "false"\n'
        "  checkpoints: quoted\n"
        "  loras: quoted\n",
        encoding="utf-8",
    )

    result = comfy_desktop.discover_comfy_desktop(
        appdata,
        desktop_executable=Path(r"C:\Program Files\Comfy Desktop\Comfy Desktop.exe"),
        system_drive="C:",
    )

    assert result.checkpoint_dirs == (
        extra / "quoted",
        own_models / "checkpoints",
        extra / "ordinary",
        comfyui / "output" / "checkpoints",
    )
    assert result.lora_dirs == (
        extra / "quoted",
        own_models / "loras",
        extra / "ordinary",
        comfyui / "output" / "loras",
    )
    assert any("runtime truthiness" in warning for warning in result.warnings)


def test_stale_desktop_generated_model_paths_never_override_current_settings(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "Roaming"
    desktop = appdata / "Comfy Desktop"
    install = tmp_path / "install"
    comfyui = install / "ComfyUI"
    current_models = tmp_path / "current-models"
    stale_models = tmp_path / "stale-models"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    for directory in (
        comfyui / "models",
        current_models / "checkpoints",
        current_models / "loras",
        input_dir,
        output_dir,
    ):
        directory.mkdir(parents=True)
    _write_json(
        desktop / "settings.json",
        {
            "modelsDirs": [str(current_models)],
            "inputDir": str(input_dir),
            "outputDir": str(output_dir),
        },
    )
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "local",
                "name": "Local",
                "sourceId": "standalone",
                "installPath": str(install),
            }
        ],
    )
    (desktop / "shared_model_paths.yaml").write_text(
        "stale:\n"
        f"  base_path: {stale_models.as_posix()}\n"
        "  is_default: true\n"
        "  checkpoints: checkpoints\n"
        "  loras: loras\n",
        encoding="utf-8",
    )

    result = comfy_desktop.discover_comfy_desktop(
        appdata,
        desktop_executable=Path(r"C:\Program Files\Comfy Desktop\Comfy Desktop.exe"),
        system_drive="C:",
    )

    assert result.checkpoint_dirs[0] == current_models / "checkpoints"
    assert result.lora_dirs[0] == current_models / "loras"
    assert stale_models / "checkpoints" not in result.checkpoint_dirs
    assert stale_models / "loras" not in result.lora_dirs
    assert any(
        "Ignored stale Comfy Desktop model-path file" in warning
        for warning in result.warnings
    )

    running_timestamp = int(time.time() * 1000) + 5_000
    _write_json(
        desktop / "port-locks" / "port-8188.json",
        {
            "pid": os.getpid(),
            "installationName": "Local",
            "timestamp": running_timestamp,
        },
    )

    running = comfy_desktop.discover_comfy_desktop(
        appdata,
        desktop_executable=Path(r"C:\Program Files\Comfy Desktop\Comfy Desktop.exe"),
        system_drive="C:",
    )

    assert running.checkpoint_dirs[0] == stale_models / "checkpoints"
    assert running.lora_dirs[0] == stale_models / "loras"
    assert current_models / "checkpoints" not in running.checkpoint_dirs
    assert current_models / "loras" not in running.lora_dirs
    assert any("launch-time model folders" in warning for warning in running.warnings)


def test_running_instance_requires_restart_after_desktop_storage_change(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "Roaming"
    desktop = appdata / "Comfy Desktop"
    install = tmp_path / "install"
    comfyui = install / "ComfyUI"
    models = tmp_path / "models"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    for directory in (
        comfyui / "models",
        models / "checkpoints",
        models / "loras",
        input_dir,
        output_dir,
    ):
        directory.mkdir(parents=True)
    _write_json(
        desktop / "settings.json",
        {
            "modelsDirs": [str(models)],
            "inputDir": str(input_dir),
            "outputDir": str(output_dir),
        },
    )
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "local",
                "name": "Local",
                "sourceId": "standalone",
                "installPath": str(install),
            }
        ],
    )
    (desktop / "shared_model_paths.yaml").write_text(
        "current:\n"
        f"  base_path: {models.as_posix()}\n"
        "  is_default: true\n"
        "  checkpoints: checkpoints\n"
        "  loras: loras\n",
        encoding="utf-8",
    )
    running_timestamp = int(time.time() * 1000)
    _write_json(
        desktop / "port-locks" / "port-8188.json",
        {
            "pid": os.getpid(),
            "installationName": "Local",
            "timestamp": running_timestamp,
        },
    )
    settings_path = desktop / "settings.json"
    changed_ns = (running_timestamp + 2_000) * 1_000_000
    os.utime(settings_path, ns=(changed_ns, changed_ns))

    result = comfy_desktop.discover_comfy_desktop(
        appdata,
        desktop_executable=Path(r"C:\Program Files\Comfy Desktop\Comfy Desktop.exe"),
        system_drive="C:",
    )

    assert result.checkpoint_dirs == ()
    assert result.lora_dirs == ()
    assert result.input_dir is None
    assert result.output_dir is None
    assert any(
        "storage settings changed after" in warning for warning in result.warnings
    )


def test_desktop_generated_config_applies_after_local_yaml_and_before_output(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "Roaming"
    desktop = appdata / "Comfy Desktop"
    install = tmp_path / "install"
    comfyui = install / "ComfyUI"
    own_models = comfyui / "models"
    local_extra = tmp_path / "local-extra"
    shared_a = tmp_path / "shared-a"
    shared_b = tmp_path / "shared-b"
    output = tmp_path / "output"
    for root in (own_models, local_extra, shared_a, shared_b):
        (root / "checkpoints").mkdir(parents=True)
        (root / "loras").mkdir()
        (root / "checkpoints" / "same.safetensors").write_bytes(str(root).encode())
    output.mkdir()
    _write_json(
        desktop / "settings.json",
        {
            "modelsDirs": [str(shared_a), str(shared_b)],
            "outputDir": str(output),
        },
    )
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "local",
                "name": "Local",
                "sourceId": "standalone",
                "installPath": str(install),
                "useSharedModels": True,
            }
        ],
    )
    (comfyui / "extra_model_paths.yaml").write_text(
        "local:\n"
        f"  base_path: {local_extra.as_posix()}\n"
        "  is_default: true\n"
        "  checkpoints: checkpoints\n"
        "  loras: loras\n",
        encoding="utf-8",
    )
    desktop.mkdir(parents=True, exist_ok=True)
    (desktop / "shared_model_paths.yaml").write_text(
        "shared_0:\n"
        f"  base_path: {shared_a.as_posix()}\n"
        "  is_default: true\n"
        "  checkpoints: checkpoints\n"
        "  loras: loras\n"
        "shared_1:\n"
        f"  base_path: {shared_b.as_posix()}\n"
        "  checkpoints: checkpoints\n"
        "  loras: loras\n",
        encoding="utf-8",
    )

    result = comfy_desktop.discover_comfy_desktop(appdata)

    assert result.effective_checkpoint_dirs == (
        shared_a / "checkpoints",
        local_extra / "checkpoints",
        own_models / "checkpoints",
        shared_b / "checkpoints",
        output / "checkpoints",
    )
    assert result.effective_lora_dirs == (
        shared_a / "loras",
        local_extra / "loras",
        own_models / "loras",
        shared_b / "loras",
        output / "loras",
    )
    assert result.checkpoint_candidates("same.safetensors") == (
        shared_a / "checkpoints" / "same.safetensors",
        local_extra / "checkpoints" / "same.safetensors",
        own_models / "checkpoints" / "same.safetensors",
        shared_b / "checkpoints" / "same.safetensors",
    )


def test_startup_path_arguments_match_comfyui_resolution_order(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "Roaming"
    desktop = appdata / "Comfy Desktop"
    install = tmp_path / "install"
    comfyui = install / "ComfyUI"
    comfyui.mkdir(parents=True)
    startup_models = install / "startup models"
    explicit_models = install / "explicit models"
    for root in (startup_models, explicit_models):
        (root / "checkpoints").mkdir(parents=True)
        (root / "loras").mkdir()
    startup_config = install / "startup paths.yaml"
    startup_config.write_text(
        "startup:\n"
        f"  base_path: {startup_models.as_posix()}\n"
        "  is_default: true\n"
        "  checkpoints: checkpoints\n"
        "  loras: loras\n",
        encoding="utf-8",
    )
    _write_json(desktop / "settings.json", {"modelsDirs": []})
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "local",
                "name": "Local",
                "sourceId": "standalone",
                "installPath": str(install),
                "useSharedModels": False,
                "useSharedInputOutput": False,
                "launchArgs": (
                    '--base-directory "ignored base" '
                    '--models-directory "explicit models" '
                    '--extra-model-paths-config "startup paths.yaml" '
                    '--output-directory "custom output"'
                ),
            }
        ],
    )

    result = comfy_desktop.discover_comfy_desktop(appdata)

    assert result.effective_checkpoint_dirs == (
        startup_models / "checkpoints",
        explicit_models / "checkpoints",
        install / "custom output" / "checkpoints",
    )
    assert result.effective_lora_dirs == (
        startup_models / "loras",
        explicit_models / "loras",
        install / "custom output" / "loras",
    )
    assert result.output_dir == install / "custom output"
    assert result.selected_installation is not None
    assert result.selected_installation.launch_args.startswith("--base-directory")


def test_adopted_base_and_shared_output_follow_desktop_precedence(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "Roaming"
    desktop = appdata / "Comfy Desktop"
    install = tmp_path / "managed-wrapper"
    (install / "ComfyUI").mkdir(parents=True)
    adopted_base = tmp_path / "legacy-data"
    shared_output = tmp_path / "shared-output"
    shared_output.mkdir()
    _write_json(
        desktop / "settings.json",
        {
            "modelsDirs": [],
            "outputDir": str(shared_output),
        },
    )
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "adopted",
                "name": "Imported Comfy",
                "sourceId": "standalone",
                "installPath": str(install),
                "adopted": True,
                "adoptedBaseDir": str(adopted_base),
                "useSharedModels": False,
                "useSharedInputOutput": True,
                "launchArgs": '--output-directory "ignored-output"',
            }
        ],
    )

    result = comfy_desktop.discover_comfy_desktop(appdata)

    assert result.effective_checkpoint_dirs == (
        adopted_base / "models" / "checkpoints",
        shared_output / "checkpoints",
    )
    assert result.effective_lora_dirs == (
        adopted_base / "models" / "loras",
        shared_output / "loras",
    )
    assert result.output_dir == shared_output


def test_user_base_directory_overrides_adopted_base(tmp_path: Path) -> None:
    appdata = tmp_path / "Roaming"
    desktop = appdata / "Comfy Desktop"
    install = tmp_path / "managed-wrapper"
    (install / "ComfyUI").mkdir(parents=True)
    adopted_base = tmp_path / "legacy-data"
    override_base = install / "custom base"
    _write_json(desktop / "settings.json", {"modelsDirs": []})
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "adopted",
                "name": "Imported Comfy",
                "sourceId": "standalone",
                "installPath": str(install),
                "adopted": True,
                "adoptedBaseDir": str(adopted_base),
                "useSharedModels": False,
                "useSharedInputOutput": False,
                "launchArgs": '--base-directory "custom base"',
            }
        ],
    )

    result = comfy_desktop.discover_comfy_desktop(appdata)

    assert result.effective_checkpoint_dirs == (
        override_base / "models" / "checkpoints",
        override_base / "output" / "checkpoints",
    )
    assert result.input_dir == override_base / "input"
    assert result.output_dir == override_base / "output"


def test_adopted_base_is_ignored_without_adopted_flag(tmp_path: Path) -> None:
    appdata = tmp_path / "Roaming"
    desktop = appdata / "Comfy Desktop"
    install = tmp_path / "managed"
    comfyui = install / "ComfyUI"
    comfyui.mkdir(parents=True)
    _write_json(desktop / "settings.json", {"modelsDirs": []})
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "ordinary",
                "name": "Ordinary Comfy",
                "sourceId": "standalone",
                "installPath": str(install),
                "adoptedBaseDir": str(tmp_path / "must-not-be-used"),
                "useSharedModels": False,
                "useSharedInputOutput": False,
            }
        ],
    )

    result = comfy_desktop.discover_comfy_desktop(appdata)

    assert result.effective_checkpoint_dirs == (
        comfyui / "models" / "checkpoints",
        comfyui / "output" / "checkpoints",
    )


def test_repeated_and_multi_value_startup_configs_preserve_argument_order(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "Roaming"
    desktop = appdata / "Comfy Desktop"
    install = tmp_path / "install"
    comfyui = install / "ComfyUI"
    own_models = comfyui / "models"
    own_models.mkdir(parents=True)
    roots = [tmp_path / f"extra-{index}" for index in range(3)]
    for index, root in enumerate(roots):
        (root / "checkpoints").mkdir(parents=True)
        (install / f"paths-{index}.yaml").write_text(
            f"extra_{index}:\n"
            f"  base_path: {root.as_posix()}\n"
            f"  is_default: {'true' if index == 2 else 'false'}\n"
            "  checkpoints: checkpoints\n",
            encoding="utf-8",
        )
    _write_json(desktop / "settings.json", {"modelsDirs": []})
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "local",
                "name": "Local",
                "sourceId": "standalone",
                "installPath": str(install),
                "useSharedModels": False,
                "useSharedInputOutput": False,
                "launchArgs": (
                    "--extra-model-paths-config=paths-0.yaml paths-1.yaml "
                    "--extra-model-paths-config paths-2.yaml"
                ),
            }
        ],
    )

    result = comfy_desktop.discover_comfy_desktop(appdata)

    assert result.effective_checkpoint_dirs == (
        roots[2] / "checkpoints",
        own_models / "checkpoints",
        roots[0] / "checkpoints",
        roots[1] / "checkpoints",
        comfyui / "output" / "checkpoints",
    )


def test_missing_startup_model_config_is_explained(tmp_path: Path) -> None:
    appdata = tmp_path / "Roaming"
    desktop = appdata / "Comfy Desktop"
    install = tmp_path / "install"
    (install / "ComfyUI").mkdir(parents=True)
    _write_json(desktop / "settings.json", {"modelsDirs": []})
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "local",
                "name": "Local",
                "sourceId": "standalone",
                "installPath": str(install),
                "useSharedModels": False,
                "launchArgs": "--extra-model-paths-config missing.yaml",
            }
        ],
    )

    result = comfy_desktop.discover_comfy_desktop(appdata)

    assert any(
        "starts with a missing extra model-path configuration" in warning
        for warning in result.warnings
    )


def test_legacy_extra_models_config_is_part_of_effective_layout(
    tmp_path: Path,
) -> None:
    legacy_dir = tmp_path / "ComfyUI"
    base = tmp_path / "legacy-data"
    extra = tmp_path / "external"
    (base / "models").mkdir(parents=True)
    (extra / "checkpoints").mkdir(parents=True)
    (extra / "loras").mkdir()
    _write_json(legacy_dir / "config.json", {"basePath": str(base)})
    (legacy_dir / "extra_models_config.yaml").write_text(
        "external:\n"
        f"  base_path: {extra.as_posix()}\n"
        "  is_default: true\n"
        "  checkpoints: checkpoints\n"
        "  loras: loras\n",
        encoding="utf-8",
    )

    result = comfy_desktop.discover_comfy_desktop(tmp_path)

    assert result.effective_checkpoint_dirs == (
        extra / "checkpoints",
        base / "models" / "checkpoints",
        base / "output" / "checkpoints",
    )
    assert result.effective_lora_dirs == (
        extra / "loras",
        base / "models" / "loras",
        base / "output" / "loras",
    )


def test_non_text_extra_model_path_is_ignored_with_a_plain_warning(
    tmp_path: Path,
) -> None:
    appdata = tmp_path / "Roaming"
    desktop = appdata / "Comfy Desktop"
    install = tmp_path / "install"
    comfyui = install / "ComfyUI"
    (comfyui / "models").mkdir(parents=True)
    _write_json(desktop / "settings.json", {"modelsDirs": []})
    _write_json(
        desktop / "installations.json",
        [
            {
                "id": "local",
                "name": "Local",
                "sourceId": "standalone",
                "installPath": str(install),
                "useSharedModels": False,
                "useSharedInputOutput": False,
            }
        ],
    )
    (comfyui / "extra_model_paths.yaml").write_text(
        "invalid:\n"
        f"  base_path: {tmp_path.as_posix()}\n"
        "  checkpoints:\n"
        "    - first\n"
        "    - second\n",
        encoding="utf-8",
    )

    result = comfy_desktop.discover_comfy_desktop(appdata)

    assert result.effective_checkpoint_dirs == (
        comfyui / "models" / "checkpoints",
        comfyui / "output" / "checkpoints",
    )
    assert any(
        "accepts one path or a multiline text block" in warning
        for warning in result.warnings
    )


def test_model_candidate_helpers_preserve_comfyui_duplicate_resolution_order(
    tmp_path: Path,
) -> None:
    preferred = tmp_path / "preferred"
    fallback = tmp_path / "fallback"
    for directory, contents in ((preferred, b"preferred"), (fallback, b"fallback")):
        directory.mkdir()
        (directory / "portrait.safetensors").write_bytes(contents)
        (directory / "nested").mkdir()
        (directory / "nested" / "style.safetensors").write_bytes(contents)
    discovery = comfy_desktop.ComfyDesktopDiscovery(
        config_dir=None,
        legacy_config_dir=None,
        installations=(),
        selected_installation=None,
        model_roots=(),
        input_dir=None,
        output_dir=None,
        host_candidates=(),
        launcher=comfy_desktop.ComfyLauncher("", "", "", (), False),
        warnings=(),
        checkpoint_dirs=(preferred, fallback),
        lora_dirs=(fallback, preferred),
    )

    assert discovery.checkpoint_candidates("portrait.safetensors") == (
        preferred / "portrait.safetensors",
        fallback / "portrait.safetensors",
    )
    assert discovery.lora_candidates(r"nested\style.safetensors") == (
        fallback / "nested" / "style.safetensors",
        preferred / "nested" / "style.safetensors",
    )
    assert discovery.checkpoint_candidates("../outside.safetensors") == ()


def test_preserves_windows_paths_when_inspected_cross_platform(tmp_path: Path) -> None:
    configured = Path(r"D:\Shared\ComfyUI\models")
    _write_json(
        tmp_path / "Comfy Desktop" / "settings.json",
        {"modelsDirs": [str(configured)]},
    )

    result = comfy_desktop.discover_comfy_desktop(tmp_path)

    assert result.model_roots[0].path == (
        tmp_path / "Local" / "Comfy-Desktop" / "ComfyUI-Shared" / "models"
    )
    assert result.model_roots[1].path == configured


def test_malformed_state_is_reported_without_aborting(tmp_path: Path) -> None:
    desktop = tmp_path / "Comfy Desktop"
    desktop.mkdir()
    (desktop / "settings.json").write_text("{not json", encoding="utf-8")
    _write_json(desktop / "installations.json", {"unexpected": "object"})
    _write_json(desktop / "last-session.json", [])

    result = comfy_desktop.discover_comfy_desktop(tmp_path)

    assert result.detected
    assert result.installations == ()
    assert result.selected_installation is None
    assert len(result.warnings) == 3
    assert any("Could not read settings.json" in warning for warning in result.warnings)
    assert any("installations.json" in warning for warning in result.warnings)
    assert any("last-session.json" in warning for warning in result.warnings)


def test_missing_appdata_is_cross_platform_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APPDATA", raising=False)

    result = comfy_desktop.discover_comfy_desktop()

    assert not result.detected
    assert result.model_roots == ()
    assert result.host_candidates == (
        "127.0.0.1:8188",
        "127.0.0.1:8000",
        "127.0.0.1:8001",
    )
    assert "APPDATA is unavailable" in result.warnings[0]


class _FakeResponse:
    def __init__(self, value: object):
        self.payload = json.dumps(value).encode()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _request_url(request: Any) -> str:
    return str(request.full_url)


def test_probe_finds_running_api_and_lists_live_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        assert timeout == 0.25
        url = _request_url(request)
        calls.append(url)
        if "127.0.0.1:9000" in url:
            raise URLError("not running")
        if url.endswith("/system_stats"):
            return _FakeResponse({"system": {"os": "nt"}})
        if url.endswith("/models/checkpoints"):
            return _FakeResponse(
                ["z.safetensors", "folder/a.ckpt", "notes.txt", "z.safetensors"]
            )
        if url.endswith("/models/loras"):
            return _FakeResponse(["people/person.safetensors", "style.pt"])
        raise AssertionError(url)

    monkeypatch.setattr(comfy_desktop.urllib.request, "urlopen", fake_urlopen)

    status = comfy_desktop.probe_comfy_api(
        ["https://remote.example:8188", "127.0.0.1:9000", "localhost:8188"],
        timeout=0.25,
    )

    assert status == comfy_desktop.ComfyApiStatus(
        reachable=True,
        host="localhost:8188",
        checkpoints=("folder/a.ckpt", "z.safetensors"),
        loras=("people/person.safetensors", "style.pt"),
        error=None,
    )
    assert calls == [
        "http://127.0.0.1:9000/api/system_stats",
        "http://127.0.0.1:9000/system_stats",
        "http://localhost:8188/api/system_stats",
        "http://localhost:8188/api/models/checkpoints",
        "http://localhost:8188/api/models/loras",
    ]


def test_live_model_enumeration_falls_back_to_object_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        url = _request_url(request)
        if "/models/" in url:
            raise URLError("old ComfyUI")
        if url.endswith("/object_info/CheckpointLoaderSimple"):
            return _FakeResponse(
                {
                    "CheckpointLoaderSimple": {
                        "input": {
                            "required": {
                                "ckpt_name": [["base.safetensors"], {"tooltip": "x"}]
                            }
                        }
                    }
                }
            )
        if url.endswith("/object_info/LoraLoader"):
            return _FakeResponse(
                {
                    "LoraLoader": {
                        "input": {
                            "required": {"lora_name": [["subject.safetensors"], {}]}
                        }
                    }
                }
            )
        raise AssertionError(url)

    monkeypatch.setattr(comfy_desktop.urllib.request, "urlopen", fake_urlopen)

    checkpoints, loras = comfy_desktop.enumerate_live_models("127.0.0.1:8188")

    assert checkpoints == ("base.safetensors",)
    assert loras == ("subject.safetensors",)


def test_probe_reports_reachable_when_model_listing_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        if _request_url(request).endswith("/system_stats"):
            return _FakeResponse({})
        raise URLError("model endpoint unavailable")

    monkeypatch.setattr(comfy_desktop.urllib.request, "urlopen", fake_urlopen)

    result = comfy_desktop.probe_comfy_api(["127.0.0.1:8188"])

    assert result.reachable
    assert result.host == "127.0.0.1:8188"
    assert result.checkpoints == ()
    assert result.loras == ()
    assert result.error is not None
    assert "running" in result.error


@pytest.mark.parametrize(
    "host",
    [
        "https://127.0.0.1:8188",
        "192.168.1.5:8188",
        "example.com:8188",
        "127.0.0.1:70000",
        "http://127.0.0.1:8188/path",
    ],
)
def test_enumeration_rejects_nonlocal_or_unsafe_hosts(host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        comfy_desktop.enumerate_live_models(host)
