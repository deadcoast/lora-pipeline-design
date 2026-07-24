from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Self
from urllib.error import URLError

import pytest

from src import comfy_desktop


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


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
    )
    assert result.preferred_model_root == result.model_roots[0]
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
    assert result.model_roots[0].path == tmp_path / "newer" / "ComfyUI" / "models"


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
                "modelDirs": [str(private_a)],
            },
            {
                "id": "b",
                "name": "Portrait B",
                "sourceId": "standalone",
                "installPath": str(install_b),
                "useSharedModels": False,
                "modelDirs": [str(private_b)],
                "modelDirsPrimary": str(private_b),
            },
        ],
    )
    _write_json(
        desktop / "port-locks" / "port-8201.json",
        {"pid": 101, "installationName": "Portrait A", "timestamp": 1000},
    )
    _write_json(
        desktop / "port-locks" / "port-8202.json",
        {"pid": 102, "installationName": "Portrait B", "timestamp": 2000},
    )

    discovered = comfy_desktop.discover_comfy_desktop(appdata)
    selected = comfy_desktop.select_comfy_installation(discovered, "b")

    assert selected.selected_installation is not None
    assert selected.selected_installation.id == "b"
    assert selected.host_candidates == ("127.0.0.1:8202",)
    assert selected.model_roots[0].path == private_b
    assert selected.model_roots[0].loras == private_b / "loras"
    assert all(root.path != private_a for root in selected.model_roots)


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


def test_preserves_windows_paths_when_inspected_cross_platform(tmp_path: Path) -> None:
    configured = Path(r"D:\Shared\ComfyUI\models")
    _write_json(
        tmp_path / "Comfy Desktop" / "settings.json",
        {"modelsDirs": [str(configured)]},
    )

    result = comfy_desktop.discover_comfy_desktop(tmp_path)

    assert result.model_roots[0].path == configured


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
