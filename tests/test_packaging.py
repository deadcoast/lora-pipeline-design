import json
import os
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from scripts.generate_windows_icon import create_icon
from src.app_models import AppSettings, RunResult, TrainingPreset
from src.app_settings import save_settings
from src.io_utils import artifact_fingerprint

REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGED_APP = REPOSITORY / "dist" / "MIVIMLoRAStudio.dist" / "MIVIMLoRAStudio.exe"


def test_windows_icon_is_a_real_multiresolution_ico(tmp_path: Path) -> None:
    destination = create_icon(tmp_path / "mivim.ico")

    with Image.open(destination) as icon:
        assert icon.format == "ICO"
        assert icon.size == (256, 256)
        sizes = icon.info.get("sizes")

    assert isinstance(sizes, set)
    assert {(16, 16), (32, 32), (48, 48), (256, 256)} <= sizes


@pytest.mark.skipif(
    not PACKAGED_APP.is_file(),
    reason="Build the Windows package before running packaged UI acceptance.",
)
def test_built_windows_package_renders_the_real_result_handoff(
    tmp_path: Path,
) -> None:
    grid = tmp_path / "portrait_grid.png"
    Image.new("RGB", (960, 540), "#29466f").save(grid)
    workflow = tmp_path / "portrait_v1_use_in_comfyui.json"
    workflow.write_text('{"version": 0.4, "nodes": []}\n', encoding="utf-8")
    installation_root = tmp_path / "ComfyUI-acceptance"
    comfyui_root = installation_root / "ComfyUI"
    lora = comfyui_root / "models" / "loras" / "portrait_v1.safetensors"
    lora.parent.mkdir(parents=True)
    lora.write_bytes(b"packaged lora acceptance")
    checkpoint = comfyui_root / "models" / "checkpoints" / "sdxl.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"packaged checkpoint acceptance")
    installation_id = "packaged-desktop"
    installation_name = "Packaged Acceptance Desktop"
    record = RunResult(
        project_name="portrait_v1",
        mode="full",
        result_path=grid,
        completed_at="2026-07-24T12:00:00+00:00",
        trigger="ohwx person",
        preset=TrainingPreset.BALANCED,
        checkpoint=checkpoint.name,
        lora_name=lora.name,
        result_fingerprint=artifact_fingerprint(grid),
        workflow_path=workflow,
        workflow_fingerprint=artifact_fingerprint(workflow),
        comfy_installation_id=installation_id,
        comfy_installation_name=installation_name,
        comfy_host="127.0.0.1:8188",
        installed_lora_path=lora,
        lora_fingerprint=artifact_fingerprint(lora),
        checkpoint_path=checkpoint,
        checkpoint_fingerprint=artifact_fingerprint(checkpoint),
    )
    settings_path = save_settings(
        AppSettings(
            comfy_installation_id=installation_id,
            comfy_lora_dir=lora.parent,
            comfy_checkpoint=checkpoint.name,
            training_base_model=checkpoint,
            last_result=record,
        ),
        tmp_path / "settings.json",
    )
    screenshot = tmp_path / "packaged-result.png"
    environment = os.environ.copy()
    appdata = tmp_path / "empty-roaming"
    desktop_config = appdata / "Comfy Desktop"
    desktop_config.mkdir(parents=True)
    (desktop_config / "settings.json").write_text(
        json.dumps({"selectedInstallationId": installation_id}),
        encoding="utf-8",
    )
    (desktop_config / "installations.json").write_text(
        json.dumps(
            [
                {
                    "id": installation_id,
                    "name": installation_name,
                    "sourceId": "standalone",
                    "installPath": str(installation_root),
                    "status": "installed",
                    "useSharedModels": False,
                    "useSharedInputOutput": False,
                }
            ]
        ),
        encoding="utf-8",
    )
    environment["APPDATA"] = str(appdata)
    environment["MIVIM_LORA_PIPELINE_DATA_DIR"] = str(tmp_path / "app-data")

    completed = subprocess.run(
        [
            str(PACKAGED_APP),
            "--settings",
            str(settings_path),
            "--screenshot",
            str(screenshot),
            "--screenshot-page",
            "result",
            "--screenshot-delay",
            "2500",
        ],
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Desktop screenshot:" in completed.stdout
    assert "Desktop result handoff: ready" in completed.stdout
    with Image.open(screenshot) as captured:
        assert captured.width >= 1040
        assert captured.height >= 690
