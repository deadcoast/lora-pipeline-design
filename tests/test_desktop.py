from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit, QPushButton
from pytestqt.exceptions import TimeoutError as QtBotTimeout

from src import app_service
from src.app_models import AppSettings
from src.app_service import ApplicationService
from src.comfy_desktop import (
    ComfyApiStatus,
    ComfyDesktopDiscovery,
    ComfyLauncher,
)
from src.desktop import NAV_RESULTS, NAV_RUN, MainWindow


class _OfflineLocalService(ApplicationService):
    def discover_comfy(self) -> ComfyDesktopDiscovery:
        return ComfyDesktopDiscovery(
            config_dir=None,
            legacy_config_dir=None,
            installations=(),
            selected_installation=None,
            model_roots=(),
            input_dir=None,
            output_dir=None,
            host_candidates=("127.0.0.1:8188",),
            launcher=ComfyLauncher(
                "Comfy Desktop",
                "test",
                "",
                (),
                False,
            ),
            warnings=(),
        )

    def apply_discovery(
        self,
        settings: AppSettings,
        discovery: ComfyDesktopDiscovery | None = None,
    ) -> AppSettings:
        return settings


def _write_training_photos(folder: Path, count: int = 12) -> None:
    folder.mkdir(parents=True)
    for index in range(count):
        Image.new(
            "RGB",
            (96 + index, 80 + index),
            (20 + index * 15, 80, 140),
        ).save(folder / f"photo_{index:02d}.png")


def test_user_can_prepare_photos_through_the_actual_desktop_pipeline(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(
            False,
            None,
            (),
            (),
            "not running",
        ),
    )
    photos = tmp_path / "photos"
    _write_training_photos(photos)
    settings = AppSettings(
        comfy_checkpoint="",
        datasets_dir=tmp_path / "datasets",
        checkpoints_dir=tmp_path / "checkpoints",
        grids_dir=tmp_path / "grids",
        run_configs_dir=tmp_path / "configs",
    )
    window = MainWindow(
        settings=settings,
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=Path(__file__).resolve().parents[1]),
    )
    qtbot.addWidget(window)
    window.show()

    qtbot.mouseClick(
        window.findChild(QPushButton, "continueToProjectButton"),
        Qt.MouseButton.LeftButton,
    )
    window.photo_drop.set_folder(photos)
    project_name = window.findChild(QLineEdit, "projectNameEdit")
    assert project_name is not None
    project_name.setText("gui_prepare_v1")
    qtbot.mouseClick(
        window.findChild(QPushButton, "reviewSetupButton"),
        Qt.MouseButton.LeftButton,
    )

    qtbot.waitUntil(
        lambda: (
            window.pages.currentIndex() == NAV_RUN and window.prepare_button.isEnabled()
        ),
        timeout=10_000,
    )
    qtbot.mouseClick(window.prepare_button, Qt.MouseButton.LeftButton)
    try:
        qtbot.waitUntil(
            lambda: window.pages.currentIndex() == NAV_RESULTS,
            timeout=20_000,
        )
    except QtBotTimeout:
        pytest.fail(
            f"Desktop pipeline did not reach results. Stage: "
            f"{window.run_stage_label.text()}\n{window.run_log.toPlainText()}"
        )

    dataset = tmp_path / "datasets" / "gui_prepare_v1"
    assert dataset.is_dir()
    assert len(list(dataset.glob("*.png"))) == 12
    assert len(list(dataset.glob("*.txt"))) == 12
    assert (dataset / ".source-manifest.json").is_file()
    assert window._latest_result == dataset
    assert "photos are prepared" in window.result_image.text().lower()
    assert window.result_title_label.text() == "Your photos are ready for training"
    assert not window.result_open_comfy_button.isVisible()


def test_editing_a_reviewed_project_invalidates_and_rechecks_readiness(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "not running"),
    )
    photos = tmp_path / "photos"
    _write_training_photos(photos)
    window = MainWindow(
        settings=AppSettings(
            comfy_checkpoint="",
            datasets_dir=tmp_path / "datasets",
            checkpoints_dir=tmp_path / "checkpoints",
            grids_dir=tmp_path / "grids",
            run_configs_dir=tmp_path / "configs",
        ),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=Path(__file__).resolve().parents[1]),
    )
    qtbot.addWidget(window)
    window.show()
    window.photo_drop.set_folder(photos)
    window.project_name_edit.setText("reviewed_v1")
    window.review_project()
    qtbot.waitUntil(lambda: window.prepare_button.isEnabled(), timeout=10_000)

    window.project_name_edit.setFocus()
    window.project_name_edit.selectAll()
    qtbot.keyClicks(window.project_name_edit, "reviewed_v2")

    assert not window.prepare_button.isEnabled()
    window.start_pipeline("prepare")
    assert window.pipeline_process is None
    assert window.run_stack.currentIndex() == 0
    qtbot.waitUntil(lambda: window.prepare_button.isEnabled(), timeout=10_000)
    assert window.draft is not None
    assert window.draft.project_name == "reviewed_v2"


class _BrokenWorkerService(_OfflineLocalService):
    def build_command(self, settings, draft, *, mode):
        command = super().build_command(settings, draft, mode=mode)
        return app_service.PipelineCommand(
            program=str(self.root / "missing-pipeline-worker.exe"),
            arguments=command.arguments,
            environment=command.environment,
            working_directory=command.working_directory,
            mode=command.mode,
            expected_result=command.expected_result,
        )


def test_failed_pipeline_start_has_visible_recovery_actions(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "not running"),
    )
    photos = tmp_path / "photos"
    _write_training_photos(photos)
    window = MainWindow(
        settings=AppSettings(
            comfy_checkpoint="",
            datasets_dir=tmp_path / "datasets",
            checkpoints_dir=tmp_path / "checkpoints",
            grids_dir=tmp_path / "grids",
            run_configs_dir=tmp_path / "configs",
        ),
        settings_path=tmp_path / "settings.json",
        service=_BrokenWorkerService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()
    window.photo_drop.set_folder(photos)
    window.project_name_edit.setText("broken_start_v1")
    window.review_project()
    qtbot.waitUntil(lambda: window.prepare_button.isEnabled(), timeout=10_000)

    qtbot.mouseClick(window.prepare_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: window.run_stage_label.text() == "The pipeline could not start",
        timeout=5_000,
    )

    assert window.retry_after_run_button.isVisible()
    assert window.edit_after_run_button.isVisible()
    assert window.nav_buttons[NAV_RUN].isEnabled()
    assert not window.cancel_button.isEnabled()

    window._installed_lora_name = "broken_start_v1.safetensors"
    window._show_recovery_actions(True)
    assert window.retry_validation_button.isVisible()


def test_packaged_build_hides_unavailable_vision_captioning(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MIVIM_PACKAGED_APP", "1")
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()

    assert window.vision_checkbox.isHidden()
    assert not window.vision_checkbox.isChecked()
