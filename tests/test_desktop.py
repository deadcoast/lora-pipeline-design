from __future__ import annotations

import json
import os
import sys
import threading
import tomllib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import shiboken6
from PIL import Image
from PySide6.QtCore import QProcess, Qt, QTimer, QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
)
from pytestqt.exceptions import TimeoutError as QtBotTimeout

from src import __version__, app_service, config, dataset_review, prepare_dataset
from src.app_models import (
    AppSettings,
    ProjectDraft,
    RunResult,
    TrainingPreset,
    ValidationRecovery,
)
from src.app_service import ApplicationService, PipelineCommand
from src.app_settings import load_settings, save_settings
from src.checkpoint_review import CheckpointReviewDialog
from src.comfy_client import (
    ComfyUserProfile,
    PromptCancellationResult,
    WorkflowPublication,
)
from src.comfy_desktop import (
    ComfyApiStatus,
    ComfyDesktopDiscovery,
    ComfyInstallation,
    ComfyLauncher,
    ComfyModelRoot,
    ComfyRuntimeHint,
)
from src.dataset_review import PreparedDatasetDialog
from src.desktop import NAV_RESULTS, NAV_RUN, MainWindow, _build_parser
from src.io_utils import artifact_fingerprint
from src.photo_review import PhotoIssueDialog
from src.training_checkpoints import TrainingCheckpoint


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


class _FixedDiscoveryService(_OfflineLocalService):
    def __init__(
        self,
        *,
        root: Path,
        discovery: ComfyDesktopDiscovery,
    ) -> None:
        super().__init__(root=root)
        self.fixed_discovery = discovery

    def discover_comfy(self) -> ComfyDesktopDiscovery:
        return self.fixed_discovery


def _write_training_photos(folder: Path, count: int = 12) -> None:
    folder.mkdir(parents=True)
    for index in range(count):
        Image.new(
            "RGB",
            (96 + index, 80 + index),
            (20 + index * 15, 80, 140),
        ).save(folder / f"photo_{index:02d}.png")


def _script_pipeline_command(tmp_path: Path, script: str) -> PipelineCommand:
    return PipelineCommand(
        program=sys.executable,
        arguments=("-u", "-c", script),
        environment=dict(os.environ),
        working_directory=tmp_path,
        mode="full",
        expected_result=tmp_path / "expected-grid.png",
    )


def _wait_for_initial_setup(qtbot, window: MainWindow) -> None:
    qtbot.waitUntil(
        lambda: window.comfy_badge.text() != "CHECKING",
        timeout=5_000,
    )


def _click_diagnostic_action(
    qtbot,
    window: MainWindow,
    action: app_service.ReadinessAction,
    *,
    code: str = "test",
) -> None:
    item = app_service.ReadinessItem(
        code,
        "Test repair",
        "error",
        "A user-facing repair is required.",
        action,
    )
    window.nav_buttons[NAV_RUN].setEnabled(True)
    window.pages.setCurrentIndex(NAV_RUN)
    window.run_stack.setCurrentIndex(0)
    window._readiness_finished(
        app_service.ReadinessReport(
            (item,),
            can_prepare=False,
            can_run_full=False,
            comfy_status=ComfyApiStatus(False, None, (), (), "test"),
            source=None,
            prepared=None,
        )
    )
    QApplication.processEvents()
    button = next(
        candidate
        for candidate in window.diagnostics_widget.findChildren(QPushButton)
        if candidate.isVisible() and candidate.text() == action.label
    )
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)


def _result_desktop_discovery(
    root: Path,
    *,
    installation_id: str = "desktop-a",
    installation_name: str = "Portrait Desktop",
) -> ComfyDesktopDiscovery:
    installation = ComfyInstallation(
        id=installation_id,
        name=installation_name,
        source_id="standalone",
        install_path=root,
        comfyui_path=root,
        version="test",
        status="installed",
        last_launched_at=1,
        use_shared_models=False,
        use_shared_input_output=False,
    )
    model_root = ComfyModelRoot(
        path=root / "models",
        loras=root / "models" / "loras",
        checkpoints=root / "models" / "checkpoints",
        source=installation_name,
        exists=True,
    )
    return ComfyDesktopDiscovery(
        config_dir=root,
        legacy_config_dir=None,
        installations=(installation,),
        selected_installation=installation,
        model_roots=(model_root,),
        input_dir=None,
        output_dir=None,
        host_candidates=(),
        launcher=ComfyLauncher("Comfy Desktop", "test", "", (), False),
        warnings=(),
    )


def _ready_full_result(
    root: Path,
    *,
    installation_id: str = "desktop-a",
    installation_name: str = "Portrait Desktop",
) -> tuple[AppSettings, RunResult, Path]:
    grid = root / "portrait_grid.png"
    Image.new("RGB", (64, 64), "blue").save(grid)
    workflow = root / "portrait_v1_use_in_comfyui.json"
    workflow.write_text('{"version": 0.4, "nodes": []}\n', encoding="utf-8")
    lora = root / "models" / "loras" / "portrait.safetensors"
    lora.parent.mkdir(parents=True)
    lora.write_bytes(b"lora")
    checkpoint = root / "models" / "checkpoints" / "sdxl.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
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
    settings = AppSettings(
        comfy_installation_id=installation_id,
        comfy_lora_dir=lora.parent,
        comfy_checkpoint=checkpoint.name,
        training_base_model=checkpoint,
        last_result=record,
    )
    return settings, record, workflow


def _open_online_result_window(
    qtbot,
    root: Path,
    monkeypatch,
    *,
    host: str = "127.0.0.1:8188",
) -> tuple[MainWindow, RunResult, Path, _FixedDiscoveryService]:
    settings, record, workflow = _ready_full_result(root)
    discovery = replace(
        _result_desktop_discovery(root),
        runtime_hints=(
            ComfyRuntimeHint(
                host=host,
                installation_name="Portrait Desktop",
                pid=1234,
                timestamp=9_999_999_999_999,
            ),
        ),
    )
    service = _FixedDiscoveryService(root=root, discovery=discovery)

    def reachable_status(
        hosts: Sequence[str],
        **_kwargs,
    ) -> ComfyApiStatus:
        candidates = tuple(hosts)
        selected_host = candidates[0] if candidates else None
        return ComfyApiStatus(
            selected_host is not None,
            selected_host,
            (record.checkpoint,),
            (record.lora_name,),
            None if selected_host is not None else "offline",
        )

    monkeypatch.setattr(app_service, "probe_comfy_api", reachable_status)
    window = MainWindow(
        settings=settings,
        settings_path=root / "settings.json",
        service=service,
    )
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(
        lambda: (
            window._verified_result_identity is not None
            and window._verified_result_identity[0] == record
            and window.comfy_status is not None
            and window.comfy_status.reachable
        ),
        timeout=5_000,
    )
    return window, record, workflow, service


def test_user_can_prepare_photos_through_the_actual_desktop_pipeline(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    opened_urls: list[QUrl] = []

    def capture_opened_url(url: QUrl) -> bool:
        opened_urls.append(url)
        return True

    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        capture_opened_url,
    )
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
    duplicate = photos / "duplicate.png"
    duplicate.write_bytes((photos / "photo_00.png").read_bytes())
    source_before = {path.name: path.read_bytes() for path in photos.iterdir()}
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
    assert window.trainer_help_button.isVisible()
    assert window.base_model_edit.isHidden()
    assert window.storage_card.isHidden()
    assert window.connection_details_card.isHidden()
    window.advanced_setup_toggle.setChecked(True)
    assert window.base_model_edit.isVisible()
    assert window.storage_card.isVisible()
    assert window.connection_details_card.isVisible()
    window.advanced_setup_toggle.setChecked(False)

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
    qtbot.waitUntil(
        lambda: window.run_stack.currentIndex() == 1,
        timeout=5_000,
    )
    assert window.open_logs_button.isVisible()
    qtbot.mouseClick(window.open_logs_button, Qt.MouseButton.LeftButton)
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
    assert {path.name: path.read_bytes() for path in photos.iterdir()} == source_before
    assert (dataset / ".source-manifest.json").is_file()
    assert (tmp_path / "configs" / "gui_prepare_v1-result.json").is_file()
    assert window.settings.last_result is not None
    assert window.settings.last_result.mode == "prepare"
    assert window.settings.last_result.result_path == dataset
    assert window._latest_result == dataset
    assert "photos are prepared" in window.result_image.text().lower()
    assert window.result_title_label.text() == "Your photos are ready for training"
    assert not window.result_open_comfy_button.isVisible()
    assert window.review_prepared_button.isVisible()
    assert window.continue_training_button.isVisible()
    window.studio_log.flush()
    log_text = window.studio_log.path.read_text(encoding="utf-8")
    assert "photos.check.finished" in log_text
    assert "readiness.check.finished" in log_text
    assert "pipeline.started" in log_text
    assert "pipeline.output" in log_text
    assert "pipeline.finished" in log_text
    assert opened_urls
    assert (
        Path(opened_urls[-1].toLocalFile()).resolve() == (tmp_path / "logs").resolve()
    )

    qtbot.mouseClick(
        window.continue_training_button,
        Qt.MouseButton.LeftButton,
    )
    qtbot.waitUntil(
        lambda: (
            window.pages.currentIndex() == NAV_RUN
            and window.readiness_report is not None
            and window.readiness_report.prepared is not None
        ),
        timeout=10_000,
    )
    assert window.readiness_report is not None
    assert window.readiness_report.prepared is not None
    assert window.readiness_report.prepared.ready
    assert not window.prepare_button.isEnabled()
    assert window.start_button.text() == "Continue with training"
    assert window.findChild(QScrollArea, "readinessScroll") is not None
    assert window._validated_settings is not None
    assert window._validated_draft is not None
    resumed = window.service.build_command(
        window._validated_settings,
        window._validated_draft,
        mode="full",
    )
    assert "--skip-prep" in resumed.arguments
    assert "--src" not in resumed.arguments


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


def test_sdxl_confirmation_is_bound_to_the_exact_selected_checkpoint(
    qtbot,
    tmp_path: Path,
) -> None:
    window = MainWindow(
        settings=AppSettings(
            comfy_checkpoint="portraitXL.safetensors",
            confirmed_sdxl_checkpoint="portraitXL.safetensors",
        ),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()

    assert window.sdxl_confirm_checkbox.isChecked()
    window.checkpoint_combo.setCurrentText("differentXL.safetensors")
    assert not window.sdxl_confirm_checkbox.isChecked()

    def accept_confirmation() -> None:
        box = next(
            widget
            for widget in QApplication.topLevelWidgets()
            if isinstance(widget, QMessageBox)
            and widget.windowTitle() == "Confirm the SDXL base model"
        )
        confirm = next(
            button
            for button in box.buttons()
            if button.text() == "Confirm SDXL base model"
        )
        qtbot.mouseClick(confirm, Qt.MouseButton.LeftButton)

    QTimer.singleShot(0, accept_confirmation)
    window.confirm_sdxl_checkpoint()

    collected = window._collect_settings()
    assert collected.confirmed_sdxl_checkpoint == "differentXL.safetensors"


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


class _BrokenSetupService(_OfflineLocalService):
    def discover_comfy(self) -> ComfyDesktopDiscovery:
        raise RuntimeError("simulated local discovery failure")


def test_background_setup_failure_is_saved_for_recovery(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    opened: list[Path] = []

    def capture_opened_log_folder(url: QUrl) -> bool:
        opened.append(Path(url.toLocalFile()))
        return True

    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        capture_opened_log_folder,
    )
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_BrokenSetupService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()

    qtbot.waitUntil(
        lambda: window.comfy_badge.text() == "CHECK FAILED",
        timeout=5_000,
    )
    window.studio_log.flush()
    log_text = window.studio_log.path.read_text(encoding="utf-8")
    plain_message = window.comfy_summary_label.text()

    assert "setup.check.failed" in log_text
    assert "simulated local discovery failure" in log_text
    assert "Traceback (most recent call last)" in log_text
    assert "RuntimeError: simulated local discovery failure" in log_text
    assert "discover_comfy" in log_text
    assert plain_message == (
        "The automatic check could not finish: simulated local discovery failure"
    )
    assert "Traceback" not in plain_message
    assert "RuntimeError" not in plain_message
    assert window.setup_logs_button.isVisible()
    qtbot.mouseClick(window.setup_logs_button, Qt.MouseButton.LeftButton)
    assert opened == [tmp_path / "logs"]


def test_unrecorded_prepared_photo_blocks_reuse_and_offers_a_new_version(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "offline"),
    )
    photos = tmp_path / "photos"
    _write_training_photos(photos)
    datasets = tmp_path / "datasets"
    monkeypatch.setattr(config, "DATASETS_DIR", datasets)
    prepared = prepare_dataset.prepare(photos, "added_photo_v1", trigger="ohwx person")
    extra = prepared / "holiday.jpg"
    Image.new("RGB", (128, 128), "blue").save(extra)
    extra.with_suffix(".txt").write_text("ohwx person, added photo\n", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in prepared.iterdir()}
    window = MainWindow(
        settings=AppSettings(
            comfy_checkpoint="",
            datasets_dir=datasets,
            checkpoints_dir=tmp_path / "checkpoints",
            grids_dir=tmp_path / "grids",
            run_configs_dir=tmp_path / "configs",
        ),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=Path(__file__).resolve().parents[1]),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)
    qtbot.mouseClick(
        window.findChild(QPushButton, "continueToProjectButton"),
        Qt.MouseButton.LeftButton,
    )
    window.photo_drop.set_folder(photos)
    window.project_name_edit.setText("added_photo_v1")
    window.trigger_edit.setText("ohwx person")
    qtbot.mouseClick(
        window.findChild(QPushButton, "reviewSetupButton"),
        Qt.MouseButton.LeftButton,
    )
    qtbot.waitUntil(lambda: window.readiness_report is not None, timeout=10_000)
    assert window.readiness_report is not None
    assert window.readiness_report.prepared is not None
    assert not window.readiness_report.prepared.ready
    collision = next(
        item for item in window.readiness_report.items if item.code == "collision"
    )
    assert collision.state == "error"
    assert "holiday.jpg was not created by this preparation run" in collision.detail
    assert not window.start_button.isEnabled()
    assert window.pipeline_process is None
    with pytest.raises(ValueError, match="holiday.jpg"):
        window.service.build_command(
            window._collect_settings(), window._build_draft(), mode="full"
        )

    repair = next(
        button
        for button in window.diagnostics_widget.findChildren(QPushButton)
        if button.text() == "Use the next version name"
    )
    qtbot.mouseClick(repair, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: (
            window.readiness_report is not None
            and window.project_name_edit.text() == "added_photo_v2"
            and window.prepare_button.isEnabled()
        ),
        timeout=10_000,
    )
    assert {path.name: path.read_bytes() for path in prepared.iterdir()} == before
    assert window.pipeline_process is None


def test_invalid_prepared_pairs_disable_start_before_qprocess(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "offline"),
    )
    photos = tmp_path / "photos"
    _write_training_photos(photos)
    datasets = tmp_path / "datasets"
    prepared = datasets / "pair_guard_v1"
    prepared.mkdir(parents=True)
    Image.new("RGB", (128, 128), "white").save(prepared / "001.png")
    Image.new("RGB", (128, 128), "white").save(prepared / "001.jpg")
    (prepared / "001.txt").write_text("ohwx person\n", encoding="utf-8")
    (prepared / ".source-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "settings": {
                    "trigger": "ohwx person",
                    "size": 1024,
                    "caption_mode": "template",
                },
                "sources": [],
            }
        ),
        encoding="utf-8",
    )
    settings = AppSettings(
        comfy_checkpoint="",
        datasets_dir=datasets,
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
    window.photo_drop.set_folder(photos)
    window.project_name_edit.setText("pair_guard_v1")
    window.review_project()

    qtbot.waitUntil(lambda: window.readiness_report is not None, timeout=10_000)
    assert window.readiness_report is not None
    collision = next(
        item for item in window.readiness_report.items if item.code == "collision"
    )
    assert collision.state == "error"
    assert "001.jpg" in collision.detail
    assert "001.png" in collision.detail
    assert collision.action is not None
    assert collision.action.kind is app_service.ReadinessActionKind.REVIEW_PREPARED
    assert collision.action.label == "Review prepared files"
    assert not window.start_button.isEnabled()
    assert window.pipeline_process is None

    qtbot.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
    assert window.pipeline_process is None
    with pytest.raises(ValueError, match="001.jpg"):
        window.service.build_command(
            window._collect_settings(),
            window._build_draft(),
            mode="full",
        )


def test_verified_missing_comfy_lora_folder_has_safe_gui_repair(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_root = tmp_path / "ComfyUI" / "models"
    model_root.mkdir(parents=True)
    lora_dir = model_root / "loras"
    installation = ComfyInstallation(
        id="selected",
        name="Selected Desktop",
        source_id="standalone",
        install_path=tmp_path / "ComfyUI",
        comfyui_path=tmp_path / "ComfyUI",
        version=None,
        status="installed",
        last_launched_at=1,
        use_shared_models=False,
        use_shared_input_output=False,
        model_dirs=(model_root,),
    )
    discovery = ComfyDesktopDiscovery(
        config_dir=tmp_path,
        legacy_config_dir=None,
        installations=(installation,),
        selected_installation=installation,
        model_roots=(
            ComfyModelRoot(
                path=model_root,
                loras=lora_dir,
                checkpoints=model_root / "checkpoints",
                source="Selected Desktop",
                exists=True,
            ),
        ),
        input_dir=None,
        output_dir=None,
        host_candidates=(),
        launcher=ComfyLauncher("", "", "", (), False),
        warnings=(),
    )
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "offline"),
    )
    window = MainWindow(
        settings=AppSettings(
            comfy_installation_id="selected",
            comfy_lora_dir=lora_dir,
            datasets_dir=tmp_path / "workspace" / "datasets",
            checkpoints_dir=tmp_path / "workspace" / "checkpoints",
            grids_dir=tmp_path / "workspace" / "grids",
            run_configs_dir=tmp_path / "workspace" / "configs",
        ),
        settings_path=tmp_path / "settings.json",
        service=_FixedDiscoveryService(root=tmp_path, discovery=discovery),
    )
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(
        lambda: (
            window.discovery is not None
            and window.discovery.selected_installation is not None
        ),
        timeout=5_000,
    )
    window.pages.setCurrentIndex(NAV_RUN)
    window.run_stack.setCurrentIndex(0)
    window.run_readiness_checks()
    qtbot.waitUntil(lambda: window.readiness_report is not None, timeout=10_000)
    assert window.readiness_report is not None
    lora_item = next(
        item for item in window.readiness_report.items if item.code == "lora_dir"
    )
    assert lora_item.action is not None
    assert lora_item.action.kind is app_service.ReadinessActionKind.CREATE_LORA_FOLDER
    assert lora_item.action.label == "Create LoRA folder"
    create_button = next(
        button
        for button in window.diagnostics_widget.findChildren(QPushButton)
        if button.text() == "Create LoRA folder"
    )
    assert create_button.isVisible()

    qtbot.mouseClick(create_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lora_dir.is_dir, timeout=5_000)
    assert window.settings.comfy_lora_dir == lora_dir

    outside_parent = tmp_path / "not-selected"
    outside_parent.mkdir()
    outside = outside_parent / "loras"
    window.lora_dir_edit.setText(str(outside))
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )
    window.create_verified_lora_folder()
    assert not outside.exists()
    assert errors
    assert errors[-1][0] == "LoRA folder was not created"


def test_typed_photo_and_project_repairs_do_exactly_what_their_labels_say(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    chosen_photos = tmp_path / "chosen-photos"
    chosen_photos.mkdir()
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)
    reviews: list[str] = []
    monkeypatch.setattr(window, "review_project", lambda: reviews.append("reviewed"))

    window.overwrite_checkbox.setChecked(True)
    _click_diagnostic_action(
        qtbot,
        window,
        app_service.ReadinessAction(
            app_service.ReadinessActionKind.DISABLE_REBUILD,
            "Turn off rebuild to preserve existing work",
        ),
    )
    assert not window.overwrite_checkbox.isChecked()
    assert reviews == ["reviewed"]

    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(chosen_photos),
    )
    _click_diagnostic_action(
        qtbot,
        window,
        app_service.ReadinessAction(
            app_service.ReadinessActionKind.CHOOSE_PHOTO_FOLDER,
            "Choose another photo folder",
        ),
    )
    assert window.source_folder == chosen_photos
    assert reviews == ["reviewed", "reviewed"]

    photo_reviews: list[bool] = []

    def review_photos() -> bool:
        photo_reviews.append(True)
        return False

    monkeypatch.setattr(
        window,
        "review_photo_checks",
        review_photos,
    )
    _click_diagnostic_action(
        qtbot,
        window,
        app_service.ReadinessAction(
            app_service.ReadinessActionKind.REVIEW_PHOTOS,
            "Review photo checks",
        ),
    )
    assert photo_reviews == [True]
    assert window.source_folder == chosen_photos


def test_typed_external_and_trainer_repairs_keep_their_action_contracts(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)

    opened_desktop: list[bool] = []

    def open_desktop() -> bool:
        opened_desktop.append(True)
        return True

    monkeypatch.setattr(
        window,
        "open_comfy_desktop",
        open_desktop,
    )
    _click_diagnostic_action(
        qtbot,
        window,
        app_service.ReadinessAction(
            app_service.ReadinessActionKind.OPEN_COMFY_DESKTOP,
            "Open Comfy Desktop and check again",
        ),
    )
    assert opened_desktop == [True]
    assert window._return_to_readiness_when_connected
    assert window.pages.currentIndex() == 0

    _click_diagnostic_action(
        qtbot,
        window,
        app_service.ReadinessAction(
            app_service.ReadinessActionKind.SELECT_COMFY_INSTALLATION,
            "Choose a Desktop instance",
        ),
    )
    assert window._return_to_readiness_after_installation
    assert window.pages.currentIndex() == 0

    _click_diagnostic_action(
        qtbot,
        window,
        app_service.ReadinessAction(
            app_service.ReadinessActionKind.SELECT_CHECKPOINT,
            "Choose an SDXL model",
        ),
    )
    assert window._return_to_readiness_after_checkpoint
    assert window.pages.currentIndex() == 0

    opened_comfyui: list[bool] = []

    def open_comfyui() -> bool:
        opened_comfyui.append(True)
        return True

    monkeypatch.setattr(
        window,
        "_open_current_comfy_ui",
        open_comfyui,
    )
    _click_diagnostic_action(
        qtbot,
        window,
        app_service.ReadinessAction(
            app_service.ReadinessActionKind.OPEN_COMFYUI,
            "Open ComfyUI to add an SDXL model",
        ),
    )
    assert opened_comfyui == [True]
    assert window.pages.currentIndex() == 0

    opened_urls: list[QUrl] = []

    def open_url(url: QUrl) -> bool:
        opened_urls.append(url)
        return True

    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        open_url,
    )
    monkeypatch.setattr(
        window,
        "choose_workspace_dir",
        lambda: pytest.fail("Storage cleanup must not open the workspace chooser"),
    )
    _click_diagnostic_action(
        qtbot,
        window,
        app_service.ReadinessAction(
            app_service.ReadinessActionKind.OPEN_STORAGE_SETTINGS,
            "Open Windows Storage settings",
        ),
    )
    assert opened_urls[-1].toString() == "ms-settings:storagesense"

    trainer_choices: list[bool] = []
    reviews: list[bool] = []

    def choose_toolkit() -> bool:
        trainer_choices.append(True)
        return True

    monkeypatch.setattr(
        window,
        "choose_toolkit_dir",
        choose_toolkit,
    )
    monkeypatch.setattr(window, "review_project", lambda: reviews.append(True))
    _click_diagnostic_action(
        qtbot,
        window,
        app_service.ReadinessAction(
            app_service.ReadinessActionKind.CHOOSE_TOOLKIT_FOLDER,
            "Choose the ai-toolkit folder",
        ),
    )
    assert trainer_choices == [True]
    assert reviews == [True]

    python_choices: list[bool] = []

    def choose_python() -> bool:
        python_choices.append(True)
        return True

    monkeypatch.setattr(
        window,
        "choose_toolkit_python",
        choose_python,
    )
    _click_diagnostic_action(
        qtbot,
        window,
        app_service.ReadinessAction(
            app_service.ReadinessActionKind.CHOOSE_TOOLKIT_PYTHON,
            "Choose the ai-toolkit venv Python",
        ),
    )
    assert python_choices == [True]
    assert reviews == [True, True]

    checkpoint_repairs: list[bool] = []
    monkeypatch.setattr(
        window,
        "confirm_sdxl_checkpoint",
        lambda: checkpoint_repairs.append(True),
    )
    _click_diagnostic_action(
        qtbot,
        window,
        app_service.ReadinessAction(
            app_service.ReadinessActionKind.CONFIRM_CHECKPOINT,
            "Confirm this is an SDXL base checkpoint",
        ),
    )
    assert checkpoint_repairs == [True]


def test_offline_model_hint_explains_how_to_load_the_real_model_list(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "offline"),
    )
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)

    assert "Connect a running local ComfyUI instance" in window.model_hint.text()
    assert "Add an SDXL model" not in window.model_hint.text()


def test_duplicate_checkpoint_hint_names_active_copy_and_ignored_count(
    qtbot,
    tmp_path: Path,
) -> None:
    preferred = tmp_path / "preferred"
    shadowed = tmp_path / "shadowed"
    for folder in (preferred, shadowed):
        folder.mkdir()
        (folder / "portraitXL.safetensors").write_bytes(folder.name.encode())
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)
    window.settings = replace(
        window.settings,
        comfy_checkpoint="portraitXL.safetensors",
        training_base_model=preferred / "portraitXL.safetensors",
    )
    window.discovery = ComfyDesktopDiscovery(
        config_dir=tmp_path,
        legacy_config_dir=None,
        installations=(),
        selected_installation=None,
        model_roots=(),
        input_dir=None,
        output_dir=None,
        host_candidates=("127.0.0.1:8188",),
        launcher=ComfyLauncher("", "", "", (), False),
        warnings=(),
        checkpoint_dirs=(preferred, shadowed),
    )
    window.comfy_status = ComfyApiStatus(
        True,
        "127.0.0.1:8188",
        ("portraitXL.safetensors",),
        (),
        None,
    )

    window._populate_settings()

    assert str(preferred / "portraitXL.safetensors") in window.model_hint.text()
    assert "1 lower-priority same-named copy" in window.model_hint.text()
    assert "Choose the exact file below" not in window.model_hint.text()


def test_desktop_launch_repair_polls_then_returns_to_readiness(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    service = _OfflineLocalService(root=tmp_path)
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=service,
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)

    def launch_desktop() -> bool:
        window._desktop_poll_remaining = 18
        return True

    reviews: list[bool] = []
    monkeypatch.setattr(window, "open_comfy_desktop", launch_desktop)
    monkeypatch.setattr(window, "review_project", lambda: reviews.append(True))
    _click_diagnostic_action(
        qtbot,
        window,
        app_service.ReadinessAction(
            app_service.ReadinessActionKind.OPEN_COMFY_DESKTOP,
            "Open Comfy Desktop and check again",
        ),
    )

    original = window._collect_settings()
    window._setup_request += 1
    window._setup_refreshed(
        window._setup_request,
        original,
        (
            original,
            service.discover_comfy(),
            ComfyApiStatus(True, "127.0.0.1:8188", (), (), None),
        ),
    )
    QApplication.processEvents()

    assert not window._return_to_readiness_when_connected
    assert window._desktop_poll_remaining == 0
    assert reviews == [True]


def test_installed_desktop_without_instance_opens_desktop_not_install_guide(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery = ComfyDesktopDiscovery(
        config_dir=tmp_path / "Comfy Desktop",
        legacy_config_dir=None,
        installations=(),
        selected_installation=None,
        model_roots=(),
        input_dir=None,
        output_dir=None,
        host_candidates=(),
        launcher=ComfyLauncher(
            "Comfy Desktop",
            "desktop-app",
            r"shell:AppsFolder\desktop-app",
            ("explorer.exe", r"shell:AppsFolder\desktop-app"),
            True,
        ),
        warnings=(),
    )
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(
            False,
            None,
            (),
            (),
            "offline",
        ),
    )
    launched: list[list[str]] = []
    monkeypatch.setattr(
        "src.desktop.subprocess.Popen",
        lambda command, **_kwargs: launched.append(command),
    )
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_FixedDiscoveryService(root=tmp_path, discovery=discovery),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)

    assert window.comfy_badge.text() == "CREATE INSTANCE"
    assert window.open_desktop_button.text() == "Open Comfy Desktop"
    assert window.open_comfy_desktop()
    assert launched == [
        ["explorer.exe", r"shell:AppsFolder\desktop-app"],
    ]


def test_unavailable_saved_desktop_requires_an_explicit_user_choice(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    current = ComfyInstallation(
        id="current",
        name="Current Desktop",
        source_id="standalone",
        install_path=tmp_path / "current",
        comfyui_path=tmp_path / "current" / "ComfyUI",
        version="v0.28.2",
        status="installed",
        last_launched_at=1,
    )
    discovery = ComfyDesktopDiscovery(
        config_dir=tmp_path,
        legacy_config_dir=None,
        installations=(current,),
        selected_installation=current,
        model_roots=(),
        input_dir=None,
        output_dir=None,
        host_candidates=("127.0.0.1:8188",),
        launcher=ComfyLauncher("", "", "", (), False),
        warnings=(),
    )
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(
            False,
            None,
            (),
            (),
            "offline",
        ),
    )
    window = MainWindow(
        settings=AppSettings(comfy_installation_id="original"),
        settings_path=tmp_path / "settings.json",
        service=_FixedDiscoveryService(root=tmp_path, discovery=discovery),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)

    assert window.settings.comfy_installation_id == "original"
    assert window.comfy_badge.text() == "CHOOSE INSTANCE"
    assert "did not connect to a substitute" in window.comfy_summary_label.text()
    assert not window.installation_combo.isHidden()
    assert window.installation_combo.currentData() == "original"
    assert "Original instance unavailable" in window.installation_combo.currentText()

    _click_diagnostic_action(
        qtbot,
        window,
        app_service.ReadinessAction(
            app_service.ReadinessActionKind.SELECT_COMFY_INSTALLATION,
            "Choose a Desktop instance",
        ),
    )
    current_index = window.installation_combo.findData("current")
    assert current_index >= 0
    window.installation_combo.setCurrentIndex(current_index)
    qtbot.waitUntil(
        lambda: (
            window.pages.currentIndex() == NAV_RUN
            and not window._return_to_readiness_after_setup
        ),
        timeout=5_000,
    )

    assert load_settings(tmp_path / "settings.json").comfy_installation_id == "current"


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

    recovery = ValidationRecovery(
        project_name="broken_start_v1",
        state="validation_pending",
        lora_name="broken_start_v1.safetensors",
        installed_lora_path=tmp_path / "broken_start_v1.safetensors",
        expected_grid_path=tmp_path / "broken_start_v1.png",
        trigger="mivim person",
        preset=TrainingPreset.BALANCED,
        checkpoint="portraitXL.safetensors",
        updated_at="2026-07-24T12:00:00+00:00",
    )
    window._pending_validation_recovery = recovery
    window._installed_lora_name = recovery.lora_name
    window._show_recovery_actions(True)
    assert window.retry_validation_button.isVisible()
    assert not window.retry_after_run_button.isVisible()


def test_real_trainer_process_uses_elapsed_indeterminate_progress_then_restores(
    qtbot,
    tmp_path: Path,
) -> None:
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)
    command = _script_pipeline_command(
        tmp_path,
        (
            "import time\n"
            "print('[train] cwd = C:/ai-toolkit', flush=True)\n"
            "time.sleep(1.4)\n"
            "print('[pipeline] Copied LoRA -> model.safetensors', flush=True)\n"
            "time.sleep(0.8)\n"
            "raise SystemExit(9)\n"
        ),
    )

    window._launch_pipeline_command(command, "Starting trainer")
    qtbot.waitUntil(
        lambda: (
            window.run_progress.minimum() == 0 and window.run_progress.maximum() == 0
        ),
        timeout=3_000,
    )

    assert window.run_stage_label.text() == "Training the LoRA"
    qtbot.waitUntil(
        lambda: "Elapsed 00:01" in window.run_detail_label.text(),
        timeout=3_000,
    )
    qtbot.waitUntil(
        lambda: (
            window.run_progress.maximum() == 100 and window.run_progress.value() >= 78
        ),
        timeout=3_000,
    )
    assert window.run_stage_label.text() == "Installing in ComfyUI"
    qtbot.waitUntil(
        lambda: (
            window.pipeline_process is not None
            and window.pipeline_process.state() == QProcess.ProcessState.NotRunning
        ),
        timeout=3_000,
    )

    assert window.run_stage_label.text() == "This run needs attention"
    assert window.retry_after_run_button.isVisible()
    assert not window.use_fast_after_run_button.isVisible()
    assert not window.repair_trainer_after_run_button.isVisible()


def test_desktop_tracks_and_reports_only_its_owned_comfy_prompt(
    qtbot,
    tmp_path: Path,
) -> None:
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    command = _script_pipeline_command(tmp_path, "raise SystemExit(1)")
    command = replace(
        command,
        environment={**command.environment, "COMFY_HOST": "127.0.0.1:8188"},
    )
    window.pipeline_command = command
    prompt_id = "118762ff-c271-48f7-934a-1061bc1abafe"
    client_id = "77605834-738e-4644-be3e-0262fe5b3b14"

    window._process_pipeline_line(
        "[comfy] owned prompt queued "
        f"prompt_id={prompt_id} client_id={client_id} host=127.0.0.1:8188"
    )

    assert window._owned_comfy_prompt == (
        prompt_id,
        client_id,
        "127.0.0.1:8188",
    )
    window.run_cancelled = True
    window._run_terminal_handled = True
    window._owned_prompt_cancelled(
        PromptCancellationResult(
            status="running_interrupt_requested",
            prompt_id=prompt_id,
            client_id=client_id,
            message="Interrupt requested for this running ComfyUI prompt.",
        )
    )
    assert "Interrupt requested" in window.run_detail_label.text()


def test_pipeline_failure_ignores_validation_recovery_from_another_attempt(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    draft = ProjectDraft(project_name="attempt_bound_v1")
    window.draft = draft
    command = replace(
        _script_pipeline_command(tmp_path, "pass"),
        mode="validate",
        attempt_id="11111111-1111-4111-8111-111111111111",
    )
    window.pipeline_command = command
    recovery = ValidationRecovery(
        project_name=draft.project_name,
        state="validation_pending",
        lora_name="attempt_bound_v1.safetensors",
        installed_lora_path=tmp_path / "attempt_bound_v1.safetensors",
        expected_grid_path=tmp_path / "attempt_bound_v1_grid.png",
        trigger=draft.trigger,
        preset=draft.preset,
        checkpoint="portraitXL.safetensors",
        updated_at="2026-07-24T12:00:00+00:00",
        attempt_id="22222222-2222-4222-8222-222222222222",
    )
    monkeypatch.setattr(
        window.service,
        "read_validation_recovery",
        lambda *_args, **_kwargs: recovery,
    )

    assert window._refresh_validation_recovery_after_run() is None
    assert window._pending_validation_recovery is None


def test_saved_recovery_model_hashing_does_not_block_window_construction(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = ProjectDraft(project_name="background_recovery_v1")
    recovery = ValidationRecovery(
        project_name=draft.project_name,
        state="validation_pending",
        lora_name="background_recovery_v1.safetensors",
        installed_lora_path=tmp_path / "background_recovery_v1.safetensors",
        expected_grid_path=tmp_path / "background_recovery_v1_grid.png",
        trigger=draft.trigger,
        preset=draft.preset,
        checkpoint="portraitXL.safetensors",
        updated_at="2026-07-24T12:00:00+00:00",
        attempt_id="33333333-3333-4333-8333-333333333333",
    )
    service = _OfflineLocalService(root=tmp_path)
    worker_threads: list[int] = []
    main_thread = threading.get_ident()
    monkeypatch.setattr(
        service,
        "read_validation_recovery",
        lambda *_args, **_kwargs: recovery,
    )

    def check_recovery(*_args, **_kwargs) -> str:
        worker_threads.append(threading.get_ident())
        return "Reconnect the original files."

    monkeypatch.setattr(service, "validation_recovery_issue", check_recovery)

    window = MainWindow(
        settings=AppSettings(last_project=draft),
        settings_path=tmp_path / "settings.json",
        service=service,
    )
    qtbot.addWidget(window)

    qtbot.waitUntil(lambda: bool(worker_threads), timeout=3_000)
    assert worker_threads[0] != main_thread


def test_stale_cancel_callbacks_and_force_timer_cannot_stop_a_later_attempt(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    stale_process = QProcess(window)
    active_process = QProcess(window)
    stale_command = replace(
        _script_pipeline_command(tmp_path, "pass"),
        attempt_id="11111111-1111-4111-8111-111111111111",
    )
    active_command = replace(
        _script_pipeline_command(tmp_path, "pass"),
        attempt_id="22222222-2222-4222-8222-222222222222",
    )
    window.pipeline_process = active_process
    window.pipeline_command = active_command
    handled: list[object] = []
    forced: list[bool] = []
    monkeypatch.setattr(window, "_owned_prompt_cancelled", handled.append)
    monkeypatch.setattr(window, "_force_stop_pipeline", lambda: forced.append(True))
    result = PromptCancellationResult(
        status="not_found",
        prompt_id="prompt",
        client_id="client",
        message="Already finished.",
    )

    window._owned_prompt_cancelled_for_attempt(
        stale_process,
        stale_command,
        result,
    )
    window._force_stop_pipeline_for_attempt(stale_process, stale_command)
    window._owned_prompt_cancelled_for_attempt(
        active_process,
        stale_command,
        result,
    )
    window._force_stop_pipeline_for_attempt(active_process, stale_command)

    assert handled == []
    assert forced == []

    window._owned_prompt_cancelled_for_attempt(
        active_process,
        active_command,
        result,
    )
    window._force_stop_pipeline_for_attempt(active_process, active_command)

    assert handled == [result]
    assert forced == [True]


def test_cancel_timer_ignores_a_deleted_qprocess_wrapper(
    qtbot,
    tmp_path: Path,
) -> None:
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    process = QProcess(window)
    command = _script_pipeline_command(tmp_path, "pass")
    window.pipeline_process = process
    window.pipeline_command = command
    shiboken6.delete(process)

    assert not window._pipeline_attempt_is_active(process, command)
    window._force_stop_pipeline()

    window.pipeline_process = None
    window.pipeline_command = None


def test_unsupported_targeted_cancel_keeps_watching_the_running_job(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    process = QProcess(window)
    process.start(sys.executable, ["-c", "import time; time.sleep(30)"])
    qtbot.waitUntil(
        lambda: process.state() == QProcess.ProcessState.Running,
        timeout=3_000,
    )
    command = replace(
        _script_pipeline_command(tmp_path, "pass"),
        attempt_id="44444444-4444-4444-8444-444444444444",
    )
    window.pipeline_process = process
    window.pipeline_command = command
    window.run_cancelled = True
    stopped: list[bool] = []
    monkeypatch.setattr(
        window,
        "_stop_process_tree",
        lambda *, force: stopped.append(force),
    )

    window._owned_prompt_cancelled_for_attempt(
        process,
        command,
        PromptCancellationResult(
            status="running_interrupt_unsupported",
            prompt_id="prompt",
            client_id="client",
            message=(
                "This ComfyUI version cannot safely stop only this running prompt."
            ),
        ),
    )

    assert process.state() == QProcess.ProcessState.Running
    assert stopped == []
    assert not window.run_cancelled
    assert window.run_stage_label.text() == "ComfyUI job is still running"
    assert "Do not start another GPU task" in window.run_detail_label.text()

    process.kill()
    process.waitForFinished(3_000)
    window.pipeline_process = None
    window.pipeline_command = None


def test_close_stays_open_when_running_comfy_prompt_cannot_be_safely_stopped(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    process = QProcess(window)
    process.start(sys.executable, ["-c", "import time; time.sleep(30)"])
    qtbot.waitUntil(
        lambda: process.state() == QProcess.ProcessState.Running,
        timeout=3_000,
    )
    command = replace(
        _script_pipeline_command(tmp_path, "pass"),
        attempt_id="33333333-3333-4333-8333-333333333333",
    )
    prompt_id = "118762ff-c271-48f7-934a-1061bc1abafe"
    client_id = "77605834-738e-4644-be3e-0262fe5b3b14"
    window.pipeline_process = process
    window.pipeline_command = command
    window._owned_comfy_prompt = (prompt_id, client_id, "127.0.0.1:8188")
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.No,
    )

    def record_warning(
        _parent,
        _title: str,
        message: str,
    ) -> QMessageBox.StandardButton:
        warnings.append(message)
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "warning", record_warning)
    monkeypatch.setattr(
        "src.desktop.ComfyClient.cancel_owned_prompt",
        lambda *_args, **_kwargs: PromptCancellationResult(
            status="running_interrupt_unsupported",
            prompt_id=prompt_id,
            client_id=client_id,
            message=(
                "This ComfyUI version cannot safely stop only this running prompt."
            ),
        ),
    )
    event = QCloseEvent()

    window.closeEvent(event)

    assert not event.isAccepted()
    assert process.state() == QProcess.ProcessState.Running
    assert not window.run_cancelled
    assert not window._closing
    assert warnings
    assert "MIVIM will stay open" in warnings[0]
    assert "stop this job from its queue" in warnings[0]

    process.kill()
    process.waitForFinished(3_000)
    window.pipeline_process = None
    window.pipeline_command = None


def test_real_cuda_oom_offers_fast_plan_and_applies_the_recovery(
    qtbot,
    tmp_path: Path,
) -> None:
    photos = tmp_path / "photos"
    _write_training_photos(photos)
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)
    window.photo_drop.set_folder(photos)
    window.project_name_edit.setText("oom_recovery_v1")
    command = _script_pipeline_command(
        tmp_path,
        (
            "import sys\n"
            "print('[train] cwd = C:/ai-toolkit', flush=True)\n"
            "print('torch.OutOfMemoryError: CUDA out of memory', "
            "file=sys.stderr, flush=True)\n"
            "raise SystemExit(1)\n"
        ),
    )

    window._launch_pipeline_command(command, "Starting trainer")
    qtbot.waitUntil(
        lambda: window.run_stage_label.text() == "GPU memory filled up",
        timeout=3_000,
    )

    assert "Use Fast & check again" in window.run_detail_label.text()
    assert window.use_fast_after_run_button.isVisible()
    assert not window.repair_trainer_after_run_button.isVisible()

    def approve_fast_rebuild() -> None:
        dialog = next(
            child for child in window.findChildren(QMessageBox) if child.isVisible()
        )
        approve = next(
            child
            for child in dialog.findChildren(QPushButton)
            if child.text() == "Use Fast and rebuild this project"
        )
        approve.click()

    QTimer.singleShot(0, approve_fast_rebuild)
    qtbot.mouseClick(
        window.use_fast_after_run_button,
        Qt.MouseButton.LeftButton,
    )
    assert window.preset_combo.currentData() == TrainingPreset.FAST.value
    assert window.overwrite_checkbox.isChecked()
    assert window.run_stack.currentIndex() == 0
    qtbot.waitUntil(lambda: not window._workers, timeout=10_000)


@pytest.mark.parametrize(
    "message",
    (
        "Another MIVIM run is already using the accelerator.",
        'Another MIVIM run is already working on project "busy_v1".',
        (
            "Another MIVIM run is already using a file or folder required "
            'by project "busy_v1".'
        ),
    ),
)
def test_run_guard_contention_has_a_clear_wait_and_retry_experience(
    qtbot,
    tmp_path: Path,
    message: str,
) -> None:
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)
    command = _script_pipeline_command(
        tmp_path,
        f"import sys\nprint({message!r}, file=sys.stderr, flush=True)\nraise SystemExit(2)\n",
    )

    window._launch_pipeline_command(command, "Starting safely")
    qtbot.waitUntil(
        lambda: window.run_stage_label.text() == "Another MIVIM run is active",
        timeout=3_000,
    )

    assert "No new work was started" in window.run_detail_label.text()
    assert "choose Check and retry" in window.run_detail_label.text()
    assert window.retry_after_run_button.isVisible()
    assert not window.retry_validation_button.isVisible()
    assert not window.edit_after_run_button.isVisible()
    assert not window.open_run_folder_button.isVisible()


def test_orphaned_comfy_work_has_a_direct_desktop_recovery(
    qtbot,
    tmp_path: Path,
) -> None:
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)
    command = _script_pipeline_command(
        tmp_path,
        (
            "import sys\n"
            "print('MIVIM_RUN_BUSY: ComfyUI is still finishing MIVIM job "
            "prompt-a.', file=sys.stderr, flush=True)\n"
            "raise SystemExit(2)\n"
        ),
    )

    window._launch_pipeline_command(command, "Checking ComfyUI safety")
    qtbot.waitUntil(
        lambda: (
            window.run_stage_label.text() == "ComfyUI is finishing earlier MIVIM work"
        ),
        timeout=3_000,
    )

    assert "No new work was started" in window.run_detail_label.text()
    assert "stop that job there" in window.run_detail_label.text()
    assert window.open_desktop_after_run_button.isVisible()
    assert window.retry_after_run_button.isVisible()
    assert not window.edit_after_run_button.isVisible()
    assert not window.open_run_folder_button.isVisible()


def test_busy_validation_recovery_shows_only_validation_retry(
    qtbot,
    tmp_path: Path,
) -> None:
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)
    recovery = ValidationRecovery(
        project_name="busy_validation_v1",
        state="validation_pending",
        lora_name="busy_validation_v1.safetensors",
        installed_lora_path=tmp_path / "busy_validation_v1.safetensors",
        expected_grid_path=tmp_path / "busy_validation_v1.png",
        trigger="mivim person",
        preset=TrainingPreset.BALANCED,
        checkpoint="portraitXL.safetensors",
        updated_at="2026-07-24T12:00:00+00:00",
    )
    window._pending_validation_recovery = recovery
    window._installed_lora_name = recovery.lora_name
    command = replace(
        _script_pipeline_command(
            tmp_path,
            (
                "import sys\n"
                "print('Another MIVIM run is already using the accelerator.', "
                "file=sys.stderr, flush=True)\n"
                "raise SystemExit(2)\n"
            ),
        ),
        mode="validate",
    )

    window._launch_pipeline_command(command, "Retrying validation")
    qtbot.waitUntil(
        lambda: window.run_stage_label.text() == "Another MIVIM run is active",
        timeout=3_000,
    )

    assert "No new validation work was started" in window.run_detail_label.text()
    assert "Retry validation only" in window.run_detail_label.text()
    assert "Photos and training will not repeat" in window.run_detail_label.text()
    assert window.retry_validation_button.isVisible()
    assert not window.retry_after_run_button.isVisible()


def test_fast_oom_recovery_does_not_enable_rebuild_without_user_approval(
    qtbot,
    tmp_path: Path,
) -> None:
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)
    window.use_fast_after_run_button.setVisible(True)

    def cancel_fast_rebuild() -> None:
        dialog = next(
            child for child in window.findChildren(QMessageBox) if child.isVisible()
        )
        cancel = dialog.button(QMessageBox.StandardButton.Cancel)
        assert cancel is not None
        cancel.click()

    QTimer.singleShot(0, cancel_fast_rebuild)
    qtbot.mouseClick(
        window.use_fast_after_run_button,
        Qt.MouseButton.LeftButton,
    )

    assert window.preset_combo.currentData() == TrainingPreset.BALANCED.value
    assert not window.overwrite_checkbox.isChecked()


def test_real_missing_trainer_dependency_opens_guided_repair(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)
    repair_opened: list[bool] = []
    monkeypatch.setattr(
        window,
        "show_trainer_help",
        lambda: repair_opened.append(True),
    )
    command = _script_pipeline_command(
        tmp_path,
        (
            "import sys\n"
            "print('[train] cwd = C:/ai-toolkit', flush=True)\n"
            "print(\"ModuleNotFoundError: No module named 'toolkit'\", "
            "file=sys.stderr, flush=True)\n"
            "raise SystemExit(1)\n"
        ),
    )

    window._launch_pipeline_command(command, "Starting trainer")
    qtbot.waitUntil(
        lambda: window.run_stage_label.text() == "Trainer needs repair",
        timeout=3_000,
    )

    assert "missing a package" in window.run_detail_label.text()
    assert window.repair_trainer_after_run_button.isVisible()
    assert not window.use_fast_after_run_button.isVisible()
    qtbot.mouseClick(
        window.repair_trainer_after_run_button,
        Qt.MouseButton.LeftButton,
    )
    assert repair_opened == [True]


def test_trainer_repair_check_opens_the_visible_review_page(
    qtbot,
    tmp_path: Path,
) -> None:
    photos = tmp_path / "photos"
    _write_training_photos(photos)
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)
    window.photo_drop.set_folder(photos)
    window.project_name_edit.setText("trainer_recheck_v1")

    def click_visible_check() -> None:
        dialog = next(
            child for child in window.findChildren(QMessageBox) if child.isVisible()
        )
        button = next(
            child
            for child in dialog.findChildren(QPushButton)
            if "Check on Review page" in child.text()
        )
        button.click()

    QTimer.singleShot(0, click_visible_check)
    window.show_trainer_help()

    assert window.pages.currentIndex() == NAV_RUN
    assert window.run_stack.currentIndex() == 0
    qtbot.waitUntil(lambda: not window._workers, timeout=10_000)


@pytest.mark.parametrize("choice", ("folder", "python"))
def test_repair_trainer_dialog_choice_persists_and_rechecks(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    choice: str,
) -> None:
    toolkit = tmp_path / "ai-toolkit"
    toolkit.mkdir()
    detected_python = toolkit / "venv" / "Scripts" / "python.exe"
    detected_python.parent.mkdir(parents=True)
    detected_python.write_text("", encoding="utf-8")
    chosen_python = tmp_path / "custom-venv" / "python.exe"
    chosen_python.parent.mkdir()
    chosen_python.write_text("", encoding="utf-8")
    settings_path = tmp_path / "settings.json"
    window = MainWindow(
        settings=AppSettings(),
        settings_path=settings_path,
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)
    if choice == "folder":
        monkeypatch.setattr(
            QFileDialog,
            "getExistingDirectory",
            lambda *_args, **_kwargs: str(toolkit),
        )
        button_text = "Choose ai-toolkit folder"
        expected_python = detected_python
    else:
        window.toolkit_dir_edit.setText(str(toolkit))
        monkeypatch.setattr(
            QFileDialog,
            "getOpenFileName",
            lambda *_args, **_kwargs: (str(chosen_python), ""),
        )
        button_text = "Choose venv Python"
        expected_python = chosen_python
    request_before = window._readiness_request

    def click_trainer_choice() -> None:
        dialogs = [
            child for child in window.findChildren(QMessageBox) if child.isVisible()
        ]
        if not dialogs:
            QTimer.singleShot(10, click_trainer_choice)
            return
        dialog = dialogs[0]
        button = next(
            child
            for child in dialog.findChildren(QPushButton)
            if child.text() == button_text
        )
        button.click()

    QTimer.singleShot(0, click_trainer_choice)
    _click_diagnostic_action(
        qtbot,
        window,
        app_service.ReadinessAction(
            app_service.ReadinessActionKind.REPAIR_TRAINER,
            "Repair trainer",
        ),
        code="trainer",
    )

    saved = load_settings(settings_path)
    assert saved.ai_toolkit_dir == toolkit
    assert saved.ai_toolkit_python == expected_python
    assert window.pages.currentIndex() == NAV_RUN
    assert window.run_stack.currentIndex() == 0
    assert window._readiness_request > request_before
    qtbot.waitUntil(lambda: not window._workers, timeout=10_000)


def test_validation_failure_recovers_after_restart_without_retraining(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        app_service,
        "probe_comfy_api",
        lambda *_args, **_kwargs: ComfyApiStatus(False, None, (), (), "not running"),
    )
    opened_urls: list[QUrl] = []

    def capture_desktop_url(url: QUrl) -> bool:
        opened_urls.append(url)
        return True

    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        capture_desktop_url,
    )
    root = Path(__file__).resolve().parents[1]
    lora_dir = tmp_path / "ComfyUI" / "models" / "loras"
    lora_dir.mkdir(parents=True)
    installed = lora_dir / "restart_recovery_v1.safetensors"
    installed.write_bytes(b"trained weights")
    base_model = (
        tmp_path / "ComfyUI" / "models" / "checkpoints" / "portraitXL.safetensors"
    )
    base_model.parent.mkdir(parents=True)
    base_model.write_bytes(b"sdxl checkpoint")
    draft = ProjectDraft(
        project_name="restart_recovery_v1",
        trigger="ohwx person",
        preset=TrainingPreset.BALANCED,
    )
    settings = AppSettings(
        comfy_host="127.0.0.1:1",
        comfy_installation_id="desktop-a",
        comfy_lora_dir=lora_dir,
        comfy_checkpoint="portraitXL.safetensors",
        training_base_model=base_model,
        datasets_dir=tmp_path / "datasets",
        checkpoints_dir=tmp_path / "checkpoints",
        grids_dir=tmp_path / "grids",
        run_configs_dir=tmp_path / "configs",
        last_project=draft,
    )
    settings_path = tmp_path / "settings.json"
    save_settings(settings, settings_path)
    window = MainWindow(
        settings=settings,
        settings_path=settings_path,
        service=_OfflineLocalService(root=root),
    )
    qtbot.addWidget(window)
    window.show()
    window.draft = draft
    window._validated_settings = settings
    window._validated_draft = draft
    environment = dict(os.environ)
    environment.update(settings.environment_overrides())
    environment["COMFY_HOST"] = "127.0.0.1:1"
    environment["PYTHONUNBUFFERED"] = "1"
    attempt_id = "3d570ae6-c094-46d9-93b0-eb90cf6a3893"
    command = PipelineCommand(
        program=sys.executable,
        arguments=(
            "-m",
            "src.pipeline",
            "--attempt-id",
            attempt_id,
            "--name",
            draft.project_name,
            "--trigger",
            draft.trigger,
            "--preset",
            draft.preset.value,
            "--strengths",
            "0",
            "0.8",
            "--skip-prep",
            "--skip-train",
            "--lora",
            installed.name,
            "--base-model",
            str(base_model),
        ),
        environment=environment,
        working_directory=root,
        mode="validate",
        expected_result=settings.grids_dir / f"{draft.project_name}_grid.png",
        attempt_id=attempt_id,
    )

    window.pages.setCurrentIndex(NAV_RUN)
    window._launch_pipeline_command(command, "Simulating validation")
    qtbot.waitUntil(
        lambda: (
            window.pipeline_process is not None
            and window.pipeline_process.state() == QProcess.ProcessState.NotRunning
            and window.retry_validation_button.isVisible()
        ),
        timeout=15_000,
    )

    assert window.run_stage_label.text() == (
        "LoRA installed; validation needs attention"
    )
    assert "will not repeat" in window.run_detail_label.text()
    assert window.retry_validation_button.isVisible()
    assert window.open_desktop_after_run_button.isVisible()
    assert not window.retry_after_run_button.isVisible()
    qtbot.mouseClick(
        window.open_desktop_after_run_button,
        Qt.MouseButton.LeftButton,
    )
    assert (
        opened_urls[-1]
        .toString()
        .startswith("https://docs.comfy.org/installation/desktop/windows")
    )
    recovery_file = settings.run_configs_dir / f"{draft.project_name}-recovery.json"
    assert recovery_file.is_file()
    window.close()
    QApplication.processEvents()

    restarted = MainWindow(
        settings=load_settings(settings_path),
        settings_path=settings_path,
        service=_OfflineLocalService(root=root),
    )
    qtbot.addWidget(restarted)
    restarted.show()

    assert restarted.pages.currentIndex() == NAV_RUN
    assert restarted.retry_validation_button.isVisible()
    assert restarted._pending_validation_recovery is not None
    assert restarted._pending_validation_recovery.lora_name == installed.name
    assert (
        restarted.service.validation_recovery_issue(
            restarted._collect_settings(),
            restarted._pending_validation_recovery,
        )
        is None
    )
    qtbot.waitUntil(
        lambda: restarted.retry_validation_button.isEnabled(),
        timeout=5_000,
    )
    qtbot.mouseClick(
        restarted.retry_validation_button,
        Qt.MouseButton.LeftButton,
    )
    qtbot.waitUntil(
        lambda: restarted.pipeline_command is not None,
        timeout=5_000,
    )
    assert restarted.pipeline_command is not None
    assert "--skip-prep" in restarted.pipeline_command.arguments
    assert "--skip-train" in restarted.pipeline_command.arguments
    assert (
        restarted.pipeline_command.arguments[
            restarted.pipeline_command.arguments.index("--lora") + 1
        ]
        == installed.name
    )
    assert (
        restarted.pipeline_command.environment["COMFY_SDXL_CKPT"]
        == "portraitXL.safetensors"
    )
    qtbot.waitUntil(
        lambda: (
            restarted.pipeline_process is not None
            and restarted.pipeline_process.state() == QProcess.ProcessState.NotRunning
        ),
        timeout=15_000,
    )


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


def test_prepared_caption_editor_saves_the_real_training_caption(
    qtbot,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (128, 128), (30, 80, 120)).save(dataset / "001.png")
    (dataset / "001.txt").write_text("old caption\n", encoding="utf-8")
    dialog = PreparedDatasetDialog(dataset, "ohwx person")
    qtbot.addWidget(dialog)
    dialog.show()

    dialog.caption_edit.setPlainText("ohwx person, smiling in a garden")
    qtbot.mouseClick(
        dialog.findChild(QPushButton, "savePreparedCaptionButton"),
        Qt.MouseButton.LeftButton,
    )

    assert (dataset / "001.txt").read_text(encoding="utf-8") == (
        "ohwx person, smiling in a garden\n"
    )


def test_prepared_review_creates_a_missing_caption_for_supported_image(
    qtbot,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (128, 128), (30, 80, 120)).save(dataset / "portrait.jpg")
    dialog = PreparedDatasetDialog(dataset, "ohwx person")
    qtbot.addWidget(dialog)
    dialog.show()

    assert "portrait.jpg" in dialog.file_issues_label.text()
    assert "portrait.txt" in dialog.file_issues_label.text()
    assert "Missing portrait.txt" in dialog.position_label.text()
    dialog.caption_edit.setPlainText("ohwx person, smiling in a garden")
    qtbot.mouseClick(
        dialog.findChild(QPushButton, "savePreparedCaptionButton"),
        Qt.MouseButton.LeftButton,
    )

    assert (dataset / "portrait.txt").read_text(encoding="utf-8") == (
        "ohwx person, smiling in a garden\n"
    )


def test_prepared_caption_editor_blocks_removing_the_identity_phrase(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (128, 128), (30, 80, 120)).save(dataset / "001.png")
    caption = dataset / "001.txt"
    caption.write_text("ohwx person, old caption\n", encoding="utf-8")
    prompts: list[tuple[str, str]] = []

    def cancel_save(_parent, title, message, *_args):
        prompts.append((title, message))
        return dataset_review.QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(
        dataset_review.QMessageBox,
        "question",
        cancel_save,
    )
    dialog = PreparedDatasetDialog(dataset, "ohwx person")
    qtbot.addWidget(dialog)
    dialog.show()

    dialog.caption_edit.setPlainText("smiling in a garden")
    dialog.caption_edit.document().setModified(True)
    qtbot.mouseClick(
        dialog.findChild(QPushButton, "savePreparedCaptionButton"),
        Qt.MouseButton.LeftButton,
    )

    assert caption.read_text(encoding="utf-8") == "ohwx person, old caption\n"
    assert prompts
    assert prompts[-1][0] == "Identity phrase is required"
    assert "ohwx person" in prompts[-1][1]
    assert dialog.caption_edit.document().isModified()
    dialog.caption_edit.document().setModified(False)
    dialog.close()


def test_prepared_caption_editor_can_restore_the_identity_phrase(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (128, 128), (30, 80, 120)).save(dataset / "001.png")
    caption = dataset / "001.txt"
    caption.write_text("ohwx person, old caption\n", encoding="utf-8")
    monkeypatch.setattr(
        dataset_review.QMessageBox,
        "question",
        lambda *_args: dataset_review.QMessageBox.StandardButton.Yes,
    )
    dialog = PreparedDatasetDialog(dataset, "ohwx person")
    qtbot.addWidget(dialog)
    dialog.show()

    dialog.caption_edit.setPlainText("smiling in a garden")
    qtbot.mouseClick(
        dialog.findChild(QPushButton, "savePreparedCaptionButton"),
        Qt.MouseButton.LeftButton,
    )

    assert caption.read_text(encoding="utf-8") == ("ohwx person, smiling in a garden\n")
    assert not dialog.caption_edit.document().isModified()


def test_prepared_caption_editor_prompts_before_unsaved_close(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (128, 128), (30, 80, 120)).save(dataset / "001.png")
    caption = dataset / "001.txt"
    caption.write_text("ohwx person, old caption\n", encoding="utf-8")
    answers = iter(
        (
            dataset_review.QMessageBox.StandardButton.Cancel,
            dataset_review.QMessageBox.StandardButton.Save,
        )
    )
    prompts: list[str] = []

    def answer_close(_parent, title, _message, _buttons, _default):
        prompts.append(title)
        return next(answers)

    monkeypatch.setattr(dataset_review.QMessageBox, "question", answer_close)
    dialog = PreparedDatasetDialog(dataset, "ohwx person")
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.caption_edit.setPlainText("ohwx person, new caption")
    dialog.caption_edit.document().setModified(True)

    dialog.close()
    QApplication.processEvents()
    assert dialog.isVisible()
    assert caption.read_text(encoding="utf-8") == "ohwx person, old caption\n"

    dialog.close()
    QApplication.processEvents()
    assert not dialog.isVisible()
    assert caption.read_text(encoding="utf-8") == "ohwx person, new caption\n"
    assert prompts == ["Save this caption?", "Save this caption?"]


def test_photo_review_names_the_exact_duplicate_for_a_casual_user(
    qtbot,
    tmp_path: Path,
) -> None:
    photos = tmp_path / "photos"
    _write_training_photos(photos, count=2)
    duplicate = photos / "duplicate.png"
    duplicate.write_bytes((photos / "photo_00.png").read_bytes())
    summary = app_service.inspect_source(photos, target_size=64)
    dialog = PhotoIssueDialog(summary)
    qtbot.addWidget(dialog)
    dialog.show()

    rows: list[tuple[str, str]] = []
    for row in range(dialog.issue_table.rowCount()):
        filename_item = dialog.issue_table.item(row, 1)
        detail_item = dialog.issue_table.item(row, 2)
        assert filename_item is not None
        assert detail_item is not None
        rows.append((filename_item.text(), detail_item.text()))

    assert any(
        {filename, detail.removeprefix("Exact copy of ").removesuffix(".")}
        == {"duplicate.png", "photo_00.png"}
        for filename, detail in rows
    )


def test_restore_uses_verified_result_record_instead_of_an_older_grid(
    qtbot,
    tmp_path: Path,
) -> None:
    datasets = tmp_path / "datasets"
    dataset = datasets / "portrait_v2"
    dataset.mkdir(parents=True)
    grids = tmp_path / "grids"
    grids.mkdir()
    Image.new("RGB", (32, 32), "red").save(grids / "portrait_v2_grid.png")
    result = RunResult(
        project_name="portrait_v2",
        mode="prepare",
        result_path=dataset,
        completed_at="2026-07-24T12:00:00+00:00",
        trigger="ohwx person",
        preset=TrainingPreset.BALANCED,
        result_fingerprint=artifact_fingerprint(dataset),
    )
    settings = AppSettings(
        datasets_dir=datasets,
        grids_dir=grids,
        last_project=ProjectDraft(
            project_name="portrait_v2",
            trigger="ohwx person",
        ),
        last_result=result,
    )

    window = MainWindow(
        settings=settings,
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()

    assert window._latest_result == dataset
    assert window.result_title_label.text() == "Your photos are ready for training"
    assert not window.review_prepared_button.isHidden()


def test_run_result_rejects_a_completion_record_from_another_attempt(
    qtbot,
    tmp_path: Path,
) -> None:
    grid = tmp_path / "grids" / "strict_v1_grid.png"
    grid.parent.mkdir()
    Image.new("RGB", (32, 32), "purple").save(grid)
    workflow = grid.parent / "strict_v1_use_in_comfyui.json"
    workflow.write_text("{}\n", encoding="utf-8")
    lora = tmp_path / "models" / "loras" / "strict_v1.safetensors"
    lora.parent.mkdir(parents=True)
    lora.write_bytes(b"lora weights")
    checkpoint = tmp_path / "models" / "checkpoints" / "portraitXL.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint weights")
    settings = AppSettings(
        comfy_host="127.0.0.1:8188",
        comfy_installation_id="desktop-a",
        comfy_lora_dir=lora.parent,
        comfy_checkpoint="portraitXL.safetensors",
        training_base_model=checkpoint,
        datasets_dir=tmp_path / "datasets",
        checkpoints_dir=tmp_path / "checkpoints",
        grids_dir=grid.parent,
        run_configs_dir=tmp_path / "configs",
    )
    draft = ProjectDraft(
        project_name="strict_v1",
        trigger="ohwx person",
        preset=TrainingPreset.BALANCED,
    )
    settings.run_configs_dir.mkdir()
    outcome_path = settings.run_configs_dir / "strict_v1-result.json"
    outcome = {
        "schema_version": 3,
        "project_name": draft.project_name,
        "attempt_id": "6e58ac26-21aa-4851-bc17-9800b1a5fa24",
        "mode": "full",
        "result_path": str(grid.resolve()),
        "result_fingerprint": artifact_fingerprint(grid),
        "trigger": draft.trigger,
        "preset": draft.preset.value,
        "checkpoint": settings.comfy_checkpoint,
        "lora_name": lora.name,
        "workflow_path": str(workflow.resolve()),
        "workflow_fingerprint": artifact_fingerprint(workflow),
        "comfy_installation_id": settings.comfy_installation_id,
        "comfy_host": settings.comfy_host,
        "installed_lora_path": str(lora.resolve()),
        "lora_fingerprint": artifact_fingerprint(lora),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_fingerprint": artifact_fingerprint(checkpoint),
        "starting_strength": 0.8,
        "completed_at": "2026-07-24T12:00:00+00:00",
    }
    outcome_path.write_text(json.dumps(outcome), encoding="utf-8")
    window = MainWindow(
        settings=settings,
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.draft = draft
    window._validated_draft = draft
    command = PipelineCommand(
        program=sys.executable,
        arguments=(),
        environment={
            "COMFY_SDXL_CKPT": settings.comfy_checkpoint,
            "COMFY_HOST": settings.comfy_host,
            "MIVIM_COMFY_INSTALLATION_ID": settings.comfy_installation_id,
        },
        working_directory=tmp_path,
        mode="full",
        expected_result=grid,
        attempt_id="0d6d97cf-7e1b-4f1d-a3eb-795153a05458",
    )

    with pytest.raises(ValueError, match="attempt identifier"):
        window._record_run_result(command)

    assert window.settings.last_result is None
    outcome["attempt_id"] = command.attempt_id
    outcome_path.write_text(json.dumps(outcome), encoding="utf-8")
    lora.write_bytes(b"substituted lora")
    with pytest.raises(ValueError, match="installed LoRA contents"):
        window._record_run_result(command)
    lora.write_bytes(b"lora weights")
    checkpoint.write_bytes(b"substituted checkpoint")
    with pytest.raises(ValueError, match="checkpoint contents"):
        window._record_run_result(command)
    checkpoint.write_bytes(b"checkpoint weights")
    record = window._record_run_result(command)
    assert record.workflow_path == workflow
    assert record.comfy_installation_id == "desktop-a"
    assert record.starting_strength == 0.8


def test_restore_ignores_a_result_whose_contents_changed(
    qtbot,
    tmp_path: Path,
) -> None:
    grid = tmp_path / "portrait_grid.png"
    Image.new("RGB", (32, 32), "blue").save(grid)
    record = RunResult(
        project_name="portrait_v1",
        mode="full",
        result_path=grid,
        completed_at="2026-07-24T12:00:00+00:00",
        trigger="ohwx person",
        preset=TrainingPreset.BALANCED,
        result_fingerprint=artifact_fingerprint(grid),
    )
    Image.new("RGB", (32, 32), "red").save(grid)

    window = MainWindow(
        settings=AppSettings(last_result=record),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)

    assert window._latest_result is None


def test_result_handoff_exposes_and_copies_the_exact_comfyui_setup(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    grid = tmp_path / "portrait_grid.png"
    Image.new("RGB", (64, 64), "blue").save(grid)
    workflow = tmp_path / "portrait_v1_use_in_comfyui.json"
    workflow.write_text('{"1": {"class_type": "LoraLoader"}}\n', encoding="utf-8")
    lora = tmp_path / "models" / "loras" / "portrait_v1_000002000.safetensors"
    lora.parent.mkdir(parents=True)
    lora.write_bytes(b"lora weights")
    checkpoint = tmp_path / "models" / "checkpoints" / "portraitXL.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint weights")
    record = RunResult(
        project_name="portrait_v1",
        mode="full",
        result_path=grid,
        completed_at="2026-07-24T12:00:00+00:00",
        trigger="ohwx person",
        preset=TrainingPreset.BALANCED,
        checkpoint="portraitXL.safetensors",
        lora_name="portrait_v1_000002000.safetensors",
        result_fingerprint=artifact_fingerprint(grid),
        workflow_path=workflow,
        workflow_fingerprint=artifact_fingerprint(workflow),
        comfy_installation_id="desktop-a",
        comfy_host="127.0.0.1:8188",
        installed_lora_path=lora,
        lora_fingerprint=artifact_fingerprint(lora),
        checkpoint_path=checkpoint,
        checkpoint_fingerprint=artifact_fingerprint(checkpoint),
        starting_strength=0.8,
    )
    window = MainWindow(
        settings=AppSettings(
            comfy_installation_id="desktop-a",
            comfy_lora_dir=lora.parent,
            comfy_checkpoint=checkpoint.name,
            training_base_model=checkpoint,
            last_result=record,
        ),
        settings_path=tmp_path / "settings.json",
        service=_FixedDiscoveryService(
            root=tmp_path,
            discovery=_result_desktop_discovery(tmp_path),
        ),
    )
    qtbot.addWidget(window)
    window.show()
    opened: list[Path] = []
    launched: list[list[str]] = []

    def capture_opened_path(url: QUrl) -> bool:
        opened.append(Path(url.toLocalFile()))
        return True

    monkeypatch.setattr(
        "src.desktop.QDesktopServices.openUrl",
        capture_opened_path,
    )
    monkeypatch.setattr("src.desktop.sys.platform", "win32")
    monkeypatch.setattr(
        "src.desktop.subprocess.Popen",
        lambda command, **_kwargs: launched.append(command),
    )
    qtbot.waitUntil(
        lambda: (
            window._verified_result_identity is not None
            and window._verified_result_identity[0] == record
            and window.discovery is not None
            and window.discovery.selected_installation is not None
        ),
        timeout=5_000,
    )
    window.comfy_status = ComfyApiStatus(
        True,
        "127.0.0.1:8299",
        (checkpoint.name,),
        (lora.name,),
        None,
    )
    assert window._result_comfy_context_issue(record) is None

    assert "portrait_v1_000002000.safetensors" in window.result_details_label.text()
    assert "ohwx person" in window.result_details_label.text()
    assert not window.open_full_grid_button.isHidden()
    qtbot.mouseClick(
        window.open_full_grid_button,
        Qt.MouseButton.LeftButton,
    )
    assert opened == [grid]
    assert not window.open_ready_workflow_button.isHidden()
    qtbot.mouseClick(
        window.open_ready_workflow_button,
        Qt.MouseButton.LeftButton,
    )
    assert opened == [grid]
    assert launched == [["explorer.exe", f"/select,{workflow.resolve()}"]]
    qtbot.mouseClick(
        window.copy_result_setup_button,
        Qt.MouseButton.LeftButton,
    )
    copied = QApplication.clipboard().text()
    assert "LoRA: portrait_v1_000002000.safetensors" in copied
    assert "Starting strength: 0.8" in copied
    assert str(workflow) in copied
    copy_reset_timers = [
        timer
        for timer in window.findChildren(QTimer)
        if timer.isSingleShot() and timer.interval() == 1800
    ]
    assert len(copy_reset_timers) == 1
    assert copy_reset_timers[0].parent() is window
    window.installation_combo.clear()
    window.installation_combo.addItem("Other Desktop", "desktop-b")
    assert "another Comfy Desktop installation" in (
        window._result_comfy_context_issue(record) or ""
    )
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )
    qtbot.mouseClick(
        window.open_ready_workflow_button,
        Qt.MouseButton.LeftButton,
    )
    assert opened == [grid]
    assert launched == [
        ["explorer.exe", f"/select,{workflow.resolve()}"],
        ["explorer.exe", f"/select,{workflow.resolve()}"],
    ]
    assert not errors


def test_result_adds_and_verifies_workflow_through_the_user_facing_button(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    grid = tmp_path / "portrait_grid.png"
    Image.new("RGB", (64, 64), "blue").save(grid)
    workflow = tmp_path / "portrait_v1_use_in_comfyui.json"
    workflow.write_text('{"version": 0.4, "nodes": []}\n', encoding="utf-8")
    lora = tmp_path / "models" / "loras" / "portrait.safetensors"
    lora.parent.mkdir(parents=True)
    lora.write_bytes(b"lora")
    checkpoint = tmp_path / "models" / "checkpoints" / "sdxl.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
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
        comfy_installation_id="desktop-a",
        comfy_installation_name="Portrait Desktop",
        comfy_host="127.0.0.1:8188",
        installed_lora_path=lora,
        lora_fingerprint=artifact_fingerprint(lora),
        checkpoint_path=checkpoint,
        checkpoint_fingerprint=artifact_fingerprint(checkpoint),
    )
    window = MainWindow(
        settings=AppSettings(
            comfy_installation_id="desktop-a",
            comfy_lora_dir=lora.parent,
            comfy_checkpoint=checkpoint.name,
            training_base_model=checkpoint,
            last_result=record,
        ),
        settings_path=tmp_path / "settings.json",
        service=_FixedDiscoveryService(
            root=tmp_path,
            discovery=_result_desktop_discovery(tmp_path),
        ),
    )
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(
        lambda: (
            window._verified_result_identity is not None
            and window._verified_result_identity[0] == record
            and window.discovery is not None
            and window.discovery.selected_installation is not None
        ),
        timeout=5_000,
    )
    profile = ComfyUserProfile("profile-a", "Portraits")
    other_profile = ComfyUserProfile("profile-b", "Other work")
    publication = WorkflowPublication(
        relative_path="workflows/MIVIM/portrait_v1_use_in_comfyui.json",
        profile=profile,
        fingerprint=record.workflow_fingerprint,
        created=True,
        renamed=False,
    )
    monkeypatch.setattr(
        "src.desktop.ComfyClient.list_user_profiles",
        lambda _client: (other_profile, profile),
    )
    monkeypatch.setattr(
        "src.desktop.QInputDialog.getItem",
        lambda *_args, **_kwargs: (profile.name, True),
    )
    monkeypatch.setattr(
        "src.desktop.ComfyClient.publish_workflow_bytes",
        lambda _client, _name, data, *, user_id, expected_fingerprint: (
            publication
            if user_id == profile.id
            and data == workflow.read_bytes()
            and expected_fingerprint == record.workflow_fingerprint
            else None
        ),
    )
    opened_desktop: list[bool] = []

    def open_desktop() -> None:
        opened_desktop.append(True)
        window._desktop_poll_remaining = 18

    monkeypatch.setattr(window, "open_comfy_desktop", open_desktop)

    assert window.publish_workflow_button.isEnabled()
    qtbot.mouseClick(
        window.publish_workflow_button,
        Qt.MouseButton.LeftButton,
    )
    assert opened_desktop == [True]
    assert window._pending_workflow_publish == record
    assert window.publish_workflow_button.text() == "Waiting for ComfyUI…"
    assert "Portrait Desktop" in window.workflow_handoff_status_label.text()

    window.comfy_status = ComfyApiStatus(
        True,
        "127.0.0.1:8299",
        (checkpoint.name,),
        (lora.name,),
        None,
    )
    window.add_ready_workflow_to_comfyui()
    qtbot.waitUntil(lambda: window._published_workflow is not None, timeout=5_000)

    published_record = window._displayed_result_record
    assert published_record is not None
    assert published_record.published_workflow_path == publication.relative_path
    assert published_record.workflow_user_id == profile.id
    assert window.settings.last_result == published_record
    assert window.settings.comfy_user_installation_id == "desktop-a"
    assert window.settings.comfy_user_id == profile.id
    assert window.publish_workflow_button.text() == "Open ComfyUI"
    assert "click Refresh in Workflows" in window.workflow_handoff_status_label.text()


def test_result_block_explains_missing_original_and_opens_connect(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings, _record, _workflow = _ready_full_result(tmp_path)
    replacement = _result_desktop_discovery(
        tmp_path / "replacement",
        installation_id="desktop-b",
        installation_name="Replacement Desktop",
    )
    window = MainWindow(
        settings=settings,
        settings_path=tmp_path / "settings.json",
        service=_FixedDiscoveryService(root=tmp_path, discovery=replacement),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)
    profile_requests: list[bool] = []
    monkeypatch.setattr(
        "src.desktop.ComfyClient.list_user_profiles",
        lambda _client: profile_requests.append(True),
    )

    assert window.publish_workflow_button.text() == "Fix in Connect"
    assert window.publish_workflow_button.isEnabled()
    guidance = window.workflow_handoff_status_label.text()
    assert "original Comfy Desktop installation" in guidance
    assert "no longer registered" in guidance

    qtbot.mouseClick(
        window.publish_workflow_button,
        Qt.MouseButton.LeftButton,
    )
    assert window.pages.currentIndex() == 0
    assert profile_requests == []


def test_result_check_files_again_revalidates_restored_model(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    window, record, _workflow, _service = _open_online_result_window(
        qtbot,
        tmp_path,
        monkeypatch,
    )
    assert record.installed_lora_path is not None
    record.installed_lora_path.write_bytes(b"changed weights")
    window._render_workflow_handoff_state(record)
    assert window.publish_workflow_button.text() == "Check files again"

    qtbot.mouseClick(
        window.publish_workflow_button,
        Qt.MouseButton.LeftButton,
    )
    qtbot.waitUntil(
        lambda: (
            window._result_identity_failure is not None
            and window._result_identity_failure[0] == record
        ),
        timeout=5_000,
    )
    assert window.publish_workflow_button.text() == "Check files again"

    record.installed_lora_path.write_bytes(b"lora")
    qtbot.mouseClick(
        window.publish_workflow_button,
        Qt.MouseButton.LeftButton,
    )
    qtbot.waitUntil(
        lambda: (
            window._verified_result_identity is not None
            and window._verified_result_identity[0] == record
            and window.publish_workflow_button.text() == "Add to ComfyUI"
        ),
        timeout=5_000,
    )


def test_workflow_retry_rediscovers_dynamic_port_and_reports_exact_match(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    window, record, workflow, service = _open_online_result_window(
        qtbot,
        tmp_path,
        monkeypatch,
    )
    profile = ComfyUserProfile("profile-a", "Portraits")
    profile_hosts: list[str] = []
    publish_hosts: list[str] = []
    errors: list[tuple[str, str]] = []

    def load_profiles(client) -> tuple[ComfyUserProfile, ...]:
        profile_hosts.append(client.host)
        if len(profile_hosts) == 1:
            raise RuntimeError("the old Desktop port closed")
        return (profile,)

    fingerprint = record.workflow_fingerprint.removeprefix("sha256:")
    publication = WorkflowPublication(
        relative_path=(
            f"workflows/MIVIM/portrait_v1_use_in_comfyui--{fingerprint}.json"
        ),
        profile=profile,
        fingerprint=record.workflow_fingerprint,
        created=False,
        renamed=False,
    )

    def publish(
        client,
        name: str,
        data: bytes,
        *,
        user_id: str,
        expected_fingerprint: str,
    ) -> WorkflowPublication:
        publish_hosts.append(client.host)
        assert name == workflow.name
        assert data == workflow.read_bytes()
        assert user_id == profile.id
        assert expected_fingerprint == record.workflow_fingerprint
        return publication

    monkeypatch.setattr("src.desktop.ComfyClient.list_user_profiles", load_profiles)
    monkeypatch.setattr("src.desktop.ComfyClient.publish_workflow_bytes", publish)
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )

    qtbot.mouseClick(
        window.publish_workflow_button,
        Qt.MouseButton.LeftButton,
    )
    qtbot.waitUntil(
        lambda: window._workflow_publish_retry_requires_refresh,
        timeout=5_000,
    )
    assert profile_hosts == ["127.0.0.1:8188"]
    assert errors

    service.fixed_discovery = replace(
        service.fixed_discovery,
        runtime_hints=(
            ComfyRuntimeHint(
                host="127.0.0.1:8299",
                installation_name="Portrait Desktop",
                pid=5678,
                timestamp=9_999_999_999_999,
            ),
        ),
    )
    qtbot.mouseClick(
        window.publish_workflow_button,
        Qt.MouseButton.LeftButton,
    )
    qtbot.waitUntil(lambda: window._published_workflow is not None, timeout=5_000)

    assert profile_hosts == ["127.0.0.1:8188", "127.0.0.1:8299"]
    assert publish_hosts == ["127.0.0.1:8299"]
    published_record = window._displayed_result_record
    assert published_record is not None
    assert published_record.workflow_user_name == profile.name
    assert window.settings.last_result == published_record
    receipt = window.workflow_handoff_status_label.text()
    assert "already available in Workflows → MIVIM" in receipt
    assert "profile Portraits" in receipt
    assert fingerprint not in receipt


def test_duplicate_profile_names_are_disambiguated_and_mapped_by_choice(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    window, record, _workflow, _service = _open_online_result_window(
        qtbot,
        tmp_path,
        monkeypatch,
    )
    first = ComfyUserProfile("personal", "Portraits")
    second = ComfyUserProfile("studio", "Portraits")
    selected_users: list[str] = []
    shown_choices: list[str] = []
    publication = WorkflowPublication(
        relative_path="workflows/MIVIM/portrait.json",
        profile=second,
        fingerprint=record.workflow_fingerprint,
        created=True,
        renamed=False,
    )

    monkeypatch.setattr(
        "src.desktop.ComfyClient.list_user_profiles",
        lambda _client: (first, second),
    )

    def choose_profile(*args, **_kwargs) -> tuple[str, bool]:
        choices = list(args[3])
        shown_choices.extend(choices)
        return choices[1], True

    def publish(
        _client,
        _name: str,
        _data: bytes,
        *,
        user_id: str,
        expected_fingerprint: str,
    ) -> WorkflowPublication:
        assert expected_fingerprint == record.workflow_fingerprint
        selected_users.append(user_id)
        return publication

    monkeypatch.setattr("src.desktop.QInputDialog.getItem", choose_profile)
    monkeypatch.setattr("src.desktop.ComfyClient.publish_workflow_bytes", publish)

    qtbot.mouseClick(
        window.publish_workflow_button,
        Qt.MouseButton.LeftButton,
    )
    qtbot.waitUntil(lambda: window._published_workflow is not None, timeout=5_000)

    assert shown_choices == ["Portraits — personal", "Portraits — studio"]
    assert selected_users == ["studio"]
    assert window.settings.comfy_user_id == "studio"


def test_profile_callback_from_an_old_runtime_generation_is_ignored(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    window, _record, _workflow, _service = _open_online_result_window(
        qtbot,
        tmp_path,
        monkeypatch,
    )
    started = threading.Event()
    release = threading.Event()
    publish_calls: list[bool] = []

    def delayed_profiles(_client) -> tuple[ComfyUserProfile, ...]:
        started.set()
        assert release.wait(timeout=5)
        return (ComfyUserProfile("default", "Default"),)

    monkeypatch.setattr(
        "src.desktop.ComfyClient.list_user_profiles",
        delayed_profiles,
    )
    monkeypatch.setattr(
        "src.desktop.ComfyClient.publish_workflow_bytes",
        lambda *_args, **_kwargs: publish_calls.append(True),
    )

    qtbot.mouseClick(
        window.publish_workflow_button,
        Qt.MouseButton.LeftButton,
    )
    qtbot.waitUntil(started.is_set, timeout=5_000)
    window.refresh_setup()
    release.set()
    qtbot.waitUntil(
        lambda: window.comfy_badge.text() != "CHECKING",
        timeout=5_000,
    )
    qtbot.wait(100)

    assert publish_calls == []
    assert window._published_workflow is None
    assert window.publish_workflow_button.text() != "Open ComfyUI"


def test_upload_receipt_from_an_old_runtime_generation_is_ignored(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    window, record, _workflow, _service = _open_online_result_window(
        qtbot,
        tmp_path,
        monkeypatch,
    )
    profile = ComfyUserProfile("default", "Default")
    started = threading.Event()
    release = threading.Event()
    publication = WorkflowPublication(
        relative_path="workflows/MIVIM/portrait.json",
        profile=profile,
        fingerprint=record.workflow_fingerprint,
        created=True,
        renamed=False,
    )

    monkeypatch.setattr(
        "src.desktop.ComfyClient.list_user_profiles",
        lambda _client: (profile,),
    )

    def delayed_publish(*_args, **_kwargs) -> WorkflowPublication:
        started.set()
        assert release.wait(timeout=5)
        return publication

    monkeypatch.setattr(
        "src.desktop.ComfyClient.publish_workflow_bytes",
        delayed_publish,
    )

    qtbot.mouseClick(
        window.publish_workflow_button,
        Qt.MouseButton.LeftButton,
    )
    qtbot.waitUntil(started.is_set, timeout=5_000)
    window.refresh_setup()
    release.set()
    qtbot.waitUntil(
        lambda: window.comfy_badge.text() != "CHECKING",
        timeout=5_000,
    )
    qtbot.wait(100)

    assert window._published_workflow is None
    assert window.settings.last_result is not None
    assert not window.settings.last_result.published_workflow_path
    assert window.publish_workflow_button.text() != "Open ComfyUI"


def test_release_version_matches_project_and_is_visible(
    qtbot,
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    project = tomllib.loads((repository / "pyproject.toml").read_text(encoding="utf-8"))
    deploy_spec = (repository / "pysidedeploy.spec").read_text(encoding="utf-8")
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)

    assert project["project"]["version"] == __version__
    assert f"--file-version={__version__}.0" in deploy_spec
    assert f"--product-version={__version__}.0" in deploy_spec
    assert __version__ in window.windowTitle()


def test_result_preview_fits_a_wide_grid_without_clipping(
    qtbot,
    tmp_path: Path,
) -> None:
    grid = tmp_path / "wide_grid.png"
    Image.new("RGB", (1200, 400), "navy").save(grid)
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.resize(1040, 690)
    window.show()
    window._show_result(grid)
    window.show_page(NAV_RESULTS)
    qtbot.waitUntil(lambda: window.result_image.pixmap() is not None)

    preview = window.result_image.pixmap()
    assert preview is not None
    viewport = window.result_image.contentsRect().size()
    assert preview.width() <= viewport.width()
    assert preview.height() <= viewport.height()
    assert preview.width() / preview.height() == pytest.approx(3.0, abs=0.02)
    assert window.result_path_label.text() == str(grid)


def test_screenshot_cli_selects_a_real_application_page() -> None:
    args = _build_parser().parse_args(
        ["--screenshot", "capture.png", "--screenshot-page", "result"]
    )

    assert args.screenshot == Path("capture.png")
    assert args.screenshot_page == "result"


def test_new_project_invalidates_stale_result_identity_state(
    qtbot,
    tmp_path: Path,
) -> None:
    result = RunResult(
        project_name="portrait_v1",
        mode="full",
        result_path=tmp_path / "grid.png",
        completed_at="2026-07-24T12:00:00+00:00",
        trigger="ohwx person",
        preset=TrainingPreset.BALANCED,
    )
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.draft = ProjectDraft(project_name="portrait_v1")
    window._latest_result = result.result_path
    window._displayed_result_record = result
    window._verified_result_identity = (result, (1, 2), (3, 4))
    window._result_identity_failure = (result, "old verification")
    window._run_settings_snapshot = AppSettings()
    window._run_draft_snapshot = window.draft
    result_request = window._result_identity_request

    window.new_project()

    assert window._result_identity_request == result_request + 1
    assert window._latest_result is None
    assert window._displayed_result_record is None
    assert window._verified_result_identity is None
    assert window._result_identity_failure is None
    assert window._run_settings_snapshot is None
    assert window._run_draft_snapshot is None


def test_unexpected_recovery_verifier_output_never_enables_retry(
    qtbot,
    tmp_path: Path,
) -> None:
    window = MainWindow(
        settings=AppSettings(),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    recovery = ValidationRecovery(
        project_name="portrait_v1",
        state="validation_pending",
        lora_name="portrait_v1.safetensors",
        installed_lora_path=tmp_path / "portrait_v1.safetensors",
        expected_grid_path=tmp_path / "portrait_v1_grid.png",
        trigger="ohwx person",
        preset=TrainingPreset.BALANCED,
        checkpoint="sdxl.safetensors",
        updated_at="2026-07-24T12:00:00+00:00",
    )
    window._pending_validation_recovery = recovery
    request = window._recovery_check_request

    window._restored_recovery_checked(
        request,
        recovery,
        ProjectDraft(
            project_name="portrait_v1",
            trigger="ohwx person",
        ),
        AppSettings(),
        object(),
    )

    assert window.pipeline_command is None
    assert not window.retry_validation_button.isEnabled()
    assert "unexpected result" in window.run_detail_label.text()


def test_checkpoint_comparison_shows_same_prompt_and_selects_earlier_weights(
    qtbot, tmp_path: Path
) -> None:
    earlier = tmp_path / "earlier.jpg"
    final = tmp_path / "final.jpg"
    Image.new("RGB", (256, 512), "red").save(earlier)
    Image.new("RGB", (256, 512), "blue").save(final)
    choices = (
        TrainingCheckpoint(
            tmp_path / "early.safetensors", 250, False, "a", ((0, earlier),)
        ),
        TrainingCheckpoint(
            tmp_path / "final.safetensors", 500, True, "b", ((0, final), (1, final))
        ),
    )
    dialog = CheckpointReviewDialog(choices)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    assert dialog.selected_checkpoint == choices[0]
    assert dialog.reference_combo.currentIndex() == 1
    assert dialog.images[0].pixmap().toImage().pixelColor(5, 5).blue() > 200
    assert dialog.images[1].pixmap().toImage().pixelColor(5, 5).red() > 200
    dialog.prompt_combo.setCurrentIndex(1)
    assert "No saved preview" in dialog.images[1].text()
    qtbot.mouseClick(dialog.validate_button, Qt.MouseButton.LeftButton)
    assert dialog.result() == CheckpointReviewDialog.DialogCode.Accepted
    assert dialog.selected_checkpoint.path.name == "early.safetensors"


def test_results_checkpoint_action_launches_selected_validation_and_saves_recovery_project(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow(
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    _wait_for_initial_setup(qtbot, window)
    record = RunResult(
        project_name="person",
        mode="full",
        result_path=tmp_path / "grid.png",
        completed_at="now",
        trigger="ohwx person",
        preset=TrainingPreset.BALANCED,
    )
    Image.new("RGB", (32, 32)).save(record.result_path)
    window._show_result(record.result_path, record)
    window.pages.setCurrentIndex(NAV_RESULTS)
    window.show()
    choice = TrainingCheckpoint(
        tmp_path / "early.safetensors", 250, False, "identity", ()
    )
    draft = ProjectDraft(project_name="person_step250_version", trigger=record.trigger)
    command = _script_pipeline_command(tmp_path, "pass")
    command = replace(command, mode="validate")
    captured: list[tuple[PipelineCommand, str]] = []
    monkeypatch.setattr(
        "src.desktop.discover_training_checkpoints", lambda *_args: (choice,)
    )
    monkeypatch.setattr(
        CheckpointReviewDialog,
        "exec",
        lambda _self: CheckpointReviewDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(window, "_result_comfy_context_issue", lambda _record: None)
    monkeypatch.setattr(
        window.service,
        "build_checkpoint_validation_command",
        lambda _settings, _record, selected: (
            (draft, command) if selected == choice else pytest.fail("wrong choice")
        ),
    )
    monkeypatch.setattr(
        window,
        "_launch_pipeline_command",
        lambda cmd, label: captured.append((cmd, label)),
    )
    assert window.compare_checkpoints_button.isVisible()
    qtbot.mouseClick(window.compare_checkpoints_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: bool(captured), timeout=5000)
    assert captured == [(command, "Validating training step 250")]
    assert window._validated_draft == draft
    assert (
        load_settings(window.settings_path).last_project.project_name
        == draft.project_name
    )
