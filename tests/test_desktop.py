from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QProcess, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
)
from pytestqt.exceptions import TimeoutError as QtBotTimeout

from src import app_service, dataset_review
from src.app_models import AppSettings, ProjectDraft, RunResult, TrainingPreset
from src.app_service import ApplicationService, PipelineCommand
from src.app_settings import load_settings, save_settings
from src.comfy_desktop import (
    ComfyApiStatus,
    ComfyDesktopDiscovery,
    ComfyInstallation,
    ComfyLauncher,
    ComfyModelRoot,
)
from src.dataset_review import PreparedDatasetDialog
from src.desktop import NAV_RESULTS, NAV_RUN, MainWindow
from src.photo_review import PhotoIssueDialog


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
    assert collision.action == "Review prepared files"
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
    assert lora_item.action == "Create LoRA folder"
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
    qtbot.mouseClick(
        window.use_fast_after_run_button,
        Qt.MouseButton.LeftButton,
    )
    assert window.preset_combo.currentData() == TrainingPreset.FAST.value
    assert window.overwrite_checkbox.isChecked()
    assert window.run_stack.currentIndex() == 0
    qtbot.waitUntil(lambda: not window._workers, timeout=10_000)


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
    draft = ProjectDraft(
        project_name="restart_recovery_v1",
        trigger="ohwx person",
        preset=TrainingPreset.BALANCED,
    )
    settings = AppSettings(
        comfy_host="127.0.0.1:1",
        comfy_lora_dir=lora_dir,
        comfy_checkpoint="portraitXL.safetensors",
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
    command = PipelineCommand(
        program=sys.executable,
        arguments=(
            "-m",
            "src.pipeline",
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
        ),
        environment=environment,
        working_directory=root,
        mode="validate",
        expected_result=settings.grids_dir / f"{draft.project_name}_grid.png",
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
    changed_settings = load_settings(settings_path)
    changed_settings.comfy_checkpoint = "different-after-training.safetensors"
    save_settings(changed_settings, settings_path)

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
    qtbot.mouseClick(
        restarted.retry_validation_button,
        Qt.MouseButton.LeftButton,
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


def test_result_handoff_exposes_and_copies_the_exact_comfyui_setup(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    grid = tmp_path / "portrait_grid.png"
    Image.new("RGB", (64, 64), "blue").save(grid)
    record = RunResult(
        project_name="portrait_v1",
        mode="full",
        result_path=grid,
        completed_at="2026-07-24T12:00:00+00:00",
        trigger="ohwx person",
        preset=TrainingPreset.BALANCED,
        checkpoint="portraitXL.safetensors",
        lora_name="portrait_v1_000002000.safetensors",
    )
    window = MainWindow(
        settings=AppSettings(last_result=record),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=tmp_path),
    )
    qtbot.addWidget(window)
    window.show()
    opened: list[Path] = []

    def capture_opened_path(url: QUrl) -> bool:
        opened.append(Path(url.toLocalFile()))
        return True

    monkeypatch.setattr(
        "src.desktop.QDesktopServices.openUrl",
        capture_opened_path,
    )

    assert "portrait_v1_000002000.safetensors" in window.result_details_label.text()
    assert "ohwx person" in window.result_details_label.text()
    assert not window.open_full_grid_button.isHidden()
    qtbot.mouseClick(
        window.open_full_grid_button,
        Qt.MouseButton.LeftButton,
    )
    assert opened == [grid]
    qtbot.mouseClick(
        window.copy_result_setup_button,
        Qt.MouseButton.LeftButton,
    )
    copied = QApplication.clipboard().text()
    assert "LoRA: portrait_v1_000002000.safetensors" in copied
    assert "Starting strength: 0.8" in copied
