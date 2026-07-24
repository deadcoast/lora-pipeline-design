"""MIVIM LoRA Studio — a guided Windows companion for Comfy Desktop.

Run with:

    python -m src.desktop

The GUI launches the same supported pipeline entry point as the CLI, streams
its real output, and supplies settings through the child environment.  Users do
not need to edit ``.env`` or understand its implementation details.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from PySide6.QtCore import (
    QElapsedTimer,
    QObject,
    QProcess,
    QProcessEnvironment,
    QRunnable,
    QSignalBlocker,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QIcon,
    QPainter,
    QPixmap,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import config
from .app_models import (
    AppSettings,
    ProjectDraft,
    RunResult,
    TrainingPreset,
    ValidationRecovery,
)
from .app_service import (
    ApplicationService,
    PipelineCommand,
    ReadinessItem,
    ReadinessReport,
    SourceSummary,
    inspect_source,
    stage_from_log,
    verified_missing_lora_directory,
)
from .app_settings import (
    default_settings_path,
    load_settings,
    save_settings,
    with_user_storage_defaults,
)
from .comfy_desktop import (
    ComfyApiStatus,
    ComfyDesktopDiscovery,
)
from .dataset_review import PreparedDatasetDialog
from .desktop_theme import APP_STYLE
from .photo_review import PhotoIssueDialog
from .studio_logging import StudioLog, log_directory_for

NAV_SETUP, NAV_PROJECT, NAV_RUN, NAV_RESULTS = range(4)
RunFailureKind = Literal["generic", "oom", "trainer", "validation"]


def _artifact_snapshot(path: Path) -> tuple[tuple[str, int, int], ...] | None:
    """Return a cheap identity for verifying that this run produced fresh output."""

    if not path.exists():
        return None
    files = (
        [path]
        if path.is_file()
        else sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    )
    snapshot: list[tuple[str, int, int]] = []
    try:
        for candidate in files:
            stat = candidate.stat()
            name = "." if candidate == path else candidate.relative_to(path).as_posix()
            snapshot.append((name, stat.st_size, stat.st_mtime_ns))
    except OSError:
        return None
    return tuple(snapshot)


def _layout_clear(layout: QVBoxLayout | QGridLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            continue
        child_layout = item.layout()
        widget = item.widget()
        if child_layout is not None:
            _layout_clear(child_layout)  # type: ignore[arg-type]
        if widget is not None:
            widget.deleteLater()


def _card(parent: QWidget | None = None, *, hero: bool = False) -> QFrame:
    frame = QFrame(parent)
    frame.setObjectName("heroCard" if hero else "card")
    return frame


def _title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("cardTitle")
    return label


def _body(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("cardBody")
    label.setWordWrap(True)
    return label


def _hint(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("hintText")
    label.setWordWrap(True)
    return label


def _button(text: str, object_name: str = "") -> QPushButton:
    button = QPushButton(text)
    if object_name:
        button.setObjectName(object_name)
    return button


def _page_header(eyebrow: str, title: str, subtitle: str) -> QVBoxLayout:
    layout = QVBoxLayout()
    layout.setSpacing(5)
    eyebrow_label = QLabel(eyebrow.upper())
    eyebrow_label.setObjectName("eyebrow")
    title_label = QLabel(title)
    title_label.setObjectName("pageTitle")
    subtitle_label = QLabel(subtitle)
    subtitle_label.setObjectName("pageSubtitle")
    subtitle_label.setWordWrap(True)
    layout.addWidget(eyebrow_label)
    layout.addWidget(title_label)
    layout.addWidget(subtitle_label)
    return layout


def _status_badge(text: str, state: str) -> QLabel:
    label = QLabel(text)
    names = {
        "ready": "statusReady",
        "warning": "statusWarning",
        "error": "statusError",
    }
    label.setObjectName(names.get(state, "statusWarning"))
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return label


def _app_icon() -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#2f80d1"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(4, 4, 56, 56, 15, 15)
    painter.setPen(QColor("#ffffff"))
    painter.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "M")
    painter.end()
    return QIcon(pixmap)


class WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str, str)


class FunctionWorker(QRunnable):
    """Run a short blocking service operation without freezing the GUI."""

    def __init__(self, function: Callable[[], object]) -> None:
        super().__init__()
        self.function = function
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self.function()
        except Exception as exc:  # noqa: BLE001 - errors are rendered in the UI
            message = str(exc).strip() or type(exc).__name__
            technical_details = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            try:
                self.signals.failed.emit(message, technical_details)
            except RuntimeError:
                pass  # The application closed while this short check was finishing.
        else:
            try:
                self.signals.finished.emit(result)
            except RuntimeError:
                pass  # The application closed while this short check was finishing.


class PhotoDropZone(QFrame):
    folder_selected = Signal(Path)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("dropZone")
        self.setProperty("hasPath", False)
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)
        self.setAccessibleName("Training photo folder drop area")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label = QLabel("＋")
        self.icon_label.setStyleSheet(
            "font-size: 28px; color: #73bfff; font-weight: 300;"
        )
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label = QLabel("Choose a folder of training photos")
        self.title_label.setObjectName("dropTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_label = _hint(
            "Drop a folder here, or browse. Use 15–40 clear, varied photos of one person."
        )
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_label.setMaximumWidth(580)
        self.browse_button = _button("Browse photos", "primaryButton")
        self.browse_button.setObjectName("choosePhotosButton")
        self.browse_button.clicked.connect(self._browse)
        layout.addWidget(self.icon_label)
        layout.addWidget(self.title_label)
        layout.addWidget(self.detail_label)
        layout.addSpacing(5)
        layout.addWidget(self.browse_button, alignment=Qt.AlignmentFlag.AlignHCenter)

    def _browse(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose training photo folder",
            str(Path.home() / "Pictures"),
            QFileDialog.Option.ShowDirsOnly,
        )
        if selected:
            self.set_folder(Path(selected))

    def set_folder(self, folder: Path) -> None:
        self.setProperty("hasPath", True)
        self.style().unpolish(self)
        self.style().polish(self)
        self.title_label.setText(folder.name or str(folder))
        self.detail_label.setText(str(folder))
        self.icon_label.setText("✓")
        self.folder_selected.emit(folder)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        urls = event.mimeData().urls()
        if len(urls) == 1 and Path(urls[0].toLocalFile()).is_dir():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if len(urls) == 1:
            folder = Path(urls[0].toLocalFile())
            if folder.is_dir():
                self.set_folder(folder)
                event.acceptProposedAction()


class DiagnosticCard(QFrame):
    def __init__(
        self,
        item: ReadinessItem,
        action_callback: Callable[[ReadinessItem], None] | None = None,
    ) -> None:
        super().__init__()
        self.setObjectName("card")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(13)

        marker = QLabel({"ready": "✓", "warning": "!", "error": "×"}[item.state])
        marker.setFixedSize(32, 32)
        marker.setAlignment(Qt.AlignmentFlag.AlignCenter)
        colors = {
            "ready": ("#123c35", "#82f3d0"),
            "warning": ("#493716", "#fcd68a"),
            "error": ("#4b2129", "#ffb4c0"),
        }
        background, foreground = colors[item.state]
        marker.setStyleSheet(
            f"background: {background}; color: {foreground}; "
            "border-radius: 16px; font-size: 17px; font-weight: 700;"
        )
        text = QVBoxLayout()
        text.setSpacing(3)
        heading = _title(item.title)
        detail = _body(item.detail)
        text.addWidget(heading)
        text.addWidget(detail)
        layout.addWidget(marker, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addLayout(text, 1)
        if item.action:
            if action_callback is None:
                action: QWidget = _hint(item.action)
            else:
                button = _button(item.action)
                button.setAccessibleName(f"{item.title}: {item.action}")
                button.clicked.connect(lambda: action_callback(item))
                action = button
            action.setMaximumWidth(260)
            layout.addWidget(action)


class MainWindow(QMainWindow):
    """Accessible, guided shell around the real LoRA pipeline."""

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        settings_path: Path | None = None,
        service: ApplicationService | None = None,
    ) -> None:
        super().__init__()
        self.settings_path = settings_path or default_settings_path()
        loaded_settings = settings or load_settings(self.settings_path)
        self.settings = with_user_storage_defaults(
            loaded_settings,
            self.settings_path,
        )
        self.studio_log = StudioLog(log_directory_for(self.settings_path))
        self.studio_log.info(
            "session.started",
            "MIVIM LoRA Studio started. Support logs stay on this computer.",
        )
        self.service = service or ApplicationService()
        self.discovery: ComfyDesktopDiscovery | None = None
        self.comfy_status: ComfyApiStatus | None = None
        self.draft: ProjectDraft | None = None
        self.readiness_report: ReadinessReport | None = None
        self.source_folder: Path | None = None
        self.source_summary: SourceSummary | None = None
        self.pipeline_process: QProcess | None = None
        self.pipeline_command: PipelineCommand | None = None
        self.pipeline_pid: int | None = None
        self.run_cancelled = False
        self._run_terminal_handled = False
        self._result_before: tuple[tuple[str, int, int], ...] | None = None
        self._installed_lora_name: str | None = None
        self.thread_pool = QThreadPool.globalInstance()
        self._workers: list[FunctionWorker] = []
        self._latest_result: Path | None = None
        self._displayed_result_record: RunResult | None = None
        self._pending_validation_recovery: ValidationRecovery | None = None
        self._recovery_before_run: ValidationRecovery | None = None
        self._pipeline_line_buffer = ""
        self._pipeline_recent_output = ""
        self._last_determinate_progress = 0
        self._training_progress_active = False
        self._training_elapsed_clock = QElapsedTimer()
        self._training_elapsed_timer = QTimer(self)
        self._training_elapsed_timer.setInterval(250)
        self._training_elapsed_timer.timeout.connect(self._update_training_elapsed)
        self._trainer_stage_seen = False
        self._run_failure_kind: RunFailureKind = "generic"
        self._closing = False
        self._validated_settings: AppSettings | None = None
        self._validated_draft: ProjectDraft | None = None
        self._setup_request = 0
        self._source_request = 0
        self._readiness_request = 0
        self._desktop_poll_remaining = 0

        self.setWindowTitle("MIVIM LoRA Studio")
        self.setWindowIcon(_app_icon())
        self.setMinimumSize(1040, 690)
        self.resize(1180, 780)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        self._populate_settings()
        self._populate_last_project()
        self._restore_last_result()
        self._restore_validation_recovery()
        self._connect_form_invalidation()
        QTimer.singleShot(80, self.refresh_setup)

    # ------------------------------------------------------------------ shell
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        navigation = QWidget()
        navigation.setObjectName("navigation")
        navigation.setFixedWidth(220)
        nav = QVBoxLayout(navigation)
        nav.setContentsMargins(18, 24, 18, 20)
        nav.setSpacing(8)
        brand = QLabel("MIVIM")
        brand.setObjectName("brandMark")
        subtitle = QLabel("LoRA Studio")
        subtitle.setObjectName("brandSubtitle")
        nav.addWidget(brand)
        nav.addWidget(subtitle)
        nav.addSpacing(26)
        self.nav_buttons: list[QPushButton] = []
        for index, text in enumerate(
            ("1  Connect", "2  New project", "3  Review & run", "4  Result")
        ):
            button = _button(text, "navButton")
            button.setCheckable(True)
            button.setEnabled(index <= NAV_PROJECT)
            button.clicked.connect(lambda _checked, page=index: self.show_page(page))
            nav.addWidget(button)
            self.nav_buttons.append(button)
        nav.addStretch(1)
        privacy = _hint(
            "Photos and models stay on this computer unless a tool you choose says otherwise."
        )
        privacy.setMaximumWidth(180)
        nav.addWidget(privacy)
        version = QLabel("MIVIM • Phase 1 SDXL")
        version.setObjectName("brandSubtitle")
        nav.addWidget(version)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_setup_page())
        self.pages.addWidget(self._build_project_page())
        self.pages.addWidget(self._build_run_page())
        self.pages.addWidget(self._build_results_page())
        shell.addWidget(navigation)
        shell.addWidget(self.pages, 1)
        self.show_page(NAV_SETUP)

    def _scroll_page(self, content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll

    def show_page(self, index: int) -> None:
        if not 0 <= index < self.pages.count():
            return
        if not self.nav_buttons[index].isEnabled():
            return
        if index == NAV_RUN and self.pages.currentIndex() != NAV_RUN:
            self.review_project()
            return
        self.pages.setCurrentIndex(index)
        for position, button in enumerate(self.nav_buttons):
            button.setChecked(position == index)

    # --------------------------------------------------------------- setup UI
    def _build_setup_page(self) -> QWidget:
        content = QWidget()
        page = QVBoxLayout(content)
        page.setContentsMargins(34, 30, 34, 34)
        page.setSpacing(18)
        page.addLayout(
            _page_header(
                "Connection",
                "Connect the tools once",
                "The studio looks for Comfy Desktop and keeps advanced paths out of "
                "your normal workflow.",
            )
        )

        hero = _card(hero=True)
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(20, 18, 20, 18)
        hero_text = QVBoxLayout()
        hero_text.setSpacing(5)
        hero_text.addWidget(_title("Comfy Desktop"))
        self.comfy_summary_label = _body("Looking for the Windows desktop app…")
        hero_text.addWidget(self.comfy_summary_label)
        self.installation_label = _hint("Desktop instance")
        self.installation_combo = QComboBox()
        self.installation_combo.setObjectName("comfyInstallationCombo")
        self.installation_combo.setAccessibleName("Comfy Desktop instance")
        self.installation_label.setVisible(False)
        self.installation_combo.setVisible(False)
        hero_text.addWidget(self.installation_label)
        hero_text.addWidget(self.installation_combo)
        hero_layout.addLayout(hero_text, 1)
        self.comfy_badge = _status_badge("CHECKING", "warning")
        hero_layout.addWidget(self.comfy_badge)
        self.open_desktop_button = _button("Open Comfy Desktop")
        self.open_desktop_button.setObjectName("openComfyDesktopButton")
        self.open_desktop_button.clicked.connect(self.open_comfy_desktop)
        hero_layout.addWidget(self.open_desktop_button)
        self.retry_comfy_button = _button("Retry", "primaryButton")
        self.retry_comfy_button.setObjectName("retryComfyButton")
        self.retry_comfy_button.clicked.connect(self.refresh_setup)
        hero_layout.addWidget(self.retry_comfy_button)
        self.setup_logs_button = _button("Open local logs")
        self.setup_logs_button.setObjectName("setupOpenLogsButton")
        self.setup_logs_button.setAccessibleName(
            "Open local support logs for this setup problem"
        )
        self.setup_logs_button.clicked.connect(self.open_logs_folder)
        self.setup_logs_button.setVisible(False)
        hero_layout.addWidget(self.setup_logs_button)
        page.addWidget(hero)

        self.advanced_setup_toggle = QCheckBox("Show advanced paths and ports")
        self.advanced_setup_toggle.setObjectName("advancedSetupToggle")
        self.advanced_setup_toggle.setAccessibleName(
            "Show technical setup paths and custom ComfyUI port"
        )
        page.addWidget(self.advanced_setup_toggle)

        form_grid = QGridLayout()
        form_grid.setHorizontalSpacing(16)
        form_grid.setVerticalSpacing(16)

        model_card = _card()
        model_layout = QVBoxLayout(model_card)
        model_layout.setContentsMargins(18, 17, 18, 18)
        model_layout.setSpacing(9)
        model_layout.addWidget(_title("SDXL base model"))
        model_layout.addWidget(
            _body(
                "Use the same local checkpoint for training and consistent "
                "before-and-after comparisons."
            )
        )
        checkpoint_label = _hint("SDXL checkpoint")
        model_layout.addWidget(checkpoint_label)
        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.setObjectName("checkpointCombo")
        self.checkpoint_combo.setEditable(True)
        self.checkpoint_combo.setAccessibleName("SDXL validation checkpoint")
        model_layout.addWidget(self.checkpoint_combo)
        self.sdxl_confirm_checkbox = QCheckBox(
            "This is an SDXL base checkpoint (not a refiner)"
        )
        self.sdxl_confirm_checkbox.setObjectName("confirmSdxlCheckpoint")
        self.sdxl_confirm_checkbox.setAccessibleName(
            "Confirm the selected checkpoint is an SDXL base model"
        )
        self.sdxl_confirm_checkbox.setToolTip(
            "Check the model's download page if you are unsure."
        )
        model_layout.addWidget(self.sdxl_confirm_checkbox)
        self.base_model_path_label = _hint("Local training model file")
        model_layout.addWidget(self.base_model_path_label)
        self.base_model_edit = QLineEdit()
        self.base_model_edit.setObjectName("trainingBaseModelEdit")
        self.base_model_edit.setPlaceholderText(
            "Local checkpoint path is filled from Comfy Desktop"
        )
        self.base_model_edit.setAccessibleName("Local SDXL training checkpoint file")
        base_model_row = QHBoxLayout()
        base_model_row.addWidget(self.base_model_edit, 1)
        self.base_model_browse_button = _button("Browse")
        self.base_model_browse_button.setObjectName("browseBaseModelButton")
        self.base_model_browse_button.setAccessibleName(
            "Browse for local SDXL checkpoint file"
        )
        self.base_model_browse_button.clicked.connect(self.choose_base_model)
        base_model_row.addWidget(self.base_model_browse_button)
        model_layout.addLayout(base_model_row)
        self.model_hint = _hint("Models appear after a ComfyUI instance is running.")
        model_layout.addWidget(self.model_hint)
        form_grid.addWidget(model_card, 0, 0)

        trainer_card = _card()
        trainer_layout = QVBoxLayout(trainer_card)
        trainer_layout.setContentsMargins(18, 17, 18, 18)
        trainer_layout.setSpacing(9)
        trainer_layout.addWidget(_title("LoRA trainer"))
        trainer_layout.addWidget(
            _body("ai-toolkit does the GPU training in its own isolated environment.")
        )
        self.toolkit_dir_edit = QLineEdit()
        self.toolkit_dir_edit.setObjectName("toolkitDirEdit")
        self.toolkit_dir_edit.setPlaceholderText("Not connected")
        self.toolkit_dir_edit.setAccessibleName("ai-toolkit folder")
        toolkit_row = QHBoxLayout()
        toolkit_row.addWidget(self.toolkit_dir_edit, 1)
        self.toolkit_browse_button = _button("Browse")
        self.toolkit_browse_button.setObjectName("browseToolkitButton")
        self.toolkit_browse_button.setAccessibleName("Browse for ai-toolkit folder")
        self.toolkit_browse_button.clicked.connect(self.choose_toolkit_dir)
        toolkit_row.addWidget(self.toolkit_browse_button)
        trainer_layout.addLayout(toolkit_row)
        self.toolkit_python_edit = QLineEdit()
        self.toolkit_python_edit.setObjectName("toolkitPythonEdit")
        self.toolkit_python_edit.setPlaceholderText(
            "The trainer's venv Python is detected automatically"
        )
        self.toolkit_python_edit.setAccessibleName("ai-toolkit Python executable")
        python_row = QHBoxLayout()
        python_row.addWidget(self.toolkit_python_edit, 1)
        self.toolkit_python_browse_button = _button("Browse")
        self.toolkit_python_browse_button.setObjectName("browseToolkitPythonButton")
        self.toolkit_python_browse_button.setAccessibleName(
            "Browse for ai-toolkit Python executable"
        )
        self.toolkit_python_browse_button.clicked.connect(self.choose_toolkit_python)
        python_row.addWidget(self.toolkit_python_browse_button)
        trainer_layout.addLayout(python_row)
        self.trainer_help_button = _button("Set up or repair trainer")
        self.trainer_help_button.setObjectName("trainerHelpButton")
        self.trainer_help_button.setAccessibleName(
            "Open guided LoRA trainer setup and repair"
        )
        self.trainer_help_button.clicked.connect(self.show_trainer_help)
        trainer_layout.addWidget(
            self.trainer_help_button,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        form_grid.addWidget(trainer_card, 0, 1)

        self.storage_card = _card()
        storage_layout = QVBoxLayout(self.storage_card)
        storage_layout.setContentsMargins(18, 17, 18, 18)
        storage_layout.setSpacing(9)
        storage_layout.addWidget(_title("ComfyUI LoRA library"))
        storage_layout.addWidget(
            _body(
                "Finished models are installed here so they immediately become "
                "available to ComfyUI."
            )
        )
        lora_row = QHBoxLayout()
        self.lora_dir_edit = QLineEdit()
        self.lora_dir_edit.setObjectName("loraDirEdit")
        self.lora_dir_edit.setPlaceholderText("Waiting for Comfy Desktop discovery")
        self.lora_dir_edit.setAccessibleName("ComfyUI LoRA folder")
        lora_row.addWidget(self.lora_dir_edit, 1)
        lora_browse = _button("Browse")
        lora_browse.setObjectName("browseLoraButton")
        lora_browse.setAccessibleName("Browse for ComfyUI LoRA folder")
        lora_browse.clicked.connect(self.choose_lora_dir)
        lora_row.addWidget(lora_browse)
        storage_layout.addLayout(lora_row)
        storage_layout.addWidget(_hint("Project workspace"))
        workspace_row = QHBoxLayout()
        self.workspace_dir_edit = QLineEdit()
        self.workspace_dir_edit.setObjectName("workspaceDirEdit")
        self.workspace_dir_edit.setAccessibleName("MIVIM project workspace folder")
        workspace_row.addWidget(self.workspace_dir_edit, 1)
        workspace_browse = _button("Browse")
        workspace_browse.setObjectName("browseWorkspaceButton")
        workspace_browse.setAccessibleName("Browse for MIVIM project workspace")
        workspace_browse.clicked.connect(self.choose_workspace_dir)
        workspace_row.addWidget(workspace_browse)
        storage_layout.addLayout(workspace_row)
        form_grid.addWidget(self.storage_card, 1, 0)

        self.connection_details_card = _card()
        advanced_layout = QVBoxLayout(self.connection_details_card)
        advanced_layout.setContentsMargins(18, 17, 18, 18)
        advanced_layout.setSpacing(9)
        advanced_layout.addWidget(_title("Connection details"))
        advanced_layout.addWidget(
            _body(
                "Normally detected automatically. Change this only for a custom "
                "local ComfyUI port."
            )
        )
        self.host_edit = QLineEdit()
        self.host_edit.setObjectName("hostEdit")
        self.host_edit.setPlaceholderText("127.0.0.1:8188")
        self.host_edit.setAccessibleName("ComfyUI local host and port")
        advanced_layout.addWidget(self.host_edit)
        self.open_comfy_button = _button("Open running ComfyUI")
        self.open_comfy_button.setObjectName("openComfyButton")
        self.open_comfy_button.clicked.connect(self.open_comfy_ui)
        self.open_comfy_button.setEnabled(False)
        advanced_layout.addWidget(self.open_comfy_button)
        form_grid.addWidget(self.connection_details_card, 1, 1)

        form_grid.setColumnStretch(0, 1)
        form_grid.setColumnStretch(1, 1)
        page.addLayout(form_grid)
        self._advanced_setup_widgets = (
            self.base_model_path_label,
            self.base_model_edit,
            self.base_model_browse_button,
            self.toolkit_dir_edit,
            self.toolkit_browse_button,
            self.toolkit_python_edit,
            self.toolkit_python_browse_button,
            self.storage_card,
            self.connection_details_card,
        )
        self.advanced_setup_toggle.toggled.connect(self._set_advanced_setup_visible)
        self._set_advanced_setup_visible(False)

        footer = QHBoxLayout()
        self.setup_save_note = _hint(
            "Settings are stored for this Windows user, not inside Comfy Desktop."
        )
        footer.addWidget(self.setup_save_note, 1)
        save_button = _button("Save settings")
        save_button.setObjectName("saveSettingsButton")
        save_button.clicked.connect(self.save_current_settings)
        footer.addWidget(save_button)
        continue_button = _button("Continue to photos  →", "primaryButton")
        continue_button.setObjectName("continueToProjectButton")
        continue_button.clicked.connect(self.continue_to_project)
        footer.addWidget(continue_button)
        page.addLayout(footer)
        page.addStretch(1)
        return self._scroll_page(content)

    def _set_advanced_setup_visible(self, visible: bool) -> None:
        for widget in self._advanced_setup_widgets:
            widget.setVisible(visible)
        self.advanced_setup_toggle.setText(
            "Hide advanced paths and ports"
            if visible
            else "Show advanced paths and ports"
        )

    # ------------------------------------------------------------- project UI
    def _build_project_page(self) -> QWidget:
        content = QWidget()
        page = QVBoxLayout(content)
        page.setContentsMargins(34, 30, 34, 34)
        page.setSpacing(18)
        page.addLayout(
            _page_header(
                "New project",
                "Who are you teaching the model?",
                "Choose varied source photos. Recommended settings are selected "
                "for you and the technical controls stay optional.",
            )
        )
        self.photo_drop = PhotoDropZone()
        self.photo_drop.folder_selected.connect(self._photo_folder_changed)
        page.addWidget(self.photo_drop)
        self.photo_summary = _hint("No folder selected.")
        photo_summary_row = QHBoxLayout()
        photo_summary_row.addWidget(self.photo_summary, 1)
        self.review_photos_button = _button("Review photo checks")
        self.review_photos_button.setObjectName("reviewPhotoChecksButton")
        self.review_photos_button.setEnabled(False)
        self.review_photos_button.clicked.connect(self.review_photo_checks)
        photo_summary_row.addWidget(self.review_photos_button)
        page.addLayout(photo_summary_row)

        project_card = _card()
        grid = QGridLayout(project_card)
        grid.setContentsMargins(18, 18, 18, 18)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(9)

        grid.addWidget(_title("Project name"), 0, 0)
        grid.addWidget(_title("Identity phrase"), 0, 1)
        self.project_name_edit = QLineEdit()
        self.project_name_edit.setObjectName("projectNameEdit")
        self.project_name_edit.setPlaceholderText("alex_v1")
        self.project_name_edit.setAccessibleName("Project name")
        self.trigger_edit = QLineEdit("ohwx person")
        self.trigger_edit.setObjectName("triggerEdit")
        self.trigger_edit.setAccessibleName("Unique identity phrase")
        grid.addWidget(self.project_name_edit, 1, 0)
        grid.addWidget(self.trigger_edit, 1, 1)
        grid.addWidget(
            _hint("Used only for folders and result history. Example: alex_v1"),
            2,
            0,
        )
        grid.addWidget(
            _hint(
                "A rare phrase that calls up this identity. The recommended value is safe."
            ),
            2,
            1,
        )
        grid.addWidget(_title("Training plan"), 3, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.setObjectName("presetCombo")
        self.preset_combo.addItem(
            "Balanced — recommended", TrainingPreset.BALANCED.value
        )
        self.preset_combo.addItem("Fast draft", TrainingPreset.FAST.value)
        self.preset_combo.addItem("High quality", TrainingPreset.QUALITY.value)
        self.preset_combo.setAccessibleName("Training quality plan")
        grid.addWidget(self.preset_combo, 4, 0)
        self.preset_hint = _hint(
            "Balanced uses 1024 px photos, a no-LoRA baseline, and three strengths."
        )
        grid.addWidget(self.preset_hint, 5, 0, 1, 2)
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)

        self.vision_checkbox = QCheckBox(
            "Describe each photo automatically (advanced; downloads a vision model)"
        )
        self.vision_checkbox.setObjectName("visionCaptionCheck")
        self.vision_checkbox.setToolTip(
            "Uses BLIP and needs a separate PyTorch/Transformers installation."
        )
        if os.environ.get("MIVIM_PACKAGED_APP") == "1":
            self.vision_checkbox.setChecked(False)
            self.vision_checkbox.setVisible(False)
        self.overwrite_checkbox = QCheckBox(
            "Rebuild this project and replace its existing outputs"
        )
        self.overwrite_checkbox.setObjectName("overwriteCheck")
        self.overwrite_checkbox.setToolTip(
            "Keeps source photos untouched, but replaces this project's prepared "
            "dataset and allows new checkpoints, grids, run records, and the "
            "installed ComfyUI LoRA to supersede older outputs."
        )
        grid.addWidget(self.vision_checkbox, 6, 0, 1, 2)
        grid.addWidget(self.overwrite_checkbox, 7, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        page.addWidget(project_card)

        actions = QHBoxLayout()
        back = _button("←  Connection")
        back.clicked.connect(lambda: self.show_page(NAV_SETUP))
        actions.addWidget(back)
        actions.addStretch(1)
        self.review_button = _button("Review setup  →", "primaryButton")
        self.review_button.setObjectName("reviewSetupButton")
        self.review_button.clicked.connect(self.review_project)
        actions.addWidget(self.review_button)
        page.addLayout(actions)
        page.addStretch(1)
        return self._scroll_page(content)

    # ---------------------------------------------------------------- run UI
    def _build_run_page(self) -> QWidget:
        content = QWidget()
        page = QVBoxLayout(content)
        page.setContentsMargins(34, 30, 34, 34)
        page.setSpacing(18)
        self.run_header = _page_header(
            "Review",
            "Check before the long run",
            "Every required part is tested through the same paths the application "
            "will actually use.",
        )
        page.addLayout(self.run_header)
        self.run_stack = QStackedWidget()

        readiness_widget = QWidget()
        readiness_layout = QVBoxLayout(readiness_widget)
        readiness_layout.setContentsMargins(0, 0, 0, 0)
        readiness_layout.setSpacing(12)
        self.readiness_summary = _body("Checking photos and tools…")
        readiness_layout.addWidget(self.readiness_summary)
        self.diagnostics_widget = QWidget()
        self.diagnostics_layout = QVBoxLayout(self.diagnostics_widget)
        self.diagnostics_layout.setContentsMargins(0, 0, 0, 0)
        self.diagnostics_layout.setSpacing(10)
        readiness_layout.addWidget(self.diagnostics_widget)
        readiness_actions = QHBoxLayout()
        back = _button("←  Edit project")
        back.clicked.connect(lambda: self.show_page(NAV_PROJECT))
        readiness_actions.addWidget(back)
        retry = _button("Check again")
        retry.setObjectName("retryReadinessButton")
        retry.clicked.connect(self.run_readiness_checks)
        readiness_actions.addWidget(retry)
        readiness_actions.addStretch(1)
        self.prepare_button = _button("Prepare photos only")
        self.prepare_button.setObjectName("prepareOnlyButton")
        self.prepare_button.clicked.connect(lambda: self.start_pipeline("prepare"))
        readiness_actions.addWidget(self.prepare_button)
        self.start_button = _button("Start full LoRA run", "successButton")
        self.start_button.setObjectName("startPipelineButton")
        self.start_button.clicked.connect(lambda: self.start_pipeline("full"))
        readiness_actions.addWidget(self.start_button)
        readiness_layout.addLayout(readiness_actions)

        monitor_widget = QWidget()
        monitor = QVBoxLayout(monitor_widget)
        monitor.setContentsMargins(0, 0, 0, 0)
        monitor.setSpacing(14)
        monitor_card = _card(hero=True)
        monitor_card_layout = QVBoxLayout(monitor_card)
        monitor_card_layout.setContentsMargins(20, 20, 20, 20)
        self.run_stage_label = QLabel("Starting…")
        self.run_stage_label.setObjectName("pageTitle")
        self.run_stage_label.setStyleSheet("font-size: 23px;")
        self.run_detail_label = _body(
            "Keep this window open. You can continue using other applications."
        )
        self.run_progress = QProgressBar()
        self.run_progress.setObjectName("pipelineProgress")
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(2)
        monitor_card_layout.addWidget(self.run_stage_label)
        monitor_card_layout.addWidget(self.run_detail_label)
        monitor_card_layout.addWidget(self.run_progress)
        monitor.addWidget(monitor_card)
        log_heading = QHBoxLayout()
        log_heading.addWidget(_title("Technical details"))
        log_heading.addStretch(1)
        self.open_logs_button = _button("Open local logs")
        self.open_logs_button.setObjectName("openLogsButton")
        self.open_logs_button.setAccessibleName(
            "Open the local MIVIM support log folder"
        )
        self.open_logs_button.setToolTip(
            "Logs stay on this computer and may include local file paths."
        )
        self.open_logs_button.clicked.connect(self.open_logs_folder)
        log_heading.addWidget(self.open_logs_button)
        self.cancel_button = _button("Stop safely", "dangerButton")
        self.cancel_button.setObjectName("cancelPipelineButton")
        self.cancel_button.clicked.connect(self.cancel_pipeline)
        log_heading.addWidget(self.cancel_button)
        monitor.addLayout(log_heading)
        self.run_log = QPlainTextEdit()
        self.run_log.setObjectName("pipelineLog")
        self.run_log.setReadOnly(True)
        self.run_log.setMaximumBlockCount(2000)
        self.run_log.setMinimumHeight(260)
        monitor.addWidget(self.run_log, 1)
        recovery_actions = QHBoxLayout()
        self.edit_after_run_button = _button("Edit project")
        self.edit_after_run_button.setObjectName("editAfterRunButton")
        self.edit_after_run_button.clicked.connect(lambda: self.show_page(NAV_PROJECT))
        self.edit_after_run_button.setVisible(False)
        recovery_actions.addWidget(self.edit_after_run_button)
        self.open_run_folder_button = _button("Open output folder")
        self.open_run_folder_button.setObjectName("openRunFolderButton")
        self.open_run_folder_button.clicked.connect(self.open_pipeline_output_folder)
        self.open_run_folder_button.setVisible(False)
        recovery_actions.addWidget(self.open_run_folder_button)
        self.use_fast_after_run_button = _button(
            "Use Fast & check again",
            "primaryButton",
        )
        self.use_fast_after_run_button.setObjectName("useFastAfterRunButton")
        self.use_fast_after_run_button.setAccessibleName(
            "Switch to the lower-memory Fast plan and check the project again"
        )
        self.use_fast_after_run_button.clicked.connect(self.use_fast_after_oom)
        self.use_fast_after_run_button.setVisible(False)
        recovery_actions.addWidget(self.use_fast_after_run_button)
        self.repair_trainer_after_run_button = _button(
            "Repair trainer",
            "primaryButton",
        )
        self.repair_trainer_after_run_button.setObjectName(
            "repairTrainerAfterRunButton"
        )
        self.repair_trainer_after_run_button.clicked.connect(
            self.repair_trainer_after_failure
        )
        self.repair_trainer_after_run_button.setVisible(False)
        recovery_actions.addWidget(self.repair_trainer_after_run_button)
        self.open_desktop_after_run_button = _button("Open Comfy Desktop")
        self.open_desktop_after_run_button.setObjectName("openDesktopAfterRunButton")
        self.open_desktop_after_run_button.clicked.connect(self.open_comfy_desktop)
        self.open_desktop_after_run_button.setVisible(False)
        recovery_actions.addWidget(self.open_desktop_after_run_button)
        self.retry_validation_button = _button("Retry validation only")
        self.retry_validation_button.setObjectName("retryValidationButton")
        self.retry_validation_button.clicked.connect(self.retry_validation)
        self.retry_validation_button.setVisible(False)
        recovery_actions.addWidget(self.retry_validation_button)
        recovery_actions.addStretch(1)
        self.retry_after_run_button = _button("Check and retry", "primaryButton")
        self.retry_after_run_button.setObjectName("retryAfterRunButton")
        self.retry_after_run_button.clicked.connect(self.retry_pipeline)
        self.retry_after_run_button.setVisible(False)
        recovery_actions.addWidget(self.retry_after_run_button)
        monitor.addLayout(recovery_actions)
        readiness_scroll = self._scroll_page(readiness_widget)
        readiness_scroll.setObjectName("readinessScroll")
        self.run_stack.addWidget(readiness_scroll)
        self.run_stack.addWidget(monitor_widget)
        page.addWidget(self.run_stack, 1)
        return content

    # ------------------------------------------------------------- results UI
    def _build_results_page(self) -> QWidget:
        content = QWidget()
        page = QVBoxLayout(content)
        page.setContentsMargins(34, 30, 34, 34)
        page.setSpacing(16)
        result_header = QVBoxLayout()
        result_header.setSpacing(5)
        result_eyebrow = QLabel("RESULT")
        result_eyebrow.setObjectName("eyebrow")
        self.result_title_label = QLabel("Your LoRA is ready to review")
        self.result_title_label.setObjectName("pageTitle")
        self.result_subtitle_label = QLabel(
            "Compare identity and flexibility across prompts and strengths, "
            "then continue in ComfyUI."
        )
        self.result_subtitle_label.setObjectName("pageSubtitle")
        self.result_subtitle_label.setWordWrap(True)
        result_header.addWidget(result_eyebrow)
        result_header.addWidget(self.result_title_label)
        result_header.addWidget(self.result_subtitle_label)
        page.addLayout(result_header)
        result_card = _card()
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(18, 18, 18, 18)
        self.result_image = QLabel("The comparison grid will appear here.")
        self.result_image.setObjectName("resultImage")
        self.result_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_image.setMinimumHeight(380)
        self.result_image.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.result_image.setWordWrap(True)
        self.result_image.setAccessibleName("LoRA validation comparison grid")
        result_layout.addWidget(self.result_image, 1)
        self.result_path_label = _hint()
        self.result_path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        result_layout.addWidget(self.result_path_label)
        page.addWidget(result_card, 1)

        self.comfy_handoff_card = _card()
        handoff_layout = QHBoxLayout(self.comfy_handoff_card)
        handoff_layout.setContentsMargins(18, 14, 18, 14)
        handoff_text = QVBoxLayout()
        handoff_text.addWidget(_title("Use this LoRA in ComfyUI"))
        self.result_details_label = _body()
        self.result_details_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        handoff_text.addWidget(self.result_details_label)
        handoff_layout.addLayout(handoff_text, 1)
        self.copy_result_setup_button = _button("Copy setup")
        self.copy_result_setup_button.setObjectName("copyResultSetupButton")
        self.copy_result_setup_button.clicked.connect(self.copy_result_setup)
        handoff_layout.addWidget(self.copy_result_setup_button)
        self.comfy_handoff_card.setVisible(False)
        page.addWidget(self.comfy_handoff_card)

        actions = QHBoxLayout()
        self.open_result_button = _button("Open result folder")
        self.open_result_button.setObjectName("openResultButton")
        self.open_result_button.clicked.connect(self.open_result_folder)
        actions.addWidget(self.open_result_button)
        self.open_full_grid_button = _button("View full-size grid")
        self.open_full_grid_button.setObjectName("openFullGridButton")
        self.open_full_grid_button.setAccessibleName(
            "Open the validation comparison grid at full size"
        )
        self.open_full_grid_button.clicked.connect(self.open_full_size_grid)
        self.open_full_grid_button.setVisible(False)
        actions.addWidget(self.open_full_grid_button)
        self.result_open_comfy_button = _button("Open ComfyUI", "primaryButton")
        self.result_open_comfy_button.setObjectName("resultOpenComfyButton")
        self.result_open_comfy_button.clicked.connect(self.open_comfy_ui)
        self.result_open_comfy_button.setEnabled(False)
        actions.addWidget(self.result_open_comfy_button)
        self.review_prepared_button = _button("Review photos & captions")
        self.review_prepared_button.setObjectName("reviewPreparedButton")
        self.review_prepared_button.clicked.connect(self.review_prepared_dataset)
        self.review_prepared_button.setVisible(False)
        actions.addWidget(self.review_prepared_button)
        self.continue_training_button = _button(
            "Check tools & continue training",
            "primaryButton",
        )
        self.continue_training_button.setObjectName("continuePreparedTrainingButton")
        self.continue_training_button.clicked.connect(self.continue_prepared_training)
        self.continue_training_button.setVisible(False)
        actions.addWidget(self.continue_training_button)
        actions.addStretch(1)
        new_project = _button("Train another version  →", "successButton")
        new_project.setObjectName("newProjectButton")
        new_project.clicked.connect(self.new_project)
        actions.addWidget(new_project)
        page.addLayout(actions)
        return content

    # ------------------------------------------------------------ async setup
    def _run_worker(
        self,
        function: Callable[[], object],
        success: Callable[[object], None],
        failure: Callable[[str], None] | None = None,
        *,
        operation: str = "background",
    ) -> None:
        worker = FunctionWorker(function)
        self._workers.append(worker)

        def done(result: object) -> None:
            if worker in self._workers:
                self._workers.remove(worker)
            if self._closing:
                return
            success(result)

        def failed(message: str, technical_details: str) -> None:
            if worker in self._workers:
                self._workers.remove(worker)
            if self._closing:
                return
            self.studio_log.exception_details(
                f"{operation}.failed",
                message,
                technical_details,
            )
            if failure is not None:
                failure(message)
            else:
                self._show_error("Could not complete the check", message)

        worker.signals.finished.connect(done)
        worker.signals.failed.connect(failed)
        self.thread_pool.start(worker)

    def refresh_setup(self) -> None:
        self._setup_request += 1
        request = self._setup_request
        self.retry_comfy_button.setEnabled(False)
        self.setup_logs_button.setVisible(False)
        self.comfy_badge.setText("CHECKING")
        self.comfy_badge.setObjectName("statusWarning")
        self.comfy_badge.style().unpolish(self.comfy_badge)
        self.comfy_badge.style().polish(self.comfy_badge)
        self.comfy_summary_label.setText(
            "Checking the installed Desktop instances and local ComfyUI API…"
        )
        self.studio_log.info(
            "setup.check.started",
            "Checking local Comfy Desktop installations and API availability.",
        )

        original = self._collect_settings()

        def work() -> tuple[AppSettings, ComfyDesktopDiscovery, ComfyApiStatus]:
            discovery = self.service.discover_comfy()
            return self.service.setup_snapshot(original, discovery)

        self._run_worker(
            work,
            lambda result: self._setup_refreshed(request, original, result),
            lambda message: self._setup_failed_for(request, message),
            operation="setup.check",
        )

    def _setup_refreshed(
        self,
        request: int,
        original: AppSettings,
        result: object,
    ) -> None:
        if request != self._setup_request:
            return
        if self._collect_settings() != original:
            self.retry_comfy_button.setEnabled(True)
            QTimer.singleShot(0, self.refresh_setup)
            return
        settings, discovery, status = cast(
            tuple[AppSettings, ComfyDesktopDiscovery, ComfyApiStatus],
            result,
        )
        self.settings = settings
        self.discovery = discovery
        self.comfy_status = status
        self._populate_settings()
        self.retry_comfy_button.setEnabled(True)
        self.open_comfy_button.setEnabled(status.reachable)
        self.result_open_comfy_button.setEnabled(status.reachable)
        local_installations = tuple(
            installation
            for installation in discovery.installations
            if installation.is_local
        )
        self.open_desktop_button.setText(
            "Open Comfy Desktop" if local_installations else "Install Comfy Desktop"
        )
        self.open_desktop_button.setEnabled(
            discovery.launcher.supported or not local_installations
        )
        if status.reachable:
            self.comfy_badge.setText("CONNECTED")
            state = "ready"
            self.comfy_summary_label.setText(
                f"Connected to a local ComfyUI instance at {status.host}. "
                f"{len(status.checkpoints)} checkpoint(s) and "
                f"{len(status.loras)} LoRA(s) visible."
            )
        elif local_installations:
            self.comfy_badge.setText("NOT RUNNING")
            state = "warning"
            self.comfy_summary_label.setText(
                f"Found {len(local_installations)} local installation(s). "
                "Open Comfy Desktop and launch the one you want to use."
            )
        else:
            self.comfy_badge.setText("NOT FOUND")
            state = "error"
            self.comfy_summary_label.setText(
                "Comfy Desktop was not found. Install it, or enter a local "
                "ComfyUI address under Connection details."
            )
        self.comfy_badge.setObjectName(
            {
                "ready": "statusReady",
                "warning": "statusWarning",
                "error": "statusError",
            }[state]
        )
        self.comfy_badge.style().unpolish(self.comfy_badge)
        self.comfy_badge.style().polish(self.comfy_badge)
        self.setup_logs_button.setVisible(False)
        self.studio_log.info(
            "setup.check.finished",
            (
                f"state={state}; local_installations={len(local_installations)}; "
                f"api_reachable={status.reachable}; checkpoints="
                f"{len(status.checkpoints)}"
            ),
        )
        self.save_current_settings(silent=True)
        if status.reachable:
            self._desktop_poll_remaining = 0
        elif self._desktop_poll_remaining > 0:
            self._desktop_poll_remaining -= 1
            QTimer.singleShot(5000, self.refresh_setup)

    def _setup_failed_for(self, request: int, message: str) -> None:
        if request == self._setup_request:
            self._setup_failed(message)

    def _setup_failed(self, message: str) -> None:
        self.studio_log.error_event("setup.check.failed", message)
        self.retry_comfy_button.setEnabled(True)
        self.setup_logs_button.setVisible(True)
        self.comfy_badge.setText("CHECK FAILED")
        self.comfy_badge.setObjectName("statusError")
        self.comfy_badge.style().unpolish(self.comfy_badge)
        self.comfy_badge.style().polish(self.comfy_badge)
        self.comfy_summary_label.setText(
            f"The automatic check could not finish: {message}"
        )

    def _populate_settings(self) -> None:
        local_installations = (
            tuple(
                installation
                for installation in self.discovery.installations
                if installation.is_local
            )
            if self.discovery is not None
            else ()
        )
        with QSignalBlocker(self.installation_combo):
            self.installation_combo.clear()
            for installation in local_installations:
                label = installation.name + (
                    f"  ·  {installation.version}" if installation.version else ""
                )
                self.installation_combo.addItem(label, installation.id)
            selected_index = self.installation_combo.findData(
                self.settings.comfy_installation_id
            )
            if selected_index >= 0:
                self.installation_combo.setCurrentIndex(selected_index)
        show_instance_picker = len(local_installations) > 1
        self.installation_label.setVisible(show_instance_picker)
        self.installation_combo.setVisible(show_instance_picker)

        self.host_edit.setText(self.settings.comfy_host)
        self.host_edit.setCursorPosition(0)
        self.lora_dir_edit.setText(
            str(self.settings.comfy_lora_dir)
            if self.settings.comfy_lora_dir is not None
            else ""
        )
        self.lora_dir_edit.setCursorPosition(0)
        self.lora_dir_edit.setToolTip(self.lora_dir_edit.text())
        self.toolkit_dir_edit.setText(
            str(self.settings.ai_toolkit_dir)
            if self.settings.ai_toolkit_dir is not None
            else ""
        )
        self.toolkit_dir_edit.setCursorPosition(0)
        self.toolkit_dir_edit.setToolTip(self.toolkit_dir_edit.text())
        self.toolkit_python_edit.setText(
            str(self.settings.ai_toolkit_python)
            if self.settings.ai_toolkit_python is not None
            else ""
        )
        self.toolkit_python_edit.setCursorPosition(0)
        self.toolkit_python_edit.setToolTip(self.toolkit_python_edit.text())
        self.base_model_edit.setText(
            str(self.settings.training_base_model)
            if self.settings.training_base_model is not None
            else ""
        )
        self.base_model_edit.setCursorPosition(0)
        self.base_model_edit.setToolTip(self.base_model_edit.text())
        workspace = self.settings.datasets_dir.parent
        self.workspace_dir_edit.setText(str(workspace))
        self.workspace_dir_edit.setCursorPosition(0)
        self.workspace_dir_edit.setToolTip(self.workspace_dir_edit.text())
        previous = self.settings.comfy_checkpoint
        names = list(self.comfy_status.checkpoints) if self.comfy_status else []
        self.checkpoint_combo.clear()
        self.checkpoint_combo.addItems(names)
        if previous and previous not in names:
            self.checkpoint_combo.addItem(previous)
        if previous:
            self.checkpoint_combo.setCurrentText(previous)
        with QSignalBlocker(self.sdxl_confirm_checkbox):
            self.sdxl_confirm_checkbox.setChecked(
                bool(previous)
                and self.settings.confirmed_sdxl_checkpoint.casefold()
                == previous.casefold()
            )
        if names:
            self.model_hint.setText(
                f"{len(names)} model(s) reported by the running ComfyUI instance."
            )
        else:
            self.model_hint.setText(
                "No checkpoint is currently visible. Add an SDXL model in ComfyUI."
            )
        if self.discovery is not None:
            matches = self.service.resolve_checkpoint_paths(
                self.discovery,
                previous,
            )
            if len(matches) > 1:
                self.model_hint.setText(
                    f"{len(matches)} local files share this checkpoint name. "
                    "Choose the exact file below before training."
                )

    def _populate_last_project(self) -> None:
        last = self.settings.last_project
        self.project_name_edit.setText(last.project_name)
        self.trigger_edit.setText(last.trigger)
        index = self.preset_combo.findData(last.preset.value)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
        self.vision_checkbox.setChecked(
            last.use_vision_captions and os.environ.get("MIVIM_PACKAGED_APP") != "1"
        )
        self.overwrite_checkbox.setChecked(last.overwrite_existing)
        if last.source_dir is not None and last.source_dir.is_dir():
            self.photo_drop.set_folder(last.source_dir)

    def _restore_last_result(self) -> None:
        record = self.settings.last_result
        if record is None:
            return
        try:
            config.validate_project_name(record.project_name)
        except ValueError:
            return
        result = self.service.project_path(record.result_path)
        expected_kind = (
            result.is_dir() if record.mode == "prepare" else result.is_file()
        )
        if result.exists() and expected_kind:
            self._show_result(result, record)

    def _draft_for_recovery(self, recovery: ValidationRecovery) -> ProjectDraft:
        last = self.settings.last_project
        source_dir = (
            last.source_dir if last.project_name == recovery.project_name else None
        )
        return ProjectDraft(
            project_name=recovery.project_name,
            source_dir=source_dir,
            trigger=recovery.trigger,
            preset=recovery.preset,
        )

    def _restore_validation_recovery(self) -> None:
        """Surface an installed LoRA whose validation did not finish before exit."""

        project_name = self.settings.last_project.project_name.strip()
        if not project_name:
            return
        try:
            recovery = self.service.read_validation_recovery(
                self.settings,
                project_name,
            )
        except ValueError:
            return
        if recovery is None or recovery.state != "validation_pending":
            return
        draft = self._draft_for_recovery(recovery)
        self._pending_validation_recovery = recovery
        self._installed_lora_name = recovery.lora_name
        self.draft = draft
        try:
            self.pipeline_command = self.service.build_validation_command(
                self.settings,
                draft,
                recovery.lora_name,
                checkpoint=recovery.checkpoint,
            )
        except (TypeError, ValueError):
            self.pipeline_command = None
        self.run_stack.setCurrentIndex(1)
        self.run_progress.setValue(78)
        self.run_stage_label.setText("LoRA installed; validation needs attention")
        self.run_detail_label.setText(
            "Training is already complete and the LoRA is safely installed. Open "
            "the selected ComfyUI instance, then retry validation without repeating "
            "photo preparation or training."
        )
        self.run_log.setPlainText(
            "[studio] Recovered an unfinished ComfyUI validation.\n"
            f"[studio] Installed LoRA: {recovery.lora_name}\n"
        )
        self.cancel_button.setEnabled(False)
        self.nav_buttons[NAV_RUN].setEnabled(True)
        self.pages.setCurrentIndex(NAV_RUN)
        for position, button in enumerate(self.nav_buttons):
            button.setChecked(position == NAV_RUN)
        self._show_recovery_actions(True, "validation")
        self.studio_log.warning(
            "validation.recovered",
            (
                f"project={recovery.project_name}; "
                f"lora={recovery.lora_name}; state={recovery.state}"
            ),
        )

    def _collect_settings(self) -> AppSettings:
        if not hasattr(self, "host_edit"):
            return self.settings

        def optional_path(text: str) -> Path | None:
            cleaned = text.strip()
            return Path(cleaned).expanduser() if cleaned else None

        checkpoint = self.checkpoint_combo.currentText().strip()
        workspace_text = self.workspace_dir_edit.text().strip()
        workspace = (
            Path(workspace_text).expanduser()
            if workspace_text
            else self.settings.datasets_dir.parent
        )
        return replace(
            self.settings,
            comfy_host=self.host_edit.text().strip() or "127.0.0.1:8188",
            comfy_installation_id=str(
                self.installation_combo.currentData()
                or self.settings.comfy_installation_id
                or ""
            ),
            comfy_lora_dir=optional_path(self.lora_dir_edit.text()),
            comfy_checkpoint=checkpoint,
            confirmed_sdxl_checkpoint=(
                checkpoint if self.sdxl_confirm_checkbox.isChecked() else ""
            ),
            training_base_model=optional_path(self.base_model_edit.text()),
            ai_toolkit_dir=optional_path(self.toolkit_dir_edit.text()),
            ai_toolkit_python=optional_path(self.toolkit_python_edit.text()),
            datasets_dir=workspace / "datasets",
            checkpoints_dir=workspace / "checkpoints",
            grids_dir=workspace / "grids",
            run_configs_dir=workspace / "configs",
        )

    def save_current_settings(self, *, silent: bool = False) -> None:
        self.settings = self._collect_settings()
        try:
            saved = save_settings(self.settings, self.settings_path)
        except (OSError, TypeError, ValueError) as exc:
            self._show_error("Settings were not saved", str(exc))
            return
        if not silent:
            self.setup_save_note.setText(f"Saved for this user: {saved}")

    def choose_lora_dir(self) -> None:
        start = self.lora_dir_edit.text().strip() or str(Path.home())
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose ComfyUI LoRA folder",
            start,
            QFileDialog.Option.ShowDirsOnly,
        )
        if selected:
            self.lora_dir_edit.setText(selected)
            self._invalidate_readiness()

    def create_verified_lora_folder(self) -> None:
        """Create only the missing LoRA directory proven by selected discovery."""

        settings = self._collect_settings()
        target = verified_missing_lora_directory(settings, self.discovery)
        if target is None:
            self._show_error(
                "LoRA folder was not created",
                "The selected Comfy Desktop model location could not be verified. "
                "Refresh the connection or choose its existing LoRA folder.",
            )
            return
        try:
            target.mkdir(parents=False, exist_ok=False)
        except OSError as exc:
            self._show_error(
                "LoRA folder was not created",
                f"Windows could not create the verified folder:\n{target}\n\n{exc}",
            )
            return
        self.lora_dir_edit.setText(str(target))
        self.save_current_settings(silent=True)
        self.studio_log.info(
            "setup.lora_folder.created",
            f"Created the selected Desktop instance's LoRA folder: {target}",
        )
        self._invalidate_readiness()
        self.refresh_setup()
        QTimer.singleShot(0, self.run_readiness_checks)

    def choose_base_model(self) -> None:
        start = self.base_model_edit.text().strip()
        if not start:
            start = str(Path.home())
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Choose local SDXL checkpoint",
            start,
            "Stable Diffusion checkpoints (*.safetensors *.ckpt);;All files (*)",
        )
        if selected:
            self.base_model_edit.setText(selected)
            self.base_model_edit.setCursorPosition(0)
            self._invalidate_readiness()

    def choose_toolkit_dir(self) -> None:
        start = self.toolkit_dir_edit.text().strip() or str(Path.home())
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose ai-toolkit folder",
            start,
            QFileDialog.Option.ShowDirsOnly,
        )
        if not selected:
            return
        folder = Path(selected)
        self.toolkit_dir_edit.setText(str(folder))
        self.toolkit_python_edit.clear()
        for python in (
            folder / "venv" / "Scripts" / "python.exe",
            folder / ".venv" / "Scripts" / "python.exe",
            folder / "venv" / "bin" / "python",
            folder / ".venv" / "bin" / "python",
        ):
            if python.is_file():
                self.toolkit_python_edit.setText(str(python))
                break
        self._invalidate_readiness()

    def choose_toolkit_python(self) -> None:
        start = self.toolkit_python_edit.text().strip()
        if not start:
            start = self.toolkit_dir_edit.text().strip() or str(Path.home())
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Choose ai-toolkit Python executable",
            start,
            "Python executable (python.exe python);;All files (*)",
        )
        if selected:
            self.toolkit_python_edit.setText(selected)
            self.toolkit_python_edit.setCursorPosition(0)
            self._invalidate_readiness()

    def show_trainer_help(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Set up or repair the LoRA trainer")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText("ai-toolkit needs its own working Python and NVIDIA PyTorch.")
        box.setInformativeText(
            "1. Choose the ai-toolkit folder.\n"
            "2. Choose that folder's venv Python if it was not detected.\n"
            "3. Install the current CUDA build from PyTorch's official selector, "
            "then use Check again.\n\n"
            "The studio tests the exact Python used by the real training run."
        )
        choose_folder = box.addButton(
            "Choose ai-toolkit folder",
            QMessageBox.ButtonRole.ActionRole,
        )
        choose_python = box.addButton(
            "Choose venv Python",
            QMessageBox.ButtonRole.ActionRole,
        )
        toolkit_guide = box.addButton(
            "Open ai-toolkit guide",
            QMessageBox.ButtonRole.ActionRole,
        )
        pytorch_guide = box.addButton(
            "Open PyTorch selector",
            QMessageBox.ButtonRole.ActionRole,
        )
        check_again = box.addButton(
            "Check on Review page",
            QMessageBox.ButtonRole.AcceptRole,
        )
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        clicked = box.clickedButton()
        if clicked is choose_folder:
            self.choose_toolkit_dir()
        elif clicked is choose_python:
            self.choose_toolkit_python()
        elif clicked is toolkit_guide:
            QDesktopServices.openUrl(QUrl("https://github.com/ostris/ai-toolkit"))
        elif clicked is pytorch_guide:
            QDesktopServices.openUrl(QUrl("https://pytorch.org/get-started/locally/"))
        elif clicked is check_again:
            self.review_project()

    def choose_workspace_dir(self) -> None:
        start = self.workspace_dir_edit.text().strip() or str(Path.home())
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose MIVIM project workspace",
            start,
            QFileDialog.Option.ShowDirsOnly,
        )
        if selected:
            self.workspace_dir_edit.setText(selected)
            self.workspace_dir_edit.setCursorPosition(0)
            self._invalidate_readiness()

    def open_comfy_desktop(self) -> None:
        local_installations = (
            tuple(
                installation
                for installation in self.discovery.installations
                if installation.is_local
            )
            if self.discovery is not None
            else ()
        )
        if not local_installations:
            QDesktopServices.openUrl(
                QUrl("https://docs.comfy.org/installation/desktop/windows")
            )
            return
        if self.discovery is None or not self.discovery.launcher.supported:
            self._show_error(
                "Comfy Desktop is not available",
                "Windows could not resolve the Comfy Desktop Start Menu application.",
            )
            return
        launcher = self.discovery.launcher
        try:
            subprocess.Popen(
                list(launcher.command),
                close_fds=True,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                ),
            )
        except OSError as exc:
            self._show_error("Could not open Comfy Desktop", str(exc))
            return
        self.comfy_summary_label.setText(
            "Comfy Desktop is opening. The studio will check the local instance again."
        )
        self._desktop_poll_remaining = 18
        QTimer.singleShot(3000, self.refresh_setup)

    def open_comfy_ui(self) -> None:
        host = (
            self.comfy_status.host
            if self.comfy_status is not None and self.comfy_status.reachable
            else None
        )
        if host is None:
            self._show_error(
                "ComfyUI is not connected",
                "Check the local connection before opening ComfyUI.",
            )
            return
        if not QDesktopServices.openUrl(QUrl(f"http://{host}")):
            self._show_error(
                "Could not open ComfyUI",
                "Windows could not open the local ComfyUI address in a browser.",
            )

    # ----------------------------------------------------------- project flow
    def continue_to_project(self) -> None:
        self.save_current_settings(silent=True)
        self.nav_buttons[NAV_PROJECT].setEnabled(True)
        self.show_page(NAV_PROJECT)

    def _connect_form_invalidation(self) -> None:
        self.host_edit.textEdited.connect(self._connection_changed)
        for editor in (
            self.lora_dir_edit,
            self.toolkit_dir_edit,
            self.toolkit_python_edit,
            self.base_model_edit,
            self.workspace_dir_edit,
            self.project_name_edit,
            self.trigger_edit,
        ):
            editor.textEdited.connect(self._invalidate_readiness)
        self.installation_combo.currentIndexChanged.connect(self._installation_changed)
        self.checkpoint_combo.currentTextChanged.connect(self._checkpoint_changed)
        self.sdxl_confirm_checkbox.toggled.connect(self._invalidate_readiness)
        self.vision_checkbox.toggled.connect(self._invalidate_readiness)
        self.overwrite_checkbox.toggled.connect(self._invalidate_readiness)

    def _connection_changed(self, *_args: object) -> None:
        self.comfy_status = None
        self.open_comfy_button.setEnabled(False)
        self.result_open_comfy_button.setEnabled(False)
        self._invalidate_readiness()

    def _installation_changed(self, *_args: object) -> None:
        self._connection_changed()
        QTimer.singleShot(0, self.refresh_setup)

    def _checkpoint_changed(self, checkpoint: str) -> None:
        confirmed = self.settings.confirmed_sdxl_checkpoint
        with QSignalBlocker(self.sdxl_confirm_checkbox):
            self.sdxl_confirm_checkbox.setChecked(
                bool(checkpoint)
                and bool(confirmed)
                and checkpoint.casefold() == confirmed.casefold()
            )
        resolved = self.service.resolve_checkpoint_path(self.discovery, checkpoint)
        current = self.base_model_edit.text().strip()
        selected_name = Path(checkpoint.replace("\\", "/")).name.casefold()
        if resolved is not None:
            self.base_model_edit.setText(str(resolved))
        elif current and Path(current).name.casefold() != selected_name:
            self.base_model_edit.clear()
        self._invalidate_readiness()

    def _invalidate_readiness(self, *_args: object) -> None:
        self._readiness_request += 1
        self.readiness_report = None
        self._validated_settings = None
        self._validated_draft = None
        if hasattr(self, "start_button"):
            self.start_button.setEnabled(False)
            self.prepare_button.setEnabled(False)
        if (
            hasattr(self, "run_stack")
            and self.pages.currentIndex() == NAV_RUN
            and self.run_stack.currentIndex() == 0
        ):
            self.readiness_summary.setText(
                "The project changed. Check again before starting."
            )

    def _photo_folder_changed(self, folder: Path) -> None:
        self.source_folder = folder
        self.source_summary = None
        self.review_photos_button.setEnabled(False)
        self._invalidate_readiness()
        if not self.project_name_edit.text().strip():
            safe = "".join(
                char if char.isalnum() or char in "._-" else "_"
                for char in folder.name.strip()
            ).strip("._-")
            if safe:
                self.project_name_edit.setText(f"{safe}_v1")
        self.photo_summary.setText("Inspecting the selected photos…")
        self.studio_log.info(
            "photos.check.started",
            f"Inspecting the selected folder: {folder}",
        )

        preset = TrainingPreset(self.preset_combo.currentData())
        self._source_request += 1
        request = self._source_request
        self._run_worker(
            lambda: inspect_source(folder, target_size=preset.image_size),
            lambda result: self._source_inspected_for(request, folder, preset, result),
            lambda message: self._source_inspection_failed(
                request, folder, preset, message
            ),
            operation="photos.check",
        )

    def _source_inspected_for(
        self,
        request: int,
        folder: Path,
        preset: TrainingPreset,
        result: object,
    ) -> None:
        if (
            request != self._source_request
            or folder != self.source_folder
            or preset.value != self.preset_combo.currentData()
        ):
            return
        self._source_inspected(result)

    def _source_inspection_failed(
        self,
        request: int,
        folder: Path,
        preset: TrainingPreset,
        message: str,
    ) -> None:
        if (
            request == self._source_request
            and folder == self.source_folder
            and preset.value == self.preset_combo.currentData()
        ):
            self.studio_log.error_event("photos.check.failed", message)
            self.source_summary = None
            self.review_photos_button.setEnabled(False)
            self.photo_summary.setText(f"The photos could not be inspected: {message}")

    def _source_inspected(self, result: object) -> None:
        summary = cast(SourceSummary, result)
        self.studio_log.info(
            "photos.check.finished",
            (
                f"usable={summary.usable_count}; unreadable="
                f"{summary.unreadable_count}; too_small={summary.small_count}; "
                f"issues={len(summary.issues)}"
            ),
        )
        self.source_summary = summary
        self.review_photos_button.setEnabled(True)
        detail = f"{summary.usable_count} usable photo(s)"
        if summary.unreadable_count:
            detail += f" • {summary.unreadable_count} unreadable"
        if summary.small_count:
            detail += f" • {summary.small_count} will be enlarged"
        if summary.duplicate_count:
            detail += f" • {summary.duplicate_count} exact duplicate(s)"
        if summary.blurry_count:
            detail += f" • {summary.blurry_count} may be blurry"
        if summary.near_duplicate_count:
            detail += f" • {summary.near_duplicate_count} very similar"
        self.photo_summary.setText(detail)

    def review_photo_checks(self) -> None:
        summary = self.source_summary
        if summary is None:
            return
        dialog = PhotoIssueDialog(summary, self)
        if dialog.exec():
            self._photo_folder_changed(summary.folder)

    def _preset_changed(self) -> None:
        preset = TrainingPreset(self.preset_combo.currentData())
        descriptions = {
            TrainingPreset.FAST: (
                "Fast draft uses 900 training steps, rank 8, 768 px preparation, "
                "a no-LoRA baseline, and one validation strength."
            ),
            TrainingPreset.BALANCED: (
                "Balanced uses 2,000 steps, rank 16, 1024 px preparation, and "
                "a no-LoRA baseline plus three comparison strengths."
            ),
            TrainingPreset.QUALITY: (
                "High quality uses 3,000 steps, rank 32, and a wider comparison "
                "sweep. Expect a longer run and larger model."
            ),
        }
        self.preset_hint.setText(descriptions[preset])
        self._invalidate_readiness()
        if self.source_folder is not None:
            self._photo_folder_changed(self.source_folder)

    def _build_draft(self) -> ProjectDraft:
        missing = self.service.root / ".no-photo-folder-selected"
        source = self.source_folder or missing
        preset = TrainingPreset(self.preset_combo.currentData())
        return ProjectDraft(
            project_name=self.project_name_edit.text().strip(),
            source_dir=source,
            trigger=self.trigger_edit.text().strip(),
            preset=preset,
            use_vision_captions=self.vision_checkbox.isChecked(),
            overwrite_existing=self.overwrite_checkbox.isChecked(),
        )

    def review_project(self) -> None:
        self.settings = self._collect_settings()
        self.draft = self._build_draft()
        self._invalidate_readiness()
        try:
            self.settings = replace(
                self.settings,
                last_project=self.draft,
            )
            save_settings(self.settings, self.settings_path)
        except (OSError, TypeError, ValueError) as exc:
            self._show_error(
                "Project settings were not saved",
                f"The project can still be checked in this session.\n\n{exc}",
            )
        self.nav_buttons[NAV_RUN].setEnabled(True)
        self.pages.setCurrentIndex(NAV_RUN)
        for position, button in enumerate(self.nav_buttons):
            button.setChecked(position == NAV_RUN)
        self.run_stack.setCurrentIndex(0)
        self.run_readiness_checks()

    # --------------------------------------------------------------- run flow
    def run_readiness_checks(self) -> None:
        draft = self._build_draft()
        settings = self._collect_settings()
        self.draft = draft
        self._readiness_request += 1
        request = self._readiness_request
        self.readiness_report = None
        self._validated_settings = None
        self._validated_draft = None
        self.readiness_summary.setText(
            "Checking the real photo, trainer, and ComfyUI paths…"
        )
        _layout_clear(self.diagnostics_layout)
        self.diagnostics_layout.addWidget(_body("Working…"))
        self.start_button.setEnabled(False)
        self.prepare_button.setEnabled(False)
        self.studio_log.info(
            "readiness.check.started",
            f"Checking project: {draft.project_name or '(unnamed)'}",
        )
        self._run_worker(
            lambda: self.service.readiness(settings, draft, self.discovery),
            lambda result: self._readiness_finished_for(
                request, settings, draft, result
            ),
            lambda message: self._readiness_failed_for(request, message),
            operation="readiness.check",
        )

    def _readiness_finished_for(
        self,
        request: int,
        settings: AppSettings,
        draft: ProjectDraft,
        result: object,
    ) -> None:
        if request != self._readiness_request:
            return
        if self._collect_settings() != settings or self._build_draft() != draft:
            self._invalidate_readiness()
            return
        self.settings = settings
        self.draft = draft
        self._validated_settings = settings
        self._validated_draft = draft
        self._readiness_finished(result)

    def _readiness_finished(self, result: object) -> None:
        report = cast(ReadinessReport, result)
        self.readiness_report = report
        _layout_clear(self.diagnostics_layout)
        for item in report.items:
            self.diagnostics_layout.addWidget(
                DiagnosticCard(item, self._handle_diagnostic_action)
            )
        report_items = report.items
        errors = sum(item.state == "error" for item in report_items)
        warnings = sum(item.state == "warning" for item in report_items)
        self.studio_log.info(
            "readiness.check.finished",
            (
                f"errors={errors}; warnings={warnings}; "
                f"can_prepare={report.can_prepare}; "
                f"can_run_full={report.can_run_full}"
            ),
        )
        for item in report_items:
            if item.state == "error":
                self.studio_log.error_event(
                    f"readiness.item.{item.code}",
                    f"{item.title}: {item.detail}",
                )
            elif item.state == "warning":
                self.studio_log.warning(
                    f"readiness.item.{item.code}",
                    f"{item.title}: {item.detail}",
                )
        if not errors:
            self.readiness_summary.setText(
                "Everything required for a full run is ready."
                + (
                    f" {warnings} recommendation(s) are worth reviewing."
                    if warnings
                    else ""
                )
            )
        else:
            self.readiness_summary.setText(
                f"{errors} item(s) still need attention. You can prepare photos "
                "independently when that option is enabled."
            )
        self.start_button.setEnabled(report.can_run_full)
        self.prepare_button.setEnabled(report.can_prepare)
        if report.prepared is not None and report.prepared.ready:
            self.start_button.setText("Continue with training")
            self.prepare_button.setText("Photos already prepared")
        else:
            self.start_button.setText("Start full LoRA run")
            self.prepare_button.setText("Prepare photos only")

    def _readiness_failed_for(self, request: int, message: str) -> None:
        if request == self._readiness_request:
            self._readiness_failed(message)

    def _readiness_failed(self, message: str) -> None:
        self.studio_log.error_event("readiness.check.failed", message)
        _layout_clear(self.diagnostics_layout)
        self.diagnostics_layout.addWidget(
            DiagnosticCard(
                ReadinessItem(
                    "internal",
                    "Readiness check",
                    "error",
                    message,
                    "Retry the check",
                ),
                self._handle_diagnostic_action,
            )
        )
        self.readiness_summary.setText("The setup check could not finish.")

    def _handle_diagnostic_action(self, item: ReadinessItem) -> None:
        if item.action in {"Review prepared captions", "Review prepared files"}:
            self.review_prepared_dataset()
        elif item.code in {"project", "collision"}:
            self.show_page(NAV_PROJECT)
            self.project_name_edit.setFocus()
        elif item.code in {"trigger", "captions"}:
            self.show_page(NAV_PROJECT)
            if item.code == "captions":
                self.vision_checkbox.setChecked(False)
            else:
                self.trigger_edit.setFocus()
        elif item.code == "photos":
            self.show_page(NAV_PROJECT)
            if self.source_summary is not None:
                self.review_photo_checks()
            else:
                self.photo_drop._browse()
        elif item.code == "comfy":
            self.show_page(NAV_SETUP)
            self.open_comfy_desktop()
        elif item.code == "checkpoint":
            self.show_page(NAV_SETUP)
            if item.action == "Confirm this is an SDXL base checkpoint":
                self.confirm_sdxl_checkpoint()
            else:
                self.checkpoint_combo.setFocus()
                self.checkpoint_combo.showPopup()
        elif item.code == "connection":
            self.show_page(NAV_SETUP)
            self.installation_combo.setFocus()
            if self.installation_combo.isVisible():
                self.installation_combo.showPopup()
        elif item.code == "base_model":
            self.show_page(NAV_SETUP)
            self.choose_base_model()
        elif item.code == "lora_dir":
            self.show_page(NAV_SETUP)
            if item.action == "Create LoRA folder":
                self.create_verified_lora_folder()
            else:
                self.choose_lora_dir()
        elif item.code == "trainer":
            self.show_page(NAV_SETUP)
            self.show_trainer_help()
        elif item.code in {"storage", "training_storage"}:
            self.show_page(NAV_SETUP)
            self.choose_workspace_dir()
        elif item.code == "internal":
            self.run_readiness_checks()

    def confirm_sdxl_checkpoint(self) -> None:
        checkpoint = self.checkpoint_combo.currentText().strip()
        if not checkpoint:
            self.checkpoint_combo.setFocus()
            self.checkpoint_combo.showPopup()
            return
        box = QMessageBox(self)
        box.setWindowTitle("Confirm the SDXL base model")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(f"Use “{checkpoint}” for both training and validation?")
        box.setInformativeText(
            "Only confirm if the model's download page says it is an SDXL base "
            "checkpoint. Do not use an SD 1.5 model, an SDXL refiner, or a LoRA. "
            "Using the wrong architecture wastes the training run."
        )
        confirm_button = box.addButton(
            "Confirm SDXL base model",
            QMessageBox.ButtonRole.AcceptRole,
        )
        choose_button = box.addButton(
            "Choose another model",
            QMessageBox.ButtonRole.ActionRole,
        )
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is confirm_button:
            self.sdxl_confirm_checkbox.setChecked(True)
            self.save_current_settings(silent=True)
            self.run_readiness_checks()
        elif box.clickedButton() is choose_button:
            self.checkpoint_combo.setFocus()
            self.checkpoint_combo.showPopup()

    def start_pipeline(self, mode: str) -> None:
        if mode not in {"full", "prepare"}:
            return
        active_process = self.pipeline_process
        if (
            active_process is not None
            and active_process.state() != QProcess.ProcessState.NotRunning
        ):
            self._show_error(
                "A run is already active",
                "Wait for the current run to finish, or stop it safely first.",
            )
            return
        current_settings = self._collect_settings()
        current_draft = self._build_draft()
        validated_settings = self._validated_settings
        validated_draft = self._validated_draft
        if (
            validated_settings is None
            or validated_draft is None
            or validated_settings != current_settings
            or validated_draft != current_draft
        ):
            self.draft = current_draft
            self.run_stack.setCurrentIndex(0)
            self.readiness_summary.setText(
                "The project changed. Checking the updated choices before starting…"
            )
            self.run_readiness_checks()
            return
        report = self.readiness_report
        allowed = report is not None and (
            report.can_run_full if mode == "full" else report.can_prepare
        )
        if not allowed:
            self._show_error(
                "The run is not ready",
                "Review the highlighted setup items and check again.",
            )
            return
        try:
            command = self.service.build_command(
                validated_settings,
                validated_draft,
                mode=mode,  # type: ignore[arg-type]
            )
        except (FileNotFoundError, TypeError, ValueError) as exc:
            self._show_error("The run could not start", str(exc))
            return
        if mode == "full":
            self._installed_lora_name = None
        self._launch_pipeline_command(
            command,
            "Preparing photos" if mode == "prepare" else "Starting your LoRA",
        )

    def _launch_pipeline_command(
        self,
        command: PipelineCommand,
        stage_label: str,
    ) -> None:
        self.pipeline_command = command
        self.pipeline_pid = None
        active_draft = self._validated_draft or self.draft
        active_settings = self._validated_settings or self._collect_settings()
        self._recovery_before_run = (
            self.service.read_validation_recovery(
                active_settings,
                active_draft.project_name,
            )
            if active_draft is not None and active_draft.project_name
            else None
        )
        self._pipeline_line_buffer = ""
        self._pipeline_recent_output = ""
        self._stop_training_progress(3)
        self._trainer_stage_seen = False
        self._run_failure_kind = "generic"
        if command.mode != "validate":
            self._installed_lora_name = None
        self.run_cancelled = False
        self._run_terminal_handled = False
        self._result_before = _artifact_snapshot(command.expected_result)
        self.studio_log.info(
            "pipeline.started",
            (f"mode={command.mode}; expected_result={command.expected_result}"),
        )
        self.pages.setCurrentIndex(NAV_RUN)
        for position, button in enumerate(self.nav_buttons):
            button.setChecked(position == NAV_RUN)
        self.run_stack.setCurrentIndex(1)
        self.run_log.clear()
        self.run_stage_label.setText(stage_label)
        self.run_detail_label.setText(
            "This uses the real project pipeline. Technical output is shown below "
            "if you need support."
        )
        self.cancel_button.setEnabled(True)
        self._show_recovery_actions(False)
        for button in self.nav_buttons:
            button.setEnabled(False)

        process = QProcess(self)
        self.pipeline_process = process
        process.setWorkingDirectory(str(command.working_directory))
        environment = QProcessEnvironment.systemEnvironment()
        for key, value in command.environment.items():
            environment.insert(key, value)
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_pipeline_output)
        process.started.connect(self._pipeline_started)
        process.errorOccurred.connect(self._pipeline_error)
        process.finished.connect(self._pipeline_finished)
        process.start(command.program, list(command.arguments))

    def _pipeline_started(self) -> None:
        if self.pipeline_process is not None:
            self.pipeline_pid = int(self.pipeline_process.processId())

    def _read_pipeline_output(self) -> None:
        process = self.pipeline_process
        if process is None:
            return
        data = bytes(process.readAllStandardOutput().data()).decode(
            "utf-8", errors="replace"
        )
        if not data:
            return
        self._pipeline_recent_output = (self._pipeline_recent_output + data)[
            -128 * 1024 :
        ]
        self.studio_log.pipeline_output(data)
        self.run_log.moveCursor(QTextCursor.MoveOperation.End)
        self.run_log.insertPlainText(data)
        self.run_log.ensureCursorVisible()
        combined = f"{self._pipeline_line_buffer}{data}"
        self._pipeline_line_buffer = ""
        for part in combined.splitlines(keepends=True):
            if part.endswith(("\n", "\r")):
                self._process_pipeline_line(part.rstrip("\r\n"))
            else:
                self._pipeline_line_buffer = part

    def _process_pipeline_line(self, line: str) -> None:
        progress = stage_from_log(line)
        if progress is not None:
            value, label = progress
            if value == 42:
                self._trainer_stage_seen = True
                self._start_training_progress(label)
            else:
                self._stop_training_progress(
                    max(self._last_determinate_progress, value)
                )
                self.run_stage_label.setText(label)
                if value > 42:
                    self.run_detail_label.setText(
                        "Training is complete. ComfyUI is installing and checking "
                        "the LoRA now."
                    )

    def _start_training_progress(self, label: str) -> None:
        """Show honest activity and elapsed time for a stage with no percentage."""

        if not self._training_progress_active:
            self._training_progress_active = True
            self._training_elapsed_clock.start()
            self._training_elapsed_timer.start()
            self.run_progress.setRange(0, 0)
        self.run_stage_label.setText(label)
        self._update_training_elapsed()

    def _update_training_elapsed(self) -> None:
        if (
            not self._training_progress_active
            or not self._training_elapsed_clock.isValid()
        ):
            return
        total_seconds = max(0, self._training_elapsed_clock.elapsed() // 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed = (
            f"{hours}:{minutes:02d}:{seconds:02d}"
            if hours
            else f"{minutes:02d}:{seconds:02d}"
        )
        self.run_detail_label.setText(
            f"Training is running • Elapsed {elapsed}. The moving bar means the "
            "trainer is still working."
        )

    def _stop_training_progress(self, value: int | None = None) -> None:
        """Return the progress bar to a concrete percentage for bounded stages."""

        self._training_elapsed_timer.stop()
        self._training_progress_active = False
        if value is not None:
            self._last_determinate_progress = max(0, min(100, value))
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(self._last_determinate_progress)

    def _flush_pipeline_line_buffer(self) -> None:
        if self._pipeline_line_buffer:
            self._process_pipeline_line(self._pipeline_line_buffer)
            self._pipeline_line_buffer = ""

    def cancel_pipeline(self) -> None:
        process = self.pipeline_process
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Stop this run?")
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText(
            "The current step will stop. Prepared photos and completed checkpoints "
            "are kept, but the current checkpoint may be incomplete."
        )
        stop_button = dialog.addButton(
            "Stop run",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        dialog.addButton("Keep running", QMessageBox.ButtonRole.RejectRole)
        dialog.exec()
        if dialog.clickedButton() is not stop_button:
            return
        self.studio_log.warning(
            "pipeline.cancelled", "The user stopped the active run."
        )
        self.run_cancelled = True
        self.cancel_button.setEnabled(False)
        self.run_stage_label.setText("Stopping safely…")
        self._stop_process_tree(force=False)
        QTimer.singleShot(3500, self._force_stop_pipeline)

    def _stop_process_tree(self, *, force: bool) -> None:
        process = self.pipeline_process
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            return
        if sys.platform != "win32" or self.pipeline_pid is None:
            process.kill() if force else process.terminate()
            return
        command = ["taskkill", "/PID", str(self.pipeline_pid), "/T"]
        if force:
            command.append("/F")
        try:
            subprocess.run(
                command,
                check=False,
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill() if force else process.terminate()

    def _force_stop_pipeline(self) -> None:
        process = self.pipeline_process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            self._stop_process_tree(force=True)

    def _pipeline_error(self, error: QProcess.ProcessError) -> None:
        process = self.pipeline_process
        if process is None:
            return
        if error == QProcess.ProcessError.Crashed and self.run_cancelled:
            return
        self.studio_log.error_event(
            "pipeline.process.error",
            f"error={error.name}; detail={process.errorString()}",
        )
        self.run_log.appendPlainText(
            f"\n[studio] Could not run the pipeline: {process.errorString()}"
        )
        if error == QProcess.ProcessError.FailedToStart:
            self.pipeline_pid = None
            self._run_terminal_handled = True
            self._stop_training_progress(0)
            self.cancel_button.setEnabled(False)
            self.run_stage_label.setText("The pipeline could not start")
            self.run_detail_label.setText(
                "No work was started. Edit the setup or check again, then retry."
            )
            self._restore_navigation()
            self._show_recovery_actions(True, "generic")

    def _refresh_validation_recovery_after_run(self) -> ValidationRecovery | None:
        draft = self._validated_draft or self.draft
        if draft is None or not draft.project_name:
            return None
        settings = self._validated_settings or self._collect_settings()
        try:
            recovery = self.service.read_validation_recovery(
                settings,
                draft.project_name,
            )
        except ValueError:
            return None
        if (
            recovery is None
            or recovery.state != "validation_pending"
            or recovery == self._recovery_before_run
        ):
            return None
        self._pending_validation_recovery = recovery
        self._installed_lora_name = recovery.lora_name
        return recovery

    def _classify_pipeline_failure(
        self,
        recovery: ValidationRecovery | None,
    ) -> RunFailureKind:
        """Map common real worker failures to one safe, direct recovery path."""

        if recovery is not None:
            return "validation"
        output = self._pipeline_recent_output.casefold()
        oom_markers = (
            "cuda out of memory",
            "torch.outofmemoryerror",
            "cuda error: out of memory",
        )
        if self._trainer_stage_seen and any(marker in output for marker in oom_markers):
            return "oom"
        trainer_markers = (
            "modulenotfounderror",
            "no module named",
            "importerror:",
            "failed to import",
        )
        if self._trainer_stage_seen and any(
            marker in output for marker in trainer_markers
        ):
            return "trainer"
        return "generic"

    def _pipeline_finished(
        self,
        exit_code: int,
        _exit_status: QProcess.ExitStatus,
    ) -> None:
        if self._run_terminal_handled:
            return
        self._run_terminal_handled = True
        self._read_pipeline_output()
        self._flush_pipeline_line_buffer()
        self._stop_training_progress()
        self.pipeline_pid = None
        self.cancel_button.setEnabled(False)
        self._restore_navigation()
        command = self.pipeline_command
        if command is None:
            return
        recovery = self._refresh_validation_recovery_after_run()
        if self.run_cancelled:
            self.studio_log.warning(
                "pipeline.finished",
                f"mode={command.mode}; result=cancelled; exit_code={exit_code}",
            )
            self.run_stage_label.setText("Run stopped")
            self.run_detail_label.setText(
                "Completed files were kept. Return to the review when you are ready."
            )
            self._stop_training_progress(0)
            self._show_recovery_actions(True, "generic")
            return
        if exit_code != 0:
            self.studio_log.error_event(
                "pipeline.finished",
                f"mode={command.mode}; result=failed; exit_code={exit_code}",
            )
            failure_kind = self._classify_pipeline_failure(recovery)
            if failure_kind == "validation":
                self.run_stage_label.setText(
                    "LoRA installed; validation needs attention"
                )
                self.run_detail_label.setText(
                    "Training is complete and the LoRA is safely installed. Open "
                    "the selected ComfyUI instance, then use Retry validation only. "
                    "Photos and training will not repeat."
                )
                self._stop_training_progress(78)
            elif failure_kind == "oom":
                self.run_stage_label.setText("GPU memory filled up")
                self.run_detail_label.setText(
                    "This training plan needed more GPU memory than was available. "
                    "Use Fast & check again to lower memory use. Prepared photos and "
                    "completed files are kept."
                )
                self._stop_training_progress(0)
            elif failure_kind == "trainer":
                self.run_stage_label.setText("Trainer needs repair")
                self.run_detail_label.setText(
                    "The selected ai-toolkit Python is missing a package it needs. "
                    "Use Repair trainer for guided folder, Python, and PyTorch checks."
                )
                self._stop_training_progress(0)
            else:
                self.run_stage_label.setText("This run needs attention")
                self.run_detail_label.setText(
                    "Completed photos or checkpoints were kept. The technical details "
                    "below show the stage that failed; fix the setup item and retry."
                )
                self._stop_training_progress(0)
            self._show_recovery_actions(True, failure_kind)
            return
        result_after = _artifact_snapshot(command.expected_result)
        if result_after is None or result_after == self._result_before:
            self.studio_log.error_event(
                "pipeline.result.unverified",
                f"Expected a new or changed result at: {command.expected_result}",
            )
            self.run_stage_label.setText("Result could not be verified")
            self.run_detail_label.setText(
                "The pipeline exited successfully, but it did not create or update "
                f"the expected result at {command.expected_result}."
            )
            self._show_recovery_actions(True, "generic")
            return
        self._stop_training_progress(100)
        self._pending_validation_recovery = None
        self.studio_log.info(
            "pipeline.finished",
            f"mode={command.mode}; result=verified; path={command.expected_result}",
        )
        self.run_stage_label.setText("Complete")
        self.run_detail_label.setText(
            "The result was verified on disk and is ready to review."
        )
        record = self._record_run_result(command)
        self._show_result(command.expected_result, record)
        QTimer.singleShot(450, lambda: self.show_page(NAV_RESULTS))

    def _restore_navigation(self) -> None:
        self.nav_buttons[NAV_SETUP].setEnabled(True)
        self.nav_buttons[NAV_PROJECT].setEnabled(True)
        self.nav_buttons[NAV_RUN].setEnabled(True)
        self.nav_buttons[NAV_RESULTS].setEnabled(self._latest_result is not None)

    def _show_recovery_actions(
        self,
        visible: bool,
        failure_kind: RunFailureKind | None = None,
    ) -> None:
        if failure_kind is not None:
            self._run_failure_kind = failure_kind
        kind = self._run_failure_kind
        self.edit_after_run_button.setVisible(visible)
        self.open_run_folder_button.setVisible(visible)
        self.use_fast_after_run_button.setVisible(visible and kind == "oom")
        self.repair_trainer_after_run_button.setVisible(visible and kind == "trainer")
        self.open_desktop_after_run_button.setVisible(visible and kind == "validation")
        self.retry_after_run_button.setVisible(visible and kind != "validation")
        self.retry_validation_button.setVisible(
            visible and self._installed_lora_name is not None
        )

    def use_fast_after_oom(self) -> None:
        """Apply the lower-memory plan and deliberately approve a safe rebuild."""

        index = self.preset_combo.findData(TrainingPreset.FAST.value)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
        self.overwrite_checkbox.setChecked(True)
        self.studio_log.info(
            "pipeline.recovery.fast",
            "The user selected the lower-memory Fast plan after a CUDA OOM.",
        )
        self._show_recovery_actions(False)
        self.review_project()

    def repair_trainer_after_failure(self) -> None:
        """Open the same guided trainer repair used by setup diagnostics."""

        self.studio_log.info(
            "pipeline.recovery.trainer",
            "The user opened guided trainer repair after a missing dependency.",
        )
        self.show_trainer_help()

    def retry_pipeline(self) -> None:
        self._show_recovery_actions(False)
        self.run_stack.setCurrentIndex(0)
        self.run_readiness_checks()

    def retry_validation(self) -> None:
        recovery = self._pending_validation_recovery
        settings: AppSettings | None
        draft: ProjectDraft | None
        lora_name: str | None
        if recovery is not None:
            settings = self._collect_settings()
            current = self.service.read_validation_recovery(
                settings,
                recovery.project_name,
            )
            if current is None or current.state != "validation_pending":
                self._show_error(
                    "Installed LoRA could not be recovered",
                    "The saved LoRA is no longer in the selected ComfyUI model "
                    "library. Reconnect that library or start a new run.",
                )
                return
            recovery = current
            self._pending_validation_recovery = current
            draft = self._draft_for_recovery(current)
            lora_name = current.lora_name
        else:
            settings = self._validated_settings
            draft = self._validated_draft
            lora_name = self._installed_lora_name
        if settings is None or draft is None or lora_name is None:
            self._show_error(
                "Validation cannot be retried yet",
                "No installed LoRA from this run was found.",
            )
            return
        try:
            command = self.service.build_validation_command(
                settings,
                draft,
                lora_name,
                checkpoint=(recovery.checkpoint if recovery is not None else None),
            )
        except (TypeError, ValueError) as exc:
            self._show_error("Validation could not start", str(exc))
            return
        self._launch_pipeline_command(command, "Retrying ComfyUI validation")

    def open_pipeline_output_folder(self) -> None:
        command = self.pipeline_command
        if command is None:
            return
        folder = (
            command.expected_result
            if command.expected_result.is_dir()
            else command.expected_result.parent
        )
        while not folder.exists() and folder.parent != folder:
            folder = folder.parent
        self._open_local_folder(folder)

    # ------------------------------------------------------------- result flow
    def _record_run_result(self, command: PipelineCommand) -> RunResult:
        draft = self._validated_draft or self.draft or self._build_draft()
        mode: Literal["prepare", "full"] = (
            "prepare" if command.mode == "prepare" else "full"
        )
        current_settings = self._collect_settings()
        outcome = self.service.read_pipeline_outcome(
            current_settings,
            draft.project_name,
        )
        outcome_matches = (
            outcome is not None
            and self.service.project_path(outcome.result_path).resolve()
            == command.expected_result.resolve()
        )
        record = RunResult(
            project_name=draft.project_name,
            mode=mode,
            result_path=command.expected_result,
            completed_at=(
                outcome.completed_at
                if outcome_matches and outcome is not None
                else datetime.now(UTC).isoformat()
            ),
            trigger=(
                outcome.trigger
                if outcome_matches and outcome is not None
                else draft.trigger
            ),
            preset=draft.preset,
            checkpoint=(
                outcome.checkpoint
                if outcome_matches and outcome is not None
                else current_settings.comfy_checkpoint
            ),
            lora_name=(
                outcome.lora_name
                if outcome_matches and outcome is not None
                else self._installed_lora_name or ""
            ),
        )
        self._installed_lora_name = record.lora_name or self._installed_lora_name
        self.settings = replace(current_settings, last_result=record)
        try:
            save_settings(self.settings, self.settings_path)
        except (OSError, TypeError, ValueError) as exc:
            self.run_log.appendPlainText(
                f"\n[studio] The verified result could not be added to history: {exc}"
            )
        return record

    def _show_result(self, result: Path, record: RunResult | None = None) -> None:
        self._latest_result = result
        if record is None and self.settings.last_result is not None:
            stored_path = self.service.project_path(
                self.settings.last_result.result_path
            )
            if stored_path == result:
                record = self.settings.last_result
        self._displayed_result_record = record
        self.nav_buttons[NAV_RESULTS].setEnabled(True)
        self.result_path_label.setText(str(result))
        self.result_image.setStyleSheet("")
        self.comfy_handoff_card.setVisible(False)
        self.review_prepared_button.setVisible(False)
        self.continue_training_button.setVisible(False)
        self.open_full_grid_button.setVisible(False)
        if result.is_file() and result.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }:
            pixmap = QPixmap(str(result))
            if not pixmap.isNull():
                self.result_title_label.setText("Your LoRA is ready to review")
                self.result_subtitle_label.setText(
                    "Compare the generated validation grid, then continue in ComfyUI."
                )
                self.result_image.setAccessibleName("LoRA validation comparison grid")
                self.open_full_grid_button.setVisible(True)
                self.result_open_comfy_button.setVisible(True)
                self.result_open_comfy_button.setEnabled(
                    self.comfy_status is not None and self.comfy_status.reachable
                )
                if record is not None:
                    preferred_strength = (
                        0.8
                        if 0.8 in record.preset.validation_strengths
                        else next(
                            (
                                value
                                for value in record.preset.validation_strengths
                                if value > 0
                            ),
                            1.0,
                        )
                    )
                    self.result_details_label.setText(
                        f"LoRA: {record.lora_name or 'See the result folder'}\n"
                        f"Identity phrase: {record.trigger}\n"
                        f"Base checkpoint: {record.checkpoint or 'Not recorded'}\n"
                        f"Suggested starting strength: {preferred_strength:g}"
                    )
                    self.comfy_handoff_card.setVisible(True)
                self.result_image.setPixmap(
                    pixmap.scaled(
                        920,
                        520,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                return
        if result.is_dir():
            self.result_title_label.setText("Your photos are ready for training")
            self.result_subtitle_label.setText(
                "Review the prepared photos and captions here, then check the "
                "remaining tools and continue without preparing everything again."
            )
            self.result_image.setAccessibleName("Prepared photo dataset status")
            self.result_open_comfy_button.setVisible(False)
            self.review_prepared_button.setVisible(True)
            self.continue_training_button.setVisible(True)
            self.result_image.setPixmap(QPixmap())
            self.result_image.setText(
                "✓\n\nYour photos are prepared.\n\n"
                "Review captions in the studio, then continue to the full LoRA. "
                "The prepared work will be reused."
            )
            self.result_image.setStyleSheet(
                "font-size: 19px; color: #9ee7d7; padding: 30px;"
            )
        else:
            self.result_image.setText("The result is ready.")

    def review_prepared_dataset(self) -> None:
        draft = self.draft or self.settings.last_project
        record = (
            self._displayed_result_record
            if self.pages.currentIndex() == NAV_RESULTS
            else None
        )
        if record is not None and record.mode == "prepare":
            result = self.service.project_path(record.result_path)
            required_trigger = record.trigger
        else:
            if not draft.project_name:
                return
            result = (
                self.service.project_path(self._collect_settings().datasets_dir)
                / draft.project_name
            )
            required_trigger = draft.trigger
        if not result.is_dir():
            self._show_error(
                "Prepared photos were not found",
                f"The prepared project folder no longer exists:\n{result}",
            )
            return
        if not required_trigger.strip():
            self._show_error(
                "Identity phrase is missing",
                "Return to the project page and choose an identity phrase before "
                "editing prepared captions.",
            )
            return
        PreparedDatasetDialog(result, required_trigger, self).exec()
        self._invalidate_readiness()

    def continue_prepared_training(self) -> None:
        record = self._displayed_result_record
        if record is not None:
            self.project_name_edit.setText(record.project_name)
            self.trigger_edit.setText(record.trigger)
            preset_index = self.preset_combo.findData(record.preset.value)
            if preset_index >= 0:
                self.preset_combo.setCurrentIndex(preset_index)
        self.overwrite_checkbox.setChecked(False)
        self.review_project()

    def copy_result_setup(self) -> None:
        record = self._displayed_result_record
        if record is None:
            return
        preferred_strength = (
            0.8
            if 0.8 in record.preset.validation_strengths
            else next(
                (value for value in record.preset.validation_strengths if value > 0),
                1.0,
            )
        )
        QApplication.clipboard().setText(
            "\n".join(
                (
                    f"LoRA: {record.lora_name}",
                    f"Identity phrase: {record.trigger}",
                    f"Base checkpoint: {record.checkpoint}",
                    f"Starting strength: {preferred_strength:g}",
                )
            )
        )
        self.copy_result_setup_button.setText("Copied")
        QTimer.singleShot(
            1800,
            lambda: self.copy_result_setup_button.setText("Copy setup"),
        )

    def open_result_folder(self) -> None:
        result = self._latest_result
        if result is None:
            return
        folder = result if result.is_dir() else result.parent
        self._open_local_folder(folder)

    def open_full_size_grid(self) -> None:
        result = self._latest_result
        if result is None or not result.is_file():
            self._show_error(
                "Comparison grid not found",
                "The full-size comparison image is no longer available.",
            )
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(result.resolve()))):
            self._show_error(
                "Could not open the comparison",
                "Windows could not open the full-size image.",
            )

    def _open_local_folder(self, folder: Path) -> None:
        if not folder.is_dir():
            self._show_error(
                "Folder not found", f"The folder no longer exists:\n{folder}"
            )
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve()))):
            self._show_error(
                "Could not open the folder",
                "Windows could not open this location in File Explorer.",
            )

    def open_logs_folder(self) -> None:
        """Open the bounded local support trail without uploading any data."""

        self.studio_log.flush()
        if not self.studio_log.available:
            detail = self.studio_log.error or "The log writer is unavailable."
            self._show_error(
                "Local logs are unavailable",
                f"The studio could not create its local log folder.\n\n{detail}",
            )
            return
        self._open_local_folder(self.studio_log.directory)

    def new_project(self) -> None:
        previous_name = self.draft.project_name if self.draft is not None else ""
        self.pipeline_process = None
        self.pipeline_command = None
        self.pipeline_pid = None
        self.draft = None
        self.readiness_report = None
        self._validated_settings = None
        self._validated_draft = None
        self._installed_lora_name = None
        self._pending_validation_recovery = None
        self._recovery_before_run = None
        self._pipeline_line_buffer = ""
        self.run_stack.setCurrentIndex(0)
        self.run_log.clear()
        self._show_recovery_actions(False)
        if previous_name:
            self.project_name_edit.setText(self._increment_project_name(previous_name))
        self.nav_buttons[NAV_RUN].setEnabled(False)
        self.nav_buttons[NAV_RESULTS].setEnabled(False)
        self.show_page(NAV_PROJECT)

    @staticmethod
    def _increment_project_name(name: str) -> str:
        if "_v" in name:
            prefix, version = name.rsplit("_v", 1)
            if version.isdigit():
                return f"{prefix}_v{int(version) + 1}"
        return f"{name}_v2" if name else ""

    # ---------------------------------------------------------------- helpers
    def _show_error(self, title: str, message: str) -> None:
        self.studio_log.error_event("ui.error", f"{title}: {message}")
        QMessageBox.critical(self, title, message)

    def closeEvent(self, event: QCloseEvent) -> None:
        process = self.pipeline_process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            answer = QMessageBox.question(
                self,
                "A run is still active",
                "Closing the studio will stop the active run. Keep the window open?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.run_cancelled = True
            self._stop_process_tree(force=True)
        self._closing = True
        self._training_elapsed_timer.stop()
        self.studio_log.info("session.closed", "MIVIM LoRA Studio closed.")
        self.studio_log.close()
        event.accept()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch MIVIM LoRA Studio.")
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Save a screenshot after the real window initializes, then exit.",
    )
    parser.add_argument(
        "--screenshot-delay",
        type=int,
        default=1400,
        help="Milliseconds to wait before a requested screenshot (default: 1400).",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        help="Use an alternate settings file (primarily for testing).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    existing_app = QApplication.instance()
    app = (
        QApplication(sys.argv[:1])
        if existing_app is None
        else cast(QApplication, existing_app)
    )
    app.setApplicationName("MIVIM LoRA Studio")
    app.setOrganizationName("MIVIM")
    app.setWindowIcon(_app_icon())
    settings_path = args.settings or default_settings_path()
    window = MainWindow(
        settings=load_settings(settings_path),
        settings_path=settings_path,
    )
    window.show()

    if args.screenshot is not None:
        destination = args.screenshot.resolve()

        def capture() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(destination)):
                print(
                    f"Could not save desktop screenshot: {destination}",
                    file=sys.stderr,
                )
                app.exit(2)
                return
            print(f"Desktop screenshot: {destination}")
            app.quit()

        QTimer.singleShot(max(200, args.screenshot_delay), capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
