"""MIVIM LoRA Studio — a guided Windows companion for Comfy Desktop.

Run with:

    python -m src.desktop

The GUI launches the same supported pipeline entry point as the CLI, streams
its real output, and supplies settings through the child environment.  Users do
not need to edit ``.env`` or understand its implementation details.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import traceback
import uuid
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import replace
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
    QResizeEvent,
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
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import __version__, config
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
    ReadinessAction,
    ReadinessActionKind,
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
from .checkpoint_review import CheckpointReviewDialog
from .comfy_client import (
    ComfyClient,
    ComfyUserProfile,
    PromptCancellationResult,
    WorkflowPublication,
)
from .comfy_desktop import (
    ComfyApiStatus,
    ComfyDesktopDiscovery,
    comfy_start_instruction,
)
from .dataset_review import PreparedDatasetDialog
from .desktop_theme import APP_STYLE
from .io_utils import artifact_fingerprint
from .model_profiles import MODEL_PROFILES, model_profile
from .photo_review import PhotoIssueDialog
from .process_control import (
    DESKTOP_OWNER_MUTEX_ENV,
    DesktopOwnerLifetime,
)
from .run_guardian import RunLeaseBusyError, RunLeaseError, pipeline_run_guard
from .studio_logging import StudioLog, log_directory_for
from .training_checkpoints import TrainingCheckpoint, discover_training_checkpoints

NAV_SETUP, NAV_PROJECT, NAV_RUN, NAV_RESULTS = range(4)
RunFailureKind = Literal[
    "busy",
    "comfy_busy",
    "generic",
    "oom",
    "trainer",
    "validation",
]


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


def _file_stat_identity(path: Path) -> tuple[int, int]:
    """Return a cheap post-hash guard for a model file."""

    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _file_bytes_fingerprint(data: bytes) -> str:
    """Return the same immutable-file fingerprint used by result records."""

    digest = hashlib.sha256()
    digest.update(b"mivim-file-v1\0")
    digest.update(data)
    return f"sha256:{digest.hexdigest()}"


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
        self.title_label = QLabel("Choose a folder of training images")
        self.title_label.setObjectName("dropTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_label = _hint(
            "Drop a folder here, or browse. Use varied images of one person or character."
        )
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_label.setMaximumWidth(580)
        self.browse_button = _button("Browse images", "primaryButton")
        self.browse_button.setObjectName("choosePhotosButton")
        self.browse_button.clicked.connect(self._browse)
        layout.addWidget(self.icon_label)
        layout.addWidget(self.title_label)
        layout.addWidget(self.detail_label)
        layout.addSpacing(5)
        layout.addWidget(self.browse_button, alignment=Qt.AlignmentFlag.AlignHCenter)

    def _browse(self) -> bool:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose training image folder",
            str(Path.home() / "Pictures"),
            QFileDialog.Option.ShowDirsOnly,
        )
        if selected:
            self.set_folder(Path(selected))
            return True
        return False

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


class AspectRatioPixmapLabel(QLabel):
    """Fit a retained source image whenever the result viewport changes."""

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self._source_pixmap: QPixmap | None = None

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = QPixmap(pixmap)
        self._fit_source_pixmap()

    def clear_source_pixmap(self) -> None:
        self._source_pixmap = None
        self.clear()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._fit_source_pixmap()

    def _fit_source_pixmap(self) -> None:
        pixmap = self._source_pixmap
        size = self.contentsRect().size()
        if pixmap is None or pixmap.isNull() or size.isEmpty():
            return
        self.setPixmap(
            pixmap.scaled(
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


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
                action: QWidget = _hint(item.action.label)
            else:
                button = _button(item.action.label)
                button.setAccessibleName(f"{item.title}: {item.action.label}")
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
        self._desktop_owner_lifetime = DesktopOwnerLifetime()
        self.settings_path = settings_path or default_settings_path()
        loaded_settings = settings or load_settings(self.settings_path)
        self.settings = with_user_storage_defaults(
            loaded_settings,
            self.settings_path,
        )
        self.studio_log = StudioLog(log_directory_for(self.settings_path))
        self.studio_log.info(
            "session.started",
            f"MIVIM LoRA Studio {__version__} started. "
            "Support logs stay on this computer.",
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
        self._owned_comfy_prompt: tuple[str, str, str] | None = None
        self._comfy_cancel_detail = ""
        self._closing = False
        self._validated_settings: AppSettings | None = None
        self._validated_draft: ProjectDraft | None = None
        self._run_settings_snapshot: AppSettings | None = None
        self._run_draft_snapshot: ProjectDraft | None = None
        self._verified_result_identity: (
            tuple[RunResult, tuple[int, int], tuple[int, int]] | None
        ) = None
        self._result_identity_failure: tuple[RunResult, str] | None = None
        self._result_identity_request = 0
        self._recovery_check_request = 0
        self._setup_request = 0
        self._source_request = 0
        self._readiness_request = 0
        self._awaiting_comfy_launch = False
        self._setup_in_progress = False
        self._comfy_retry_timer = QTimer(self)
        self._comfy_retry_timer.setSingleShot(True)
        self._comfy_retry_timer.timeout.connect(self._poll_comfy_connection)
        self._return_to_readiness_when_connected = False
        self._return_to_readiness_after_installation = False
        self._return_to_readiness_after_setup = False
        self._return_to_readiness_after_checkpoint = False
        self._workflow_publish_request = 0
        self._pending_workflow_publish: RunResult | None = None
        self._published_workflow: tuple[RunResult, WorkflowPublication] | None = None
        self._workflow_publish_busy = False
        self._workflow_publish_retry_requires_refresh = False

        self.setWindowTitle(f"MIVIM LoRA Studio {__version__}")
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
        self._single_shot(80, self.refresh_setup)

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
        version = QLabel(f"Version {__version__} • SDXL")
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
        if not content.objectName():
            content.setObjectName("scrollPageContent")
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
        if (
            index == NAV_SETUP
            and self.discovery is not None
            and any(item.is_local for item in self.discovery.installations)
            and not (self.comfy_status is not None and self.comfy_status.reachable)
        ):
            self._comfy_retry_timer.start(0)

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
        profile_label = _hint("Model profile")
        model_layout.addWidget(profile_label)
        self.model_profile_combo = QComboBox()
        self.model_profile_combo.setObjectName("modelProfileCombo")
        self.model_profile_combo.setAccessibleName("Model profile")
        profile_label.setBuddy(self.model_profile_combo)
        for profile in MODEL_PROFILES:
            self.model_profile_combo.addItem(profile.label, profile.id)
        profile_index = self.model_profile_combo.findData(self.settings.model_profile)
        if profile_index < 0:
            self.model_profile_combo.addItem(
                "Unavailable profile · choose a replacement",
                self.settings.model_profile,
            )
            profile_index = self.model_profile_combo.count() - 1
        self.model_profile_combo.setCurrentIndex(profile_index)
        model_layout.addWidget(self.model_profile_combo)
        self.model_profile_hint = _body("")
        model_layout.addWidget(self.model_profile_hint)
        self._update_model_profile_hint()
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
        continue_button = _button("Continue to images  →", "primaryButton")
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
                "Choose varied source images. Recommended settings are selected "
                "for you and the technical controls stay optional.",
            )
        )
        self.photo_drop = PhotoDropZone()
        self.photo_drop.folder_selected.connect(self._photo_folder_changed)
        page.addWidget(self.photo_drop)
        self.photo_summary = _hint("No folder selected.")
        photo_summary_row = QHBoxLayout()
        photo_summary_row.addWidget(self.photo_summary, 1)
        self.review_photos_button = _button("Review images")
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
        self.preset_combo.addItem("Extended training", TrainingPreset.QUALITY.value)
        self.preset_combo.setAccessibleName("Training quality plan")
        grid.addWidget(self.preset_combo, 4, 0)
        steps_label = _title("Training steps")
        grid.addWidget(steps_label, 3, 1)
        self.training_steps_spin = QSpinBox()
        self.training_steps_spin.setObjectName("trainingStepsSpin")
        self.training_steps_spin.setAccessibleName(
            "Training steps; zero uses the plan default"
        )
        self.training_steps_spin.setRange(0, 100000)
        self.training_steps_spin.setSingleStep(100)
        self.training_steps_spin.setGroupSeparatorShown(True)
        self.training_steps_spin.setSpecialValueText("Plan default (2,000)")
        steps_label.setBuddy(self.training_steps_spin)
        grid.addWidget(self.training_steps_spin, 4, 1)
        self.preset_hint = _hint(
            "Balanced uses rank 16, 1024 px preparation, a no-LoRA baseline, and three strengths."
        )
        grid.addWidget(self.preset_hint, 5, 0, 1, 2)
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        self.training_steps_spin.valueChanged.connect(self._training_steps_changed)

        self.training_duration_hint = _hint("")
        grid.addWidget(self.training_duration_hint, 6, 0, 1, 2)
        self.caption_source_hint = _hint(
            "Existing .txt captions beside your images are imported automatically. "
            "Untagged images get the identity phrase; edit them after preparation."
        )
        self.caption_source_hint.setAccessibleName("Source caption summary")
        grid.addWidget(self.caption_source_hint, 7, 0, 1, 2)
        self.preview_suite_hint = _hint("")
        self.preview_suite_hint.setAccessibleName("Preview shots for this model")
        grid.addWidget(self.preview_suite_hint, 8, 0, 1, 2)

        self.vision_checkbox = QCheckBox(
            "Describe untagged images with BLIP (advanced; downloads a vision model)"
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
        grid.addWidget(self.vision_checkbox, 9, 0, 1, 2)
        grid.addWidget(self.overwrite_checkbox, 10, 0, 1, 2)
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
        content.setObjectName("resultsPage")
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
        self.compare_checkpoints_button = _button("Compare training checkpoints")
        self.compare_checkpoints_button.setObjectName("compareCheckpointsButton")
        self.compare_checkpoints_button.clicked.connect(
            self.compare_training_checkpoints
        )
        self.compare_checkpoints_button.setVisible(False)
        page.addWidget(self.compare_checkpoints_button, 0, Qt.AlignmentFlag.AlignLeft)
        result_card = _card()
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(18, 18, 18, 18)
        self.result_image = AspectRatioPixmapLabel(
            "The comparison grid will appear here."
        )
        self.result_image.setObjectName("resultImage")
        self.result_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_image.setMinimumHeight(300)
        self.result_image.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.result_image.setWordWrap(True)
        self.result_image.setAccessibleName("LoRA validation comparison grid")
        result_layout.addWidget(self.result_image, 1)
        result_path = QHBoxLayout()
        result_path.setSpacing(10)
        result_path.addWidget(_hint("Saved result"))
        self.result_path_label = QLineEdit()
        self.result_path_label.setObjectName("resultPath")
        self.result_path_label.setReadOnly(True)
        self.result_path_label.setAccessibleName("Saved result path")
        result_path.addWidget(self.result_path_label, 1)
        result_layout.addLayout(result_path)
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
        self.workflow_handoff_status_label = _hint("")
        self.workflow_handoff_status_label.setObjectName("workflowHandoffStatus")
        self.workflow_handoff_status_label.setWordWrap(True)
        self.workflow_handoff_status_label.setVisible(False)
        handoff_text.addWidget(self.workflow_handoff_status_label)
        handoff_layout.addLayout(handoff_text, 1)
        self.publish_workflow_button = _button(
            "Add to ComfyUI",
            "primaryButton",
        )
        self.publish_workflow_button.setObjectName("publishWorkflowButton")
        self.publish_workflow_button.clicked.connect(self.add_ready_workflow_to_comfyui)
        self.publish_workflow_button.setVisible(False)
        handoff_layout.addWidget(self.publish_workflow_button)
        self.open_ready_workflow_button = _button("Show JSON file")
        self.open_ready_workflow_button.setObjectName("openReadyWorkflowButton")
        self.open_ready_workflow_button.clicked.connect(self.open_ready_workflow)
        self.open_ready_workflow_button.setVisible(False)
        handoff_layout.addWidget(self.open_ready_workflow_button)
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
        self.result_open_comfy_button = _button(
            "Open ComfyUI in browser",
            "primaryButton",
        )
        self.result_open_comfy_button.setObjectName("resultOpenComfyButton")
        self.result_open_comfy_button.clicked.connect(self.open_comfy_ui)
        self.result_open_comfy_button.setEnabled(False)
        actions.addWidget(self.result_open_comfy_button)
        self.result_open_desktop_button = _button("Open Comfy Desktop")
        self.result_open_desktop_button.setObjectName("resultOpenDesktopButton")
        self.result_open_desktop_button.clicked.connect(self.open_result_comfy_desktop)
        self.result_open_desktop_button.setVisible(False)
        actions.addWidget(self.result_open_desktop_button)
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
        results_scroll = self._scroll_page(content)
        results_scroll.setObjectName("resultsScroll")
        return results_scroll

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

    def _single_shot(self, delay_ms: int, callback: Callable[[], None]) -> None:
        """Run a delayed UI callback only while this window still exists."""

        timer = QTimer(self)
        timer.setSingleShot(True)

        def invoke() -> None:
            timer.deleteLater()
            if not self._closing:
                callback()

        timer.timeout.connect(invoke)
        timer.start(max(0, delay_ms))

    def _poll_comfy_connection(self) -> None:
        if not self._closing and (
            self.pages.currentIndex() == NAV_SETUP or self._awaiting_comfy_launch
        ):
            self.refresh_setup(background=True)

    def refresh_setup(self, *, background: bool = False) -> None:
        if self._setup_in_progress or self._closing:
            return
        self._comfy_retry_timer.stop()
        self._setup_in_progress = True
        self._setup_request += 1
        request = self._setup_request
        if self._workflow_publish_busy:
            # A discovery refresh changes the runtime generation.  An in-flight
            # upload may still finish remotely, but its callback must not update
            # this UI as though it used the newly discovered instance/port.
            self._workflow_publish_request += 1
            self._workflow_publish_busy = False
            self._workflow_publish_retry_requires_refresh = False
            self.publish_workflow_button.setEnabled(False)
            self.publish_workflow_button.setText("Checking connection…")
            self.workflow_handoff_status_label.setText(
                "The ComfyUI connection changed while the workflow was being "
                "added. MIVIM is checking the current Desktop instance before "
                "you retry."
            )
        self.retry_comfy_button.setEnabled(False)
        self.setup_logs_button.setVisible(False)
        if not background:
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
        self._setup_in_progress = False
        if self._collect_settings() != original:
            self.retry_comfy_button.setEnabled(True)
            self._single_shot(0, self.refresh_setup)
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
        self.result_open_comfy_button.setEnabled(
            status.reachable
            and self._result_comfy_context_issue(
                self._displayed_result_record,
            )
            is None
        )
        displayed = self._displayed_result_record
        local_installations = tuple(
            installation
            for installation in discovery.installations
            if installation.is_local
        )
        desktop_detected = discovery.detected
        self.open_desktop_button.setText(
            "Open Comfy Desktop"
            if local_installations or desktop_detected
            else "Install Comfy Desktop"
        )
        self.open_desktop_button.setEnabled(
            discovery.launcher.supported
            or not (local_installations or desktop_detected)
        )
        local_ids = {installation.id for installation in local_installations}
        unavailable_selection = bool(settings.comfy_installation_id) and (
            settings.comfy_installation_id not in local_ids
        )
        selected_installation = discovery.selected_installation
        selected_name = (
            selected_installation.name if selected_installation is not None else ""
        )
        ambiguous_runtime_identity = (
            selected_installation is not None
            and sum(
                installation.name.casefold() == selected_name.casefold()
                for installation in local_installations
            )
            > 1
        )
        if unavailable_selection:
            self.comfy_badge.setText("CHOOSE INSTANCE")
            state = "warning"
            self.comfy_summary_label.setText(
                "The Comfy Desktop installation used last time is no longer "
                "registered. Choose another instance explicitly; MIVIM did not "
                "connect to a substitute."
            )
        elif ambiguous_runtime_identity:
            self.comfy_badge.setText("RENAME INSTANCES")
            state = "warning"
            self.comfy_summary_label.setText(
                f"More than one Desktop installation is named "
                f'"{selected_name}". Rename one in Comfy Desktop so '
                "MIVIM can identify the correct running instance safely."
            )
        elif status.reachable:
            self.comfy_badge.setText("CONNECTED")
            state = "ready"
            self.comfy_summary_label.setText(
                f"Connected to a local ComfyUI instance at {status.host}. "
                f"{len(status.checkpoints)} checkpoint(s) and "
                f"{len(status.loras)} LoRA(s) visible."
            )
        elif local_installations:
            state = "warning"
            starting_hosts = tuple(
                hint.host
                for hint in discovery.runtime_hints
                if hint.installation_name.casefold() == selected_name.casefold()
            )
            if starting_hosts:
                self.comfy_badge.setText("CONNECTING")
                self.comfy_summary_label.setText(
                    f'Desktop started "{selected_name}". Waiting for ComfyUI to '
                    f"answer at {starting_hosts[0]}. MIVIM checks automatically; "
                    "if startup fails, check that instance's terminal in Desktop."
                )
            else:
                self.comfy_badge.setText("START INSTANCE")
                self.comfy_summary_label.setText(
                    "Comfy Desktop was found. "
                    + comfy_start_instruction(selected_installation)
                    + " MIVIM will connect automatically when it is ready."
                )
        elif desktop_detected:
            self.comfy_badge.setText("CREATE INSTANCE")
            state = "warning"
            self.comfy_summary_label.setText(
                "Comfy Desktop is installed, but it does not have a local "
                "ComfyUI instance yet. Open Desktop and finish creating or "
                "selecting one."
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
        self.setup_logs_button.setVisible(
            bool(status.error and discovery.host_candidates)
        )
        self.studio_log.info(
            "setup.check.finished",
            (
                f"state={state}; local_installations={len(local_installations)}; "
                f"api_reachable={status.reachable}; checkpoints="
                f"{len(status.checkpoints)}; candidates="
                f"{','.join(discovery.host_candidates) or 'none'}; "
                f"detail={status.error or 'connected'}"
            ),
        )
        self.save_current_settings(silent=True)
        if status.reachable:
            self._awaiting_comfy_launch = False
            self._comfy_retry_timer.stop()
            if self._return_to_readiness_when_connected:
                self._return_to_readiness_when_connected = False
                self._single_shot(0, self.review_project)
            elif self._pending_workflow_publish == self._displayed_result_record:
                self._single_shot(0, self.add_ready_workflow_to_comfyui)
        elif (
            not unavailable_selection
            and not ambiguous_runtime_identity
            and (
                self._awaiting_comfy_launch
                or (local_installations and self.pages.currentIndex() == NAV_SETUP)
            )
        ):
            self._comfy_retry_timer.start(5000)
        if (
            displayed is not None
            and self._pending_workflow_publish is None
            and not self._workflow_publish_busy
        ):
            self._render_workflow_handoff_state(displayed)
        if self._return_to_readiness_after_setup:
            self._return_to_readiness_after_setup = False
            self._single_shot(0, self.review_project)

    def _setup_failed_for(self, request: int, message: str) -> None:
        if request == self._setup_request:
            self._setup_failed(message)

    def _setup_failed(self, message: str) -> None:
        self._setup_in_progress = False
        self._comfy_retry_timer.stop()
        self.studio_log.error_event("setup.check.failed", message)
        self._return_to_readiness_when_connected = False
        self._return_to_readiness_after_setup = False
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
        local_ids = {installation.id for installation in local_installations}
        unavailable_selection = bool(self.settings.comfy_installation_id) and (
            self.settings.comfy_installation_id not in local_ids
        )
        with QSignalBlocker(self.installation_combo):
            self.installation_combo.clear()
            if unavailable_selection:
                self.installation_combo.addItem(
                    "Original instance unavailable — choose another",
                    self.settings.comfy_installation_id,
                )
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
        show_instance_picker = len(local_installations) > 1 or unavailable_selection
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
        with QSignalBlocker(self.checkpoint_combo):
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
        elif self.comfy_status is not None and not self.comfy_status.reachable:
            self.model_hint.setText(
                "Connect a running local ComfyUI instance, then choose Retry to "
                "load its model list."
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
                    f"ComfyUI will use its first matching file: {matches[0]}. "
                    f"{len(matches) - 1} lower-priority same-named copy or copies "
                    "will be ignored."
                )

    def _populate_last_project(self) -> None:
        last = self.settings.last_project
        self.project_name_edit.setText(last.project_name)
        self.trigger_edit.setText(last.trigger)
        index = self.preset_combo.findData(last.preset.value)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
        self.training_steps_spin.setValue(last.training_steps or 0)
        self._update_training_hints()
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
        if not result.exists() or not expected_kind or not record.result_fingerprint:
            return
        try:
            current_fingerprint = artifact_fingerprint(result)
        except (FileNotFoundError, OSError, ValueError):
            return
        if current_fingerprint != record.result_fingerprint:
            return
        if record.mode == "full":
            workflow = record.workflow_path
            if (
                workflow is None
                or not record.workflow_fingerprint
                or not workflow.is_file()
            ):
                return
            try:
                if artifact_fingerprint(workflow) != record.workflow_fingerprint:
                    return
            except (FileNotFoundError, OSError, ValueError):
                return
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
        self.pipeline_command = None
        self.run_stack.setCurrentIndex(1)
        self.run_progress.setValue(78)
        self.run_stage_label.setText("Checking saved validation…")
        self.run_detail_label.setText(
            "MIVIM is verifying the exact LoRA and SDXL checkpoint in the "
            "background. The window remains available while large models are checked."
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
        self.retry_validation_button.setEnabled(False)
        self._recovery_check_request += 1
        request = self._recovery_check_request
        settings_snapshot = deepcopy(self.settings)
        self._run_worker(
            lambda: self.service.validation_recovery_issue(
                settings_snapshot,
                recovery,
            ),
            lambda result: self._restored_recovery_checked(
                request,
                recovery,
                draft,
                settings_snapshot,
                result,
            ),
            lambda message: self._restored_recovery_check_failed(
                request,
                recovery,
                message,
            ),
            operation="validation.recovery.verify",
        )

    def _restored_recovery_checked(
        self,
        request: int,
        recovery: ValidationRecovery,
        draft: ProjectDraft,
        settings_snapshot: AppSettings,
        result: object,
    ) -> None:
        if (
            request != self._recovery_check_request
            or self._pending_validation_recovery != recovery
        ):
            return
        if result is None:
            recovery_issue = None
        elif isinstance(result, str):
            recovery_issue = result
        else:
            recovery_issue = (
                "Saved validation verification returned an unexpected result. "
                "Use Check again before retrying."
            )
        try:
            self.pipeline_command = (
                None
                if recovery_issue is not None
                else self.service.build_validation_command(
                    settings_snapshot,
                    draft,
                    recovery.lora_name,
                    checkpoint=recovery.checkpoint,
                    strengths=recovery.strengths,
                    seed=recovery.seed,
                )
            )
        except (TypeError, ValueError) as exc:
            self.pipeline_command = None
            recovery_issue = str(exc)
        if recovery_issue is None:
            self.run_stage_label.setText("LoRA installed; validation needs attention")
            self.run_detail_label.setText(
                "Training is already complete and the exact model files were "
                "verified. Open the selected ComfyUI instance, then retry validation "
                "without repeating photo preparation or training."
            )
            self.retry_validation_button.setEnabled(True)
        else:
            self.run_stage_label.setText("Reconnect the original ComfyUI setup")
            self.run_detail_label.setText(recovery_issue)
            self.retry_validation_button.setEnabled(False)

    def _restored_recovery_check_failed(
        self,
        request: int,
        recovery: ValidationRecovery,
        message: str,
    ) -> None:
        if (
            request != self._recovery_check_request
            or self._pending_validation_recovery != recovery
        ):
            return
        self.pipeline_command = None
        self.run_stage_label.setText("Saved validation could not be verified")
        self.run_detail_label.setText(
            "MIVIM could not safely read the saved LoRA and checkpoint. Reconnect "
            "the original files, then check again."
        )
        self.retry_validation_button.setEnabled(False)
        self.studio_log.warning("validation.recovery.verify_failed", message)

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
            model_profile=str(self.model_profile_combo.currentData()),
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

    def choose_lora_dir(self) -> bool:
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
            return True
        return False

    def create_verified_lora_folder(self) -> bool:
        """Create only the missing LoRA directory proven by selected discovery."""

        settings = self._collect_settings()
        target = verified_missing_lora_directory(settings, self.discovery)
        if target is None:
            self._show_error(
                "LoRA folder was not created",
                "The selected Comfy Desktop model location could not be verified. "
                "Refresh the connection or choose its existing LoRA folder.",
            )
            return False
        try:
            target.mkdir(parents=False, exist_ok=False)
        except OSError as exc:
            self._show_error(
                "LoRA folder was not created",
                f"Windows could not create the verified folder:\n{target}\n\n{exc}",
            )
            return False
        self.lora_dir_edit.setText(str(target))
        self.save_current_settings(silent=True)
        self.studio_log.info(
            "setup.lora_folder.created",
            f"Created the selected Desktop instance's LoRA folder: {target}",
        )
        self._invalidate_readiness()
        return True

    def choose_base_model(self) -> bool:
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
            return True
        return False

    def choose_toolkit_dir(self) -> bool:
        start = self.toolkit_dir_edit.text().strip() or str(Path.home())
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose ai-toolkit folder",
            start,
            QFileDialog.Option.ShowDirsOnly,
        )
        if not selected:
            return False
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
        return True

    def choose_toolkit_python(self) -> bool:
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
            return True
        return False

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
            if self.choose_toolkit_dir():
                self._complete_readiness_repair()
        elif clicked is choose_python:
            if self.choose_toolkit_python():
                self._complete_readiness_repair()
        elif clicked is toolkit_guide:
            QDesktopServices.openUrl(QUrl("https://github.com/ostris/ai-toolkit"))
        elif clicked is pytorch_guide:
            QDesktopServices.openUrl(QUrl("https://pytorch.org/get-started/locally/"))
        elif clicked is check_again:
            self.review_project()

    def choose_workspace_dir(self) -> bool:
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
            return True
        return False

    def open_comfy_desktop(self) -> bool:
        local_installations = (
            tuple(
                installation
                for installation in self.discovery.installations
                if installation.is_local
            )
            if self.discovery is not None
            else ()
        )
        desktop_detected = self.discovery is not None and self.discovery.detected
        if not local_installations and not desktop_detected:
            if not QDesktopServices.openUrl(
                QUrl("https://docs.comfy.org/installation/desktop/windows")
            ):
                self._show_error(
                    "Could not open the install guide",
                    "Open https://docs.comfy.org/installation/desktop/windows "
                    "in your browser.",
                )
            return False
        if self.discovery is None or not self.discovery.launcher.supported:
            self._show_error(
                "Comfy Desktop is not available",
                "Windows could not resolve the Comfy Desktop Start Menu application.",
            )
            return False
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
            return False
        self.comfy_summary_label.setText(
            comfy_start_instruction(self.discovery.selected_installation)
            + " MIVIM will connect automatically when it is ready."
        )
        self._awaiting_comfy_launch = True
        self._comfy_retry_timer.start(3000)
        return True

    def open_result_comfy_desktop(self) -> None:
        issue = self._result_comfy_context_issue(self._displayed_result_record)
        if issue is not None:
            self.show_page(NAV_SETUP)
        self.open_comfy_desktop()
        if issue is not None and self._awaiting_comfy_launch:
            self.comfy_summary_label.setText(
                f"{issue} Comfy Desktop is opening so you can reconnect it."
            )

    def _start_result_identity_verification(self, record: RunResult) -> None:
        """Hash restored model files off the GUI thread before handoff actions."""

        if (
            record.mode != "full"
            or record.installed_lora_path is None
            or record.checkpoint_path is None
        ):
            return
        lora_path = record.installed_lora_path
        checkpoint_path = record.checkpoint_path
        cached = self._verified_result_identity
        if cached is not None and cached[0] == record:
            try:
                if (
                    _file_stat_identity(lora_path) == cached[1]
                    and _file_stat_identity(checkpoint_path) == cached[2]
                ):
                    self._render_workflow_handoff_state(record)
                    return
            except OSError:
                pass
        self._result_identity_request += 1
        request = self._result_identity_request
        self._verified_result_identity = None
        self._result_identity_failure = None
        self.result_open_comfy_button.setEnabled(False)
        self._render_workflow_handoff_state(record)

        def verify() -> tuple[tuple[int, int], tuple[int, int]]:
            if (
                not record.lora_fingerprint
                or artifact_fingerprint(lora_path) != record.lora_fingerprint
            ):
                raise ValueError(
                    "The installed LoRA no longer matches this validation result."
                )
            if (
                not record.checkpoint_fingerprint
                or artifact_fingerprint(checkpoint_path)
                != record.checkpoint_fingerprint
            ):
                raise ValueError(
                    "The SDXL checkpoint no longer matches this validation result."
                )
            return (
                _file_stat_identity(lora_path),
                _file_stat_identity(checkpoint_path),
            )

        def verified(result: object) -> None:
            if (
                request != self._result_identity_request
                or self._displayed_result_record != record
                or not isinstance(result, tuple)
                or len(result) != 2
            ):
                return
            lora_identity, checkpoint_identity = result
            if not (
                isinstance(lora_identity, tuple)
                and len(lora_identity) == 2
                and isinstance(checkpoint_identity, tuple)
                and len(checkpoint_identity) == 2
            ):
                self._result_identity_failed(
                    request,
                    record,
                    "Model verification returned an unexpected value.",
                )
                return
            self._verified_result_identity = (
                record,
                cast(tuple[int, int], lora_identity),
                cast(tuple[int, int], checkpoint_identity),
            )
            self._result_identity_failure = None
            self.result_open_comfy_button.setEnabled(
                self.comfy_status is not None
                and self.comfy_status.reachable
                and self._result_comfy_context_issue(record) is None
            )
            self._render_workflow_handoff_state(record)
            self.studio_log.info(
                "result.models.verified",
                f"project={record.project_name}",
            )

        self._run_worker(
            verify,
            verified,
            lambda message: self._result_identity_failed(
                request,
                record,
                message,
            ),
            operation="result.models.verify",
        )

    def _result_identity_failed(
        self,
        request: int,
        record: RunResult,
        message: str,
    ) -> None:
        if (
            request != self._result_identity_request
            or self._displayed_result_record != record
        ):
            return
        self._verified_result_identity = None
        self._result_identity_failure = (record, message)
        self.result_open_comfy_button.setEnabled(False)
        self._render_workflow_handoff_state(record)
        self.studio_log.warning(
            "result.models.unverified",
            f"project={record.project_name}; detail={message}",
        )

    def _result_comfy_setup_issue(self, record: RunResult | None) -> str | None:
        """Explain why a result is not bound to one current Desktop instance."""

        if record is None or record.mode != "full":
            return None
        installation_id = record.comfy_installation_id.strip()
        installation_name = self._result_installation_name(record)
        if not installation_id:
            return (
                "This result does not record which Comfy Desktop installation "
                "created it. Open Connect and select the original instance before "
                "adding the workflow."
            )
        current = self._collect_settings()
        if installation_id != current.comfy_installation_id.strip():
            return (
                "This result belongs to another Comfy Desktop installation. "
                f'Open Connect and select "{installation_name}" before adding or '
                "opening the workflow in ComfyUI."
            )
        if self.discovery is None:
            return (
                "MIVIM has not checked the original Comfy Desktop installation "
                "yet. Open Connect and check the connection before adding the "
                "workflow."
            )
        selected = self.discovery.selected_installation
        original = next(
            (
                installation
                for installation in self.discovery.installations
                if installation.is_local and installation.id == installation_id
            ),
            None,
        )
        if original is None:
            return (
                f'The original Comfy Desktop installation "{installation_name}" '
                "is no longer registered. Open Connect and restore or explicitly "
                "replace that instance."
            )
        if selected is None or selected.id != installation_id:
            return (
                f'The original Comfy Desktop installation "{installation_name}" '
                "is not selected. Open Connect and select it before adding the "
                "workflow."
            )
        if (
            sum(
                installation.is_local
                and installation.name.casefold() == selected.name.casefold()
                for installation in self.discovery.installations
            )
            > 1
        ):
            return (
                f'More than one Desktop installation is named "{selected.name}". '
                "Rename one in Comfy Desktop, then check the connection again."
            )
        if record.installed_lora_path is not None and (
            current.comfy_lora_dir is None
            or record.installed_lora_path.resolve().parent
            != self.service.project_path(current.comfy_lora_dir).resolve()
        ):
            return (
                "The selected ComfyUI model library is not the one that received "
                "this LoRA. Open Connect and restore the original installation."
            )
        if record.checkpoint_path is not None and (
            current.training_base_model is None
            or self.service.project_path(current.training_base_model).resolve()
            != record.checkpoint_path.resolve()
            or current.comfy_checkpoint.strip() != record.checkpoint
        ):
            return (
                "A different SDXL checkpoint is selected. Open Connect and restore "
                "the exact checkpoint used for this result."
            )
        return None

    def _result_comfy_context_issue(self, record: RunResult | None) -> str | None:
        setup_issue = self._result_comfy_setup_issue(record)
        if setup_issue is not None:
            return setup_issue
        if record is None or record.mode != "full":
            return None
        if (
            record.installed_lora_path is None
            or not record.installed_lora_path.is_file()
        ):
            return "The LoRA is missing from the ComfyUI installation used by this run."
        if record.checkpoint_path is None or not record.checkpoint_path.is_file():
            return "The SDXL checkpoint used for this result is missing."
        failure = self._result_identity_failure
        if failure is not None and failure[0] == record:
            return (
                f"{failure[1]} Retry validation after restoring the original "
                "model files."
            )
        verified = self._verified_result_identity
        if verified is None or verified[0] != record:
            return (
                "MIVIM is still checking the exact LoRA and checkpoint files. "
                "Wait for verification before opening the workflow."
            )
        try:
            lora_changed = (
                _file_stat_identity(record.installed_lora_path) != verified[1]
            )
            checkpoint_changed = (
                _file_stat_identity(record.checkpoint_path) != verified[2]
            )
        except OSError:
            return "The LoRA or checkpoint could not be verified safely."
        if lora_changed or checkpoint_changed:
            return (
                "The LoRA or checkpoint changed after verification. Retry "
                "validation before using this workflow."
            )
        return None

    def _render_workflow_handoff_state(self, record: RunResult) -> None:
        """Keep the Results action and guidance aligned with the exact blocker."""

        if (
            self._displayed_result_record != record
            or record.mode != "full"
            or self._workflow_publish_busy
            or self._pending_workflow_publish is not None
        ):
            return
        setup_issue = self._result_comfy_setup_issue(record)
        if setup_issue is not None:
            self.publish_workflow_button.setText("Fix in Connect")
            self.publish_workflow_button.setEnabled(True)
            self.workflow_handoff_status_label.setText(
                f"Cannot add this workflow yet. {setup_issue}"
            )
            return

        failure = self._result_identity_failure
        if failure is not None and failure[0] == record:
            self.publish_workflow_button.setText("Check files again")
            self.publish_workflow_button.setEnabled(True)
            self.workflow_handoff_status_label.setText(
                f"Cannot add this workflow yet. {failure[1]} Restore the original "
                "files, then choose Check files again."
            )
            return

        verified = self._verified_result_identity
        if verified is None or verified[0] != record:
            self.publish_workflow_button.setText("Checking model files…")
            self.publish_workflow_button.setEnabled(False)
            self.workflow_handoff_status_label.setText(
                "Checking the exact LoRA and checkpoint before enabling the "
                "ComfyUI handoff…"
            )
            return

        context_issue = self._result_comfy_context_issue(record)
        if context_issue is not None:
            self.publish_workflow_button.setText("Check files again")
            self.publish_workflow_button.setEnabled(True)
            self.workflow_handoff_status_label.setText(
                f"Cannot add this workflow yet. {context_issue}"
            )
            return

        if (
            record.published_workflow_path
            and record.published_workflow_fingerprint == record.workflow_fingerprint
        ):
            workflow_name = (
                record.workflow_path.stem
                if record.workflow_path is not None
                else record.project_name
            )
            profile_note = (
                f" for profile {record.workflow_user_name}"
                if record.workflow_user_name
                else ""
            )
            self.publish_workflow_button.setText("Check in ComfyUI")
            self.workflow_handoff_status_label.setText(
                f"Previously added {workflow_name} under Workflows → MIVIM"
                f"{profile_note}. Check that it is still available before opening "
                "ComfyUI."
            )
        else:
            self.publish_workflow_button.setText(
                "Retry add to ComfyUI"
                if self._workflow_publish_retry_requires_refresh
                else "Add to ComfyUI"
            )
            self.workflow_handoff_status_label.setText(
                "Add the verified JSON to Local User Workflows. MIVIM will not "
                "overwrite an existing workflow."
            )
        self.publish_workflow_button.setEnabled(True)

    def open_comfy_ui(self) -> None:
        context_issue = self._result_comfy_context_issue(self._displayed_result_record)
        if context_issue is not None:
            self._show_error("Reconnect the original ComfyUI setup", context_issue)
            return
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
        self.model_profile_combo.currentIndexChanged.connect(
            self._model_profile_changed
        )
        self.sdxl_confirm_checkbox.toggled.connect(self._invalidate_readiness)
        self.vision_checkbox.toggled.connect(self._invalidate_readiness)
        self.overwrite_checkbox.toggled.connect(self._invalidate_readiness)

    def _update_model_profile_hint(self) -> None:
        try:
            profile = model_profile(str(self.model_profile_combo.currentData()))
            self.model_profile_hint.setText(profile.description)
        except ValueError as exc:
            self.model_profile_hint.setText(str(exc))

    def _model_profile_changed(self, *_args: object) -> None:
        self._update_model_profile_hint()
        self._invalidate_readiness()
        self._update_training_hints()
        if self.source_folder is not None:
            self._photo_folder_changed(self.source_folder)

    def _connection_changed(self, *_args: object) -> None:
        self.comfy_status = None
        self.open_comfy_button.setEnabled(False)
        self.result_open_comfy_button.setEnabled(False)
        self._workflow_publish_request += 1
        self._workflow_publish_busy = False
        self._pending_workflow_publish = None
        self.publish_workflow_button.setEnabled(False)
        self._invalidate_readiness()

    def _installation_changed(self, *_args: object) -> None:
        self._connection_changed()
        if self._return_to_readiness_after_installation:
            self._return_to_readiness_after_installation = False
            self._return_to_readiness_after_setup = True
            self.save_current_settings(silent=True)
        self._single_shot(0, self.refresh_setup)

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
        if self._return_to_readiness_after_checkpoint and checkpoint.strip():
            self._return_to_readiness_after_checkpoint = False
            self._single_shot(0, self.review_project)

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
        self._update_training_hints()
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
        illustrations = self.model_profile_combo.currentData() != "sdxl"
        self._source_request += 1
        request = self._source_request
        self._run_worker(
            lambda: inspect_source(
                folder, target_size=preset.image_size, illustrations=illustrations
            ),
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
        self._update_training_hints()
        self.review_photos_button.setEnabled(True)
        detail = f"{summary.usable_count} usable image(s)"
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

    def review_photo_checks(self) -> bool:
        summary = self.source_summary
        if summary is None:
            return False
        dialog = PhotoIssueDialog(summary, self)
        if dialog.exec():
            self._photo_folder_changed(summary.folder)
            return True
        return False

    def _preset_changed(self) -> None:
        preset = TrainingPreset(self.preset_combo.currentData())
        descriptions = {
            TrainingPreset.FAST: (
                "Fast draft uses rank 8, 768 px preparation, "
                "a no-LoRA baseline, and one validation strength."
            ),
            TrainingPreset.BALANCED: (
                "Balanced uses rank 16, 1024 px preparation, and "
                "a no-LoRA baseline plus three comparison strengths."
            ),
            TrainingPreset.QUALITY: (
                "Extended training uses rank 32 and a wider comparison sweep. "
                "More steps or rank do not automatically improve the result."
            ),
        }
        self.preset_hint.setText(descriptions[preset])
        self.training_steps_spin.setSpecialValueText(
            f"Plan default ({preset.training_steps:,})"
        )
        self._update_training_hints()
        self._invalidate_readiness()
        if self.source_folder is not None:
            self._photo_folder_changed(self.source_folder)

    def _training_steps_changed(self, *_args: object) -> None:
        self._update_training_hints()
        self._invalidate_readiness()

    def _update_training_hints(self) -> None:
        preset = TrainingPreset(self.preset_combo.currentData())
        steps = self.training_steps_spin.value() or preset.training_steps
        count = self.source_summary.unique_usable_count if self.source_summary else 0
        duration = f"{steps:,} training steps"
        if count:
            duration += f" · about {steps / count:,.1f} image presentations per source (batch 1)"
        self.training_duration_hint.setText(
            duration + ". Compare saved checkpoints to choose when to stop learning."
        )
        if self.source_summary:
            captioned = self.source_summary.captioned_count
            self.caption_source_hint.setText(
                f"{captioned} of {count} unique images have .txt captions to import. "
                "Missing captions use BLIP if enabled, otherwise the identity phrase. "
                "Review and edit them after preparation."
            )
        else:
            self.caption_source_hint.setText(
                "Existing .txt captions beside your images are imported automatically. "
                "Untagged images get the identity phrase; edit them after preparation."
            )
        anime = self.model_profile_combo.currentData() != "sdxl"
        self.preview_suite_hint.setText(
            "Anime previews: face, full body, profile and action · 832×1216. "
            "The same shots are used during training and in the final strength comparison. "
            "Use a unique character trigger; keep clothing, pose and background tags in captions."
            if anime
            else "Photo previews compare portraits, studio and outdoor scenes. "
            "Choose a NoobAI profile in Connect for anime character previews."
        )

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
            training_steps=self.training_steps_spin.value() or None,
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
        if (
            report.source is not None
            and self.source_folder is not None
            and report.source.folder == self.source_folder
        ):
            self.source_summary = report.source
            self.review_photos_button.setEnabled(True)
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
                    ReadinessAction(
                        ReadinessActionKind.RETRY_READINESS,
                        "Retry the check",
                    ),
                ),
                self._handle_diagnostic_action,
            )
        )
        self.readiness_summary.setText("The setup check could not finish.")

    def _complete_readiness_repair(self) -> None:
        """Persist a successful repair and verify the real path again."""

        self.review_project()

    def _next_available_project_name(self, current: str) -> str:
        """Return the next simple version whose complete artifact namespace is empty."""

        candidate = current
        settings = self._collect_settings()
        for _attempt in range(100):
            candidate = self._increment_project_name(candidate)
            artifacts = self.service.project_artifacts(settings, candidate)
            if artifacts.dataset is None and not artifacts.has_non_dataset_outputs:
                return candidate
        return self._increment_project_name(current)

    def _open_current_comfy_ui(self) -> bool:
        """Open the checked live API without applying a restored-result identity gate."""

        status = self.comfy_status
        if status is None or not status.reachable or not status.host:
            self._show_error(
                "ComfyUI is not connected",
                "Start the selected local instance and choose Retry before opening it.",
            )
            return False
        if not QDesktopServices.openUrl(QUrl(f"http://{status.host}")):
            self._show_error(
                "Could not open ComfyUI",
                "Windows could not open the local ComfyUI address in a browser.",
            )
            return False
        return True

    def open_storage_settings(self) -> bool:
        """Open the Windows storage cleanup surface; never disguise a folder picker."""

        if not QDesktopServices.openUrl(QUrl("ms-settings:storagesense")):
            self._show_error(
                "Windows Storage settings could not be opened",
                "Open Windows Settings → System → Storage, free some space, then "
                "return here and choose Check again.",
            )
            return False
        return True

    def _handle_diagnostic_action(self, item: ReadinessItem) -> None:
        action = item.action
        if action is None:
            return
        kind = action.kind
        self.studio_log.info(
            "readiness.action",
            f"code={item.code}; action={kind.value}",
        )

        if kind is ReadinessActionKind.EDIT_PROJECT_NAME:
            self.show_page(NAV_PROJECT)
            self.project_name_edit.setFocus()
        elif kind is ReadinessActionKind.USE_NEXT_PROJECT_NAME:
            current = action.payload or self.project_name_edit.text().strip()
            self.project_name_edit.setText(self._next_available_project_name(current))
            self.overwrite_checkbox.setChecked(False)
            self._complete_readiness_repair()
        elif kind is ReadinessActionKind.DISABLE_REBUILD:
            self.overwrite_checkbox.setChecked(False)
            self._complete_readiness_repair()
        elif kind is ReadinessActionKind.RESTORE_TRIGGER:
            self.trigger_edit.setText(action.payload or config.DEFAULT_TRIGGER)
            self._complete_readiness_repair()
        elif kind is ReadinessActionKind.REVIEW_PHOTOS:
            self.show_page(NAV_PROJECT)
            if self.review_photo_checks():
                self._complete_readiness_repair()
        elif kind is ReadinessActionKind.CHOOSE_PHOTO_FOLDER:
            self.show_page(NAV_PROJECT)
            if self.photo_drop._browse():
                self._complete_readiness_repair()
        elif kind is ReadinessActionKind.REVIEW_PREPARED:
            self.review_prepared_dataset()
            self._complete_readiness_repair()
        elif kind is ReadinessActionKind.DISABLE_VISION_CAPTIONS:
            self.vision_checkbox.setChecked(False)
            self._complete_readiness_repair()
        elif kind is ReadinessActionKind.OPEN_COMFY_DESKTOP:
            self.show_page(NAV_SETUP)
            if self.open_comfy_desktop():
                self._return_to_readiness_when_connected = True
        elif kind is ReadinessActionKind.SELECT_COMFY_INSTALLATION:
            self.show_page(NAV_SETUP)
            self._return_to_readiness_after_installation = True
            self.installation_combo.setFocus()
            if self.installation_combo.isVisible():
                self.installation_combo.showPopup()
        elif kind is ReadinessActionKind.OPEN_COMFYUI:
            self.show_page(NAV_SETUP)
            self._open_current_comfy_ui()
        elif kind is ReadinessActionKind.SELECT_CHECKPOINT:
            self.show_page(NAV_SETUP)
            self._return_to_readiness_after_checkpoint = True
            self.checkpoint_combo.setFocus()
            self.checkpoint_combo.showPopup()
        elif kind is ReadinessActionKind.SELECT_MODEL_PROFILE:
            self.show_page(NAV_SETUP)
            self.model_profile_combo.setFocus()
            self.model_profile_combo.showPopup()
        elif kind is ReadinessActionKind.CONFIRM_CHECKPOINT:
            self.show_page(NAV_SETUP)
            self.confirm_sdxl_checkpoint()
        elif kind is ReadinessActionKind.CHOOSE_BASE_MODEL:
            self.show_page(NAV_SETUP)
            if self.choose_base_model():
                self._complete_readiness_repair()
        elif kind is ReadinessActionKind.CREATE_LORA_FOLDER:
            self.show_page(NAV_SETUP)
            if self.create_verified_lora_folder():
                self._complete_readiness_repair()
        elif kind is ReadinessActionKind.CHOOSE_LORA_FOLDER:
            self.show_page(NAV_SETUP)
            if self.choose_lora_dir():
                self._complete_readiness_repair()
        elif kind is ReadinessActionKind.CHOOSE_TOOLKIT_FOLDER:
            self.show_page(NAV_SETUP)
            if self.choose_toolkit_dir():
                self._complete_readiness_repair()
        elif kind is ReadinessActionKind.CHOOSE_TOOLKIT_PYTHON:
            self.show_page(NAV_SETUP)
            if self.choose_toolkit_python():
                self._complete_readiness_repair()
        elif kind is ReadinessActionKind.REPAIR_TRAINER:
            self.show_page(NAV_SETUP)
            self.show_trainer_help()
        elif kind is ReadinessActionKind.CHOOSE_WORKSPACE:
            self.show_page(NAV_SETUP)
            if self.choose_workspace_dir():
                self._complete_readiness_repair()
        elif kind is ReadinessActionKind.OPEN_STORAGE_SETTINGS:
            self.open_storage_settings()
        elif kind is ReadinessActionKind.RETRY_READINESS:
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
            self._complete_readiness_repair()
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
        self._run_settings_snapshot = deepcopy(active_settings)
        self._run_draft_snapshot = deepcopy(active_draft)
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
        self._owned_comfy_prompt = None
        self._comfy_cancel_detail = ""
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
            (
                f"mode={command.mode}; program={command.program}; "
                f"cwd={command.working_directory}; "
                f"expected_result={command.expected_result}"
            ),
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
        if self._desktop_owner_lifetime.name:
            environment.insert(
                DESKTOP_OWNER_MUTEX_ENV,
                self._desktop_owner_lifetime.name,
            )
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
        owned_prompt = re.fullmatch(
            r"\[comfy\] owned prompt queued "
            r"prompt_id=([0-9A-Fa-f-]+) "
            r"client_id=([0-9A-Fa-f-]+) host=(\S+)",
            line.strip(),
        )
        if owned_prompt is not None:
            prompt_id, client_id, host = owned_prompt.groups()
            command = self.pipeline_command
            expected_host = (
                command.environment.get("COMFY_HOST", "").strip()
                if command is not None
                else ""
            )
            try:
                prompt_id = str(uuid.UUID(prompt_id))
                client_id = str(uuid.UUID(client_id))
            except ValueError:
                self.studio_log.warning(
                    "comfy.prompt.marker_rejected",
                    "The worker reported invalid prompt ownership identifiers.",
                )
            else:
                if host == expected_host:
                    self._owned_comfy_prompt = (prompt_id, client_id, host)
                else:
                    self.studio_log.warning(
                        "comfy.prompt.marker_rejected",
                        "The worker prompt host did not match the selected run host.",
                    )
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
        command = self.pipeline_command
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Stop this run?")
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText(
            "MIVIM will safely stop the current step when it can prove that the "
            "ComfyUI job belongs to this run. Prepared photos and completed "
            "checkpoints are kept. If this ComfyUI version cannot stop only that "
            "job, MIVIM will keep watching and tell you how to stop it in ComfyUI."
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
        owned_prompt = self._owned_comfy_prompt
        if owned_prompt is None:
            self._stop_process_tree(force=False)
            self._single_shot(
                3500,
                lambda: self._force_stop_pipeline_for_attempt(process, command),
            )
            return
        prompt_id, client_id, host = owned_prompt
        self.run_detail_label.setText(
            "Asking ComfyUI to stop only this run's verified prompt…"
        )
        self._run_worker(
            lambda: ComfyClient(host=host, timeout=2.5).cancel_owned_prompt(
                prompt_id,
                client_id,
            ),
            lambda result: self._owned_prompt_cancelled_for_attempt(
                process,
                command,
                result,
            ),
            lambda message: self._owned_prompt_cancel_failed_for_attempt(
                process,
                command,
                message,
            ),
            operation="comfy.prompt.cancel",
        )

    def _pipeline_attempt_is_active(
        self,
        process: QProcess,
        command: PipelineCommand | None,
    ) -> bool:
        """Confirm an asynchronous stop still belongs to the active run."""

        try:
            process.state()
        except RuntimeError:
            # The Python wrapper can outlive its Qt object until a delayed
            # single-shot callback runs.
            return False
        if self.pipeline_process is not process:
            return False
        if command is None:
            return self.pipeline_command is None
        active_command = self.pipeline_command
        return (
            active_command is not None
            and active_command is command
            and active_command.attempt_id == command.attempt_id
        )

    def _owned_prompt_cancelled_for_attempt(
        self,
        process: QProcess,
        command: PipelineCommand | None,
        result: object,
    ) -> None:
        if not self._pipeline_attempt_is_active(process, command):
            return
        if not isinstance(result, PromptCancellationResult):
            self._owned_prompt_cancel_failed_for_attempt(
                process,
                command,
                "Unexpected ComfyUI cancellation result.",
            )
            return
        if not result.requested and result.status != "not_found":
            self._owned_prompt_still_running(
                result.message,
                status=result.status,
            )
            return
        self._owned_prompt_cancelled(result)
        self._single_shot(
            3500,
            lambda: self._force_stop_pipeline_for_attempt(process, command),
        )

    def _owned_prompt_cancel_failed_for_attempt(
        self,
        process: QProcess,
        command: PipelineCommand | None,
        message: str,
    ) -> None:
        if not self._pipeline_attempt_is_active(process, command):
            return
        self._owned_prompt_cancel_failed(message)

    def _owned_prompt_cancelled(self, result: object) -> None:
        if not isinstance(result, PromptCancellationResult):
            self._owned_prompt_cancel_failed("Unexpected ComfyUI cancellation result.")
            return
        self._comfy_cancel_detail = result.message
        self.run_log.appendPlainText(f"[studio] {result.message}")
        log = self.studio_log.info if result.requested else self.studio_log.warning
        log(
            "comfy.prompt.cancel.finished",
            f"status={result.status}; prompt_id={result.prompt_id}",
        )
        self._refresh_cancelled_detail()
        self._stop_process_tree(force=False)

    def _owned_prompt_cancel_failed(self, message: str) -> None:
        detail = (
            "MIVIM could not confirm that the ComfyUI prompt stopped. Check the "
            "ComfyUI queue and stop this job there. MIVIM is still watching it."
        )
        self._owned_prompt_still_running(
            detail,
            status="verification_failed",
            technical_detail=message,
        )

    def _owned_prompt_still_running(
        self,
        detail: str,
        *,
        status: str,
        technical_detail: str = "",
    ) -> None:
        """Keep the worker attached when a safe prompt-specific stop is unavailable."""

        self.run_cancelled = False
        self._comfy_cancel_detail = detail
        self.run_stage_label.setText("ComfyUI job is still running")
        self.run_detail_label.setText(
            f"{detail} MIVIM will keep watching this run. Do not start another "
            "GPU task until ComfyUI shows that it stopped."
        )
        self.run_log.appendPlainText(
            f"[studio] {detail}"
            + (
                f"\n[studio] Technical detail: {technical_detail}"
                if technical_detail
                else ""
            )
        )
        self.studio_log.warning(
            "comfy.prompt.cancel.left_running",
            f"status={status}"
            + (f"; detail={technical_detail}" if technical_detail else ""),
        )

    def _refresh_cancelled_detail(self) -> None:
        if self.run_cancelled and self._run_terminal_handled:
            self.run_detail_label.setText(
                f"{self._comfy_cancel_detail} Completed files were kept. "
                "Return to the review when you are ready."
            )

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
        if process is None:
            return
        try:
            running = process.state() != QProcess.ProcessState.NotRunning
        except RuntimeError:
            return
        if running:
            self._stop_process_tree(force=True)

    def _force_stop_pipeline_for_attempt(
        self,
        process: QProcess,
        command: PipelineCommand | None,
    ) -> None:
        if not self._pipeline_attempt_is_active(process, command):
            return
        self._force_stop_pipeline()

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
        command = self.pipeline_command
        if (
            recovery is None
            or recovery.state != "validation_pending"
            or recovery == self._recovery_before_run
            or command is None
            or recovery.attempt_id != command.attempt_id
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

        output = self._pipeline_recent_output.casefold()
        if "mivim_run_busy" in output:
            return "comfy_busy"
        if recovery is not None:
            return "validation"
        if (
            "another mivim run is already using the accelerator" in output
            or "another mivim run is already working on project" in output
            or "another mivim run is already using a file or folder" in output
        ):
            return "busy"
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
                (f"{self._comfy_cancel_detail} " if self._comfy_cancel_detail else "")
                + "Completed files were kept. Return to the review when you are ready."
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
            if failure_kind == "comfy_busy":
                self.run_stage_label.setText("ComfyUI is finishing earlier MIVIM work")
                if (
                    command.mode == "validate"
                    and self._pending_validation_recovery is not None
                ):
                    self.run_detail_label.setText(
                        "No new validation work was started. Open the selected "
                        "ComfyUI instance and let its earlier MIVIM job finish, or "
                        "stop that job there, then choose Retry validation only. "
                        "Photos and training will not repeat."
                    )
                else:
                    self.run_detail_label.setText(
                        "No new work was started. Open the selected ComfyUI instance "
                        "and let its earlier MIVIM job finish, or stop that job "
                        "there, then choose Check and retry."
                    )
                self._stop_training_progress(0)
            elif failure_kind == "busy":
                self.run_stage_label.setText("Another MIVIM run is active")
                if (
                    command.mode == "validate"
                    and self._pending_validation_recovery is not None
                ):
                    self.run_detail_label.setText(
                        "No new validation work was started. Let the other run "
                        "finish, or stop it safely from its own MIVIM window, then "
                        "choose Retry validation only. Photos and training will not "
                        "repeat."
                    )
                else:
                    self.run_detail_label.setText(
                        "No new work was started. Let the other run finish, or stop "
                        "it safely from its own MIVIM window, then choose Check and "
                        "retry."
                    )
                self._stop_training_progress(0)
            elif failure_kind == "validation":
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
        settings_snapshot = deepcopy(
            self._run_settings_snapshot or self._collect_settings()
        )
        draft_snapshot = deepcopy(
            self._run_draft_snapshot
            or self._validated_draft
            or self.draft
            or self._build_draft()
        )
        self.run_stage_label.setText("Verifying exact result files…")
        self.run_detail_label.setText(
            "MIVIM is checking the grid, workflow, LoRA, and checkpoint before "
            "approving this result."
        )
        self._stop_training_progress(98)
        for button in self.nav_buttons:
            button.setEnabled(False)
        self._run_worker(
            lambda: self._record_run_result(
                command,
                settings_snapshot=settings_snapshot,
                draft_snapshot=draft_snapshot,
                persist=False,
            ),
            lambda result: self._run_result_verified(command, result),
            lambda message: self._run_result_verification_failed(command, message),
            operation="pipeline.result.verify",
        )

    def _run_result_verification_failed(
        self,
        command: PipelineCommand,
        message: str,
    ) -> None:
        if self.pipeline_command is not command:
            return
        self.studio_log.error_event(
            "pipeline.result.identity_mismatch",
            message,
        )
        self._restore_navigation()
        self.run_stage_label.setText("Result could not be verified")
        self.run_detail_label.setText(
            "The worker finished, but its completion record or exact model files "
            "did not match this run. The files were kept; retry validation before "
            "using the result."
        )
        self._show_recovery_actions(True, "generic")

    def _run_result_verified(
        self,
        command: PipelineCommand,
        result: object,
    ) -> None:
        if self.pipeline_command is not command:
            return
        if not isinstance(result, RunResult):
            self._run_result_verification_failed(
                command,
                "The result verifier returned an unexpected value.",
            )
            return
        record = result
        self._persist_run_result(record)
        if (
            record.mode == "full"
            and record.installed_lora_path is not None
            and record.checkpoint_path is not None
        ):
            try:
                self._verified_result_identity = (
                    record,
                    _file_stat_identity(record.installed_lora_path),
                    _file_stat_identity(record.checkpoint_path),
                )
                self._result_identity_failure = None
            except OSError:
                self._verified_result_identity = None
        self._restore_navigation()
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
        self._show_result(command.expected_result, record)
        self._single_shot(450, lambda: self.show_page(NAV_RESULTS))

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
        run_is_busy = kind in {"busy", "comfy_busy"}
        self.edit_after_run_button.setVisible(visible and not run_is_busy)
        self.open_run_folder_button.setVisible(visible and not run_is_busy)
        self.use_fast_after_run_button.setVisible(visible and kind == "oom")
        self.repair_trainer_after_run_button.setVisible(visible and kind == "trainer")
        self.open_desktop_after_run_button.setVisible(
            visible and kind in {"validation", "comfy_busy"}
        )
        validation_recovery_available = (
            self._pending_validation_recovery is not None
            and self._installed_lora_name is not None
        )
        self.retry_after_run_button.setVisible(
            visible and kind != "validation" and not validation_recovery_available
        )
        self.retry_validation_button.setVisible(
            visible and validation_recovery_available
        )

    def use_fast_after_oom(self) -> None:
        """Apply the lower-memory plan only after visible rebuild approval."""

        box = QMessageBox(self)
        box.setWindowTitle("Use the Fast plan and rebuild?")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText("Changing from this plan to Fast requires rebuilding this project.")
        box.setInformativeText(
            "Your source photos stay untouched. MIVIM may replace this project's "
            "prepared copies, checkpoints, validation files, result record, and "
            "installed ComfyUI LoRA. Existing installed LoRA bytes are preserved "
            "as a previous copy when applicable."
        )
        approve = box.addButton(
            "Use Fast and rebuild this project",
            QMessageBox.ButtonRole.AcceptRole,
        )
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is not approve:
            self.studio_log.info(
                "pipeline.recovery.fast.cancelled",
                "The user did not approve rebuilding this project.",
            )
            return

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
            self._pending_validation_recovery = current
            self.retry_validation_button.setEnabled(False)
            self.run_stage_label.setText("Checking exact model files…")
            self.run_detail_label.setText(
                "MIVIM is verifying the installed LoRA and checkpoint before "
                "resuming only the unfinished ComfyUI validation."
            )
            self._recovery_check_request += 1
            request = self._recovery_check_request
            self._run_worker(
                lambda: self.service.validation_recovery_issue(settings, current),
                lambda result: self._retry_validation_checked(
                    request,
                    settings,
                    current,
                    result,
                ),
                lambda message: self._restored_recovery_check_failed(
                    request,
                    current,
                    message,
                ),
                operation="validation.retry.verify",
            )
            return
        validated_settings = self._validated_settings
        draft = self._validated_draft
        lora_name = self._installed_lora_name
        if validated_settings is None or draft is None or lora_name is None:
            self._show_error(
                "Validation cannot be retried yet",
                "No installed LoRA from this run was found.",
            )
            return
        self._launch_validation_retry(validated_settings, draft, lora_name)

    def _retry_validation_checked(
        self,
        request: int,
        settings: AppSettings,
        recovery: ValidationRecovery,
        result: object,
    ) -> None:
        if (
            request != self._recovery_check_request
            or self._pending_validation_recovery != recovery
        ):
            return
        if result is not None:
            issue = (
                result
                if isinstance(result, str)
                else "Validation verification returned an unexpected result."
            )
            self.retry_validation_button.setEnabled(False)
            self.run_stage_label.setText("Reconnect the original ComfyUI setup")
            self.run_detail_label.setText(issue)
            self._show_error("Reconnect the original ComfyUI setup", issue)
            return
        self._launch_validation_retry(
            settings,
            self._draft_for_recovery(recovery),
            recovery.lora_name,
            recovery=recovery,
        )

    def _launch_validation_retry(
        self,
        settings: AppSettings,
        draft: ProjectDraft,
        lora_name: str,
        *,
        recovery: ValidationRecovery | None = None,
    ) -> None:
        try:
            command = self.service.build_validation_command(
                settings,
                draft,
                lora_name,
                checkpoint=(recovery.checkpoint if recovery is not None else None),
                strengths=(recovery.strengths if recovery is not None else None),
                seed=(recovery.seed if recovery is not None else 42),
            )
            if recovery is not None and recovery.training_project_name:
                command = replace(
                    command,
                    arguments=(
                        *command.arguments,
                        "--training-project",
                        recovery.training_project_name,
                    ),
                )
        except (TypeError, ValueError) as exc:
            self.retry_validation_button.setEnabled(True)
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
    def _record_run_result(
        self,
        command: PipelineCommand,
        *,
        settings_snapshot: AppSettings | None = None,
        draft_snapshot: ProjectDraft | None = None,
        persist: bool = True,
    ) -> RunResult:
        """Verify one worker outcome against the immutable launch context."""

        draft = (
            draft_snapshot
            or self._run_draft_snapshot
            or self._validated_draft
            or self.draft
            or self._build_draft()
        )
        mode: Literal["prepare", "full"] = (
            "prepare" if command.mode == "prepare" else "full"
        )
        current_settings = (
            settings_snapshot or self._run_settings_snapshot or self._collect_settings()
        )
        outcome = self.service.read_pipeline_outcome(
            current_settings,
            draft.project_name,
        )
        if outcome is None:
            raise ValueError("The worker completion record is missing or unreadable.")
        actual_fingerprint = artifact_fingerprint(command.expected_result)
        expected_checkpoint = command.environment.get("COMFY_SDXL_CKPT", "").strip()
        mismatches: list[str] = []
        if not command.attempt_id or outcome.attempt_id != command.attempt_id:
            mismatches.append("attempt identifier")
        if outcome.mode != command.mode:
            mismatches.append("run mode")
        if (
            self.service.project_path(outcome.result_path).resolve()
            != command.expected_result.resolve()
        ):
            mismatches.append("result path")
        if (
            not outcome.result_fingerprint
            or outcome.result_fingerprint != actual_fingerprint
        ):
            mismatches.append("result contents")
        if outcome.trigger != draft.trigger.strip():
            mismatches.append("trigger phrase")
        if outcome.preset != draft.preset.value:
            mismatches.append("training plan")
        if "--training-project" in command.arguments:
            parent_index = command.arguments.index("--training-project") + 1
            if outcome.training_project_name != command.arguments[parent_index]:
                mismatches.append("original training project")
        if "--expected-lora-fingerprint" in command.arguments:
            choice_index = command.arguments.index("--expected-lora-fingerprint") + 1
            if outcome.lora_fingerprint != command.arguments[choice_index]:
                mismatches.append("selected training checkpoint")
        if expected_checkpoint and outcome.checkpoint != expected_checkpoint:
            mismatches.append("checkpoint")
        if outcome.model_profile != command.environment.get(
            "MIVIM_MODEL_PROFILE", "sdxl"
        ):
            mismatches.append("model profile")
        if command.mode != "prepare" and (
            not outcome.lora_name
            or Path(outcome.lora_name).name != outcome.lora_name
            or Path(outcome.lora_name).suffix.casefold() != ".safetensors"
        ):
            mismatches.append("installed LoRA")
        workflow_fingerprint = ""
        if command.mode != "prepare":
            expected_workflow = command.expected_result.parent / (
                f"{draft.project_name}_use_in_comfyui.json"
            )
            if (
                outcome.workflow_path is None
                or outcome.workflow_path.resolve() != expected_workflow.resolve()
                or not expected_workflow.is_file()
            ):
                mismatches.append("ready workflow path")
            else:
                workflow_fingerprint = artifact_fingerprint(expected_workflow)
                if (
                    not outcome.workflow_fingerprint
                    or workflow_fingerprint != outcome.workflow_fingerprint
                ):
                    mismatches.append("ready workflow contents")
            expected_installation = command.environment.get(
                "MIVIM_COMFY_INSTALLATION_ID",
                "",
            ).strip()
            expected_host = (
                command.environment.get("COMFY_HOST", "")
                .strip()
                .removeprefix("http://")
                .removeprefix("https://")
                .rstrip("/")
            )
            if outcome.comfy_installation_id != expected_installation:
                mismatches.append("Comfy Desktop installation")
            if outcome.comfy_host != expected_host:
                mismatches.append("ComfyUI address")
            expected_lora_dir = current_settings.comfy_lora_dir
            actual_lora_fingerprint = ""
            if (
                expected_lora_dir is None
                or outcome.installed_lora_path is None
                or outcome.installed_lora_path.resolve()
                != (
                    self.service.project_path(expected_lora_dir) / outcome.lora_name
                ).resolve()
                or not outcome.lora_fingerprint
            ):
                mismatches.append("installed LoRA identity")
            elif outcome.installed_lora_path is not None:
                try:
                    actual_lora_fingerprint = artifact_fingerprint(
                        outcome.installed_lora_path
                    )
                except (FileNotFoundError, OSError, ValueError):
                    actual_lora_fingerprint = ""
                if actual_lora_fingerprint != outcome.lora_fingerprint:
                    mismatches.append("installed LoRA contents")
            expected_base = current_settings.training_base_model
            actual_checkpoint_fingerprint = ""
            if (
                expected_base is None
                or outcome.checkpoint_path is None
                or outcome.checkpoint_path.resolve()
                != self.service.project_path(expected_base).resolve()
                or not outcome.checkpoint_fingerprint
            ):
                mismatches.append("checkpoint identity")
            elif outcome.checkpoint_path is not None:
                try:
                    actual_checkpoint_fingerprint = artifact_fingerprint(
                        outcome.checkpoint_path
                    )
                except (FileNotFoundError, OSError, ValueError):
                    actual_checkpoint_fingerprint = ""
                if actual_checkpoint_fingerprint != outcome.checkpoint_fingerprint:
                    mismatches.append("checkpoint contents")
        if mismatches:
            raise ValueError(
                "The worker completion record did not match the "
                + ", ".join(mismatches)
                + " for this run."
            )
        installation_name = (
            next(
                (
                    installation.name
                    for installation in self.discovery.installations
                    if installation.id == outcome.comfy_installation_id
                ),
                "",
            )
            if self.discovery is not None
            else ""
        )
        record = RunResult(
            project_name=draft.project_name,
            training_project_name=outcome.training_project_name,
            mode=mode,
            result_path=command.expected_result,
            completed_at=outcome.completed_at,
            trigger=outcome.trigger,
            preset=draft.preset,
            checkpoint=outcome.checkpoint,
            model_profile=outcome.model_profile,
            lora_name=outcome.lora_name,
            result_fingerprint=actual_fingerprint,
            workflow_path=outcome.workflow_path,
            workflow_fingerprint=workflow_fingerprint,
            comfy_installation_id=outcome.comfy_installation_id,
            comfy_installation_name=installation_name,
            comfy_host=outcome.comfy_host,
            installed_lora_path=outcome.installed_lora_path,
            lora_fingerprint=outcome.lora_fingerprint,
            checkpoint_path=outcome.checkpoint_path,
            checkpoint_fingerprint=outcome.checkpoint_fingerprint,
            starting_strength=outcome.starting_strength,
        )
        if persist:
            self._persist_run_result(record)
        return record

    def _persist_run_result(self, record: RunResult) -> None:
        """Publish a verified record on the UI thread without replacing setup."""

        self._installed_lora_name = record.lora_name or self._installed_lora_name
        self.settings = replace(self._collect_settings(), last_result=record)
        try:
            save_settings(self.settings, self.settings_path)
        except (OSError, TypeError, ValueError) as exc:
            self.run_log.appendPlainText(
                f"\n[studio] The verified result could not be added to history: {exc}"
            )

    def _show_result(self, result: Path, record: RunResult | None = None) -> None:
        self._latest_result = result
        previous_record = self._displayed_result_record
        if record is None and self.settings.last_result is not None:
            stored_path = self.service.project_path(
                self.settings.last_result.result_path
            )
            if stored_path == result:
                record = self.settings.last_result
        self._displayed_result_record = record
        if previous_record != record:
            self._workflow_publish_request += 1
            self._workflow_publish_busy = False
            self._workflow_publish_retry_requires_refresh = False
        if (
            self._published_workflow is not None
            and self._published_workflow[0] != record
        ):
            self._published_workflow = None
        if self._pending_workflow_publish != record:
            self._pending_workflow_publish = None
        self.nav_buttons[NAV_RESULTS].setEnabled(True)
        self.result_path_label.setText(str(result))
        self.result_path_label.setCursorPosition(0)
        self.result_image.clear_source_pixmap()
        self.compare_checkpoints_button.setVisible(
            record is not None and record.mode == "full"
        )
        self.result_image.setStyleSheet("")
        self.comfy_handoff_card.setVisible(False)
        self.review_prepared_button.setVisible(False)
        self.continue_training_button.setVisible(False)
        self.open_full_grid_button.setVisible(False)
        self.open_ready_workflow_button.setVisible(False)
        self.publish_workflow_button.setVisible(False)
        self.publish_workflow_button.setEnabled(False)
        self.publish_workflow_button.setText("Add to ComfyUI")
        self.workflow_handoff_status_label.setVisible(False)
        self.result_open_desktop_button.setVisible(False)
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
                    self.comfy_status is not None
                    and self.comfy_status.reachable
                    and self._result_comfy_context_issue(record) is None
                )
                if record is not None:
                    workflow_ready = (
                        record.workflow_path is not None
                        and record.workflow_path.is_file()
                        and bool(record.workflow_fingerprint)
                    )
                    self.open_ready_workflow_button.setVisible(workflow_ready)
                    self.publish_workflow_button.setVisible(workflow_ready)
                    self.workflow_handoff_status_label.setVisible(workflow_ready)
                    if workflow_ready:
                        self._render_workflow_handoff_state(record)
                    self.result_open_desktop_button.setVisible(True)
                    self.result_details_label.setText(
                        f"LoRA: {record.lora_name or 'See the result folder'}\n"
                        f"Identity phrase: {record.trigger}\n"
                        f"Base checkpoint: {record.checkpoint or 'Not recorded'}\n"
                        f"Tested starting strength: {record.starting_strength:g}\n"
                        "Ready workflow: "
                        f"{record.workflow_path.name if record.workflow_path else 'Not recorded'}\n"
                        "Use the guided action here, or fall back to Workflow → Open."
                    )
                    self.comfy_handoff_card.setVisible(True)
                self.result_image.set_source_pixmap(pixmap)
                if record is not None:
                    self._start_result_identity_verification(record)
                return
        if result.is_dir():
            self.result_title_label.setText("Your photos are ready for training")
            self.result_subtitle_label.setText(
                "Review the prepared photos and captions here, then check the "
                "remaining tools and continue without preparing everything again."
            )
            self.result_image.setAccessibleName("Prepared photo dataset status")
            self.result_open_comfy_button.setVisible(False)
            self.result_open_desktop_button.setVisible(False)
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

    def compare_training_checkpoints(self) -> None:
        record = self._displayed_result_record
        if record is None or record.mode != "full":
            return
        settings = self._collect_settings()
        self.compare_checkpoints_button.setEnabled(False)

        def failed(message: str) -> None:
            self.compare_checkpoints_button.setEnabled(True)
            self._show_error("Checkpoint comparison could not open", message)

        def show_choices(value: object) -> None:
            self.compare_checkpoints_button.setEnabled(True)
            if (
                self._displayed_result_record != record
                or self._collect_settings() != settings
            ):
                return
            choices = cast(tuple[TrainingCheckpoint, ...], value)
            if not choices:
                failed(
                    "No readable checkpoints remain in this project's training folder. The completed validation result is still available."
                )
                return
            dialog = CheckpointReviewDialog(choices, self)
            try:
                if dialog.exec() == CheckpointReviewDialog.DialogCode.Accepted:
                    self._validate_checkpoint_choice(
                        record, settings, dialog.selected_checkpoint
                    )
            finally:
                dialog.deleteLater()

        self._run_worker(
            lambda: discover_training_checkpoints(
                self.service.project_path(settings.checkpoints_dir),
                record.training_project_name or record.project_name,
            ),
            show_choices,
            failed,
            operation="checkpoints.discover",
        )

    def _validate_checkpoint_choice(
        self, record: RunResult, settings: AppSettings, choice: TrainingCheckpoint
    ) -> None:
        issue = self._result_comfy_context_issue(record)
        if issue is not None:
            self._show_error("Reconnect before validating", issue)
            return
        self.compare_checkpoints_button.setEnabled(False)
        self.result_subtitle_label.setText(
            f"Checking {choice.label.lower()} before validation…"
        )

        def failed(message: str) -> None:
            self.compare_checkpoints_button.setEnabled(True)
            self.result_subtitle_label.setText(
                "Checkpoint validation could not start. Your existing result is still available."
            )
            self._show_error("Checkpoint validation could not start", message)

        def launch(value: object) -> None:
            self.compare_checkpoints_button.setEnabled(True)
            if (
                self._displayed_result_record != record
                or self._collect_settings() != settings
            ):
                return
            issue = self._result_comfy_context_issue(record)
            if issue is not None:
                failed(issue)
                return
            if (
                self.pipeline_process is not None
                and self.pipeline_process.state() != QProcess.ProcessState.NotRunning
            ):
                failed(
                    "Another run is active. Wait for it to finish before validating this checkpoint."
                )
                return
            draft, command = cast(tuple[ProjectDraft, PipelineCommand], value)
            self.settings = replace(settings, last_project=draft)
            self.draft = draft
            self._populate_last_project()
            self._validated_draft = draft
            self._validated_settings = self._collect_settings()
            try:
                save_settings(self.settings, self.settings_path)
            except OSError as exc:
                failed(str(exc))
                return
            self._installed_lora_name = None
            self._launch_pipeline_command(
                command, f"Validating training step {choice.step:,}"
            )

        self._run_worker(
            lambda: self.service.build_checkpoint_validation_command(
                settings, record, choice
            ),
            launch,
            failed,
            operation="checkpoints.validate",
        )

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
        try:
            # Own the displayed dataset before reading any of its pairs. Keep
            # that ownership through edits, save prompts, and the final close,
            # so another Studio or CLI run cannot train on changing captions.
            with pipeline_run_guard(
                project=result.name,
                workspace=result.parent,
                mode="caption-review",
                use_accelerator=False,
                resource_paths=(result,),
            ):
                PreparedDatasetDialog(result, required_trigger, self).exec()
        except RunLeaseBusyError as exc:
            self._show_error("Prepared photos are in use", str(exc))
        except (RunLeaseError, OSError) as exc:
            self._show_error("Prepared photos could not be opened", str(exc))
        finally:
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

    def _ready_workflow_for_handoff(self, record: RunResult | None) -> Path | None:
        workflow = record.workflow_path if record is not None else None
        if (
            record is None
            or workflow is None
            or not workflow.is_file()
            or not record.workflow_fingerprint
        ):
            self._show_error(
                "Ready workflow not found",
                "The ComfyUI workflow is no longer available. Retry validation to "
                "recreate it without retraining.",
            )
            return None
        try:
            fingerprint = artifact_fingerprint(workflow)
        except (FileNotFoundError, OSError, ValueError):
            fingerprint = ""
        if fingerprint != record.workflow_fingerprint:
            self._show_error(
                "Ready workflow changed",
                "This workflow no longer matches the verified validation result. "
                "Retry validation to recreate the safe version.",
            )
            return None
        return workflow

    def _ready_workflow_snapshot(
        self,
        record: RunResult,
    ) -> tuple[str, bytes] | None:
        """Read and verify the exact immutable bytes that will be uploaded."""

        workflow = record.workflow_path
        if (
            workflow is None
            or not workflow.is_file()
            or workflow.is_symlink()
            or not record.workflow_fingerprint
        ):
            self._show_error(
                "Ready workflow not found",
                "The ComfyUI workflow is no longer available. Retry validation to "
                "recreate it without retraining.",
            )
            return None
        try:
            data = workflow.read_bytes()
        except OSError as exc:
            self._show_error(
                "Ready workflow could not be read",
                f"{exc}\n\nRetry validation to recreate the safe workflow.",
            )
            return None
        if _file_bytes_fingerprint(data) != record.workflow_fingerprint:
            self._show_error(
                "Ready workflow changed",
                "This workflow no longer matches the verified validation result. "
                "Retry validation to recreate the safe version.",
            )
            return None
        return workflow.name, data

    def _result_installation_name(self, record: RunResult) -> str:
        if record.comfy_installation_name:
            return record.comfy_installation_name
        if self.discovery is not None:
            match = next(
                (
                    installation
                    for installation in self.discovery.installations
                    if installation.id == record.comfy_installation_id
                ),
                None,
            )
            if match is not None:
                return match.name
        return "the original local instance"

    def _workflow_publish_is_current(
        self,
        request: int,
        setup_request: int,
        record: RunResult,
        host: str,
    ) -> bool:
        """Return whether an async handoff still targets its exact runtime."""

        status = self.comfy_status
        selected = (
            self.discovery.selected_installation if self.discovery is not None else None
        )
        return (
            request == self._workflow_publish_request
            and setup_request == self._setup_request
            and self._displayed_result_record == record
            and self._collect_settings().comfy_installation_id
            == record.comfy_installation_id
            and selected is not None
            and selected.id == record.comfy_installation_id
            and status is not None
            and status.reachable
            and status.host == host
        )

    def _open_result_connect_repair(self, record: RunResult, issue: str) -> None:
        """Take a blocked Results handoff directly to a fresh Connect check."""

        self.studio_log.warning(
            "workflow.publish.blocked",
            f"project={record.project_name}; detail={issue}",
        )
        self.show_page(NAV_SETUP)
        self.comfy_summary_label.setText(issue)
        self._single_shot(0, self.refresh_setup)

    def add_ready_workflow_to_comfyui(self) -> None:
        """Publish the verified UI workflow through ComfyUI's user-data API."""

        record = self._displayed_result_record
        published = self._published_workflow
        if published is not None and published[0] == record:
            self.open_comfy_ui()
            return
        setup_issue = self._result_comfy_setup_issue(record)
        if record is not None and setup_issue is not None:
            self._open_result_connect_repair(record, setup_issue)
            return
        identity_failure = self._result_identity_failure
        if (
            record is not None
            and identity_failure is not None
            and identity_failure[0] == record
        ):
            self._start_result_identity_verification(record)
            return
        context_issue = self._result_comfy_context_issue(record)
        if context_issue is not None:
            if (
                record is not None
                and record.installed_lora_path is not None
                and record.checkpoint_path is not None
            ):
                self._start_result_identity_verification(record)
            else:
                self._show_error(
                    "Result files need attention",
                    f"{context_issue}\n\nRetry validation to recreate this handoff.",
                )
            return
        if record is None:
            return

        if self._workflow_publish_retry_requires_refresh:
            self._workflow_publish_retry_requires_refresh = False
            self._pending_workflow_publish = record
            self.publish_workflow_button.setEnabled(False)
            self.publish_workflow_button.setText("Refreshing connection…")
            self.workflow_handoff_status_label.setText(
                "Checking the current Comfy Desktop runtime and port before retrying…"
            )
            self.refresh_setup()
            return

        status = self.comfy_status
        host = status.host if status is not None and status.reachable else None
        if host is None:
            self._pending_workflow_publish = record
            self.publish_workflow_button.setEnabled(False)
            self.publish_workflow_button.setText("Waiting for ComfyUI…")
            self.workflow_handoff_status_label.setText(
                "Comfy Desktop is opening. Launch "
                f"{self._result_installation_name(record)}; MIVIM will finish "
                "adding the workflow automatically."
            )
            self.open_comfy_desktop()
            if not self._awaiting_comfy_launch:
                self._pending_workflow_publish = None
                self.publish_workflow_button.setText("Retry add to ComfyUI")
                self.publish_workflow_button.setEnabled(True)
                self.workflow_handoff_status_label.setText(
                    "LoRA ready; workflow not added yet. Open the original "
                    "ComfyUI instance, then retry."
                )
            return

        workflow_snapshot = self._ready_workflow_snapshot(record)
        if workflow_snapshot is None:
            return
        workflow_name, workflow_data = workflow_snapshot
        self._pending_workflow_publish = None
        self._workflow_publish_request += 1
        request = self._workflow_publish_request
        setup_request = self._setup_request
        self._workflow_publish_busy = True
        self.publish_workflow_button.setEnabled(False)
        self.publish_workflow_button.setText("Checking profiles…")
        self.workflow_handoff_status_label.setText(
            "Checking which ComfyUI profile should receive the workflow…"
        )

        def load_profiles() -> tuple[ComfyUserProfile, ...]:
            return ComfyClient(host=host, timeout=10.0).list_user_profiles()

        self._run_worker(
            load_profiles,
            lambda result: self._workflow_profiles_loaded(
                request,
                setup_request,
                record,
                workflow_name,
                workflow_data,
                host,
                result,
            ),
            lambda message: self._workflow_publish_failed(
                request,
                setup_request,
                record,
                message,
            ),
            operation="workflow.profiles",
        )

    def _workflow_profiles_loaded(
        self,
        request: int,
        setup_request: int,
        record: RunResult,
        workflow_name: str,
        workflow_data: bytes,
        host: str,
        result: object,
    ) -> None:
        if not self._workflow_publish_is_current(
            request,
            setup_request,
            record,
            host,
        ):
            return
        context_issue = self._result_comfy_context_issue(record)
        if context_issue is not None:
            self._workflow_publish_failed(
                request,
                setup_request,
                record,
                context_issue,
            )
            return
        if not isinstance(result, tuple) or not all(
            isinstance(item, ComfyUserProfile) for item in result
        ):
            self._workflow_publish_failed(
                request,
                setup_request,
                record,
                "ComfyUI returned an unexpected user-profile list.",
            )
            return
        profiles = cast(tuple[ComfyUserProfile, ...], result)
        if not profiles:
            self._workflow_publish_failed(
                request,
                setup_request,
                record,
                "ComfyUI has no regular profile that can receive this workflow.",
            )
            return

        if len(profiles) == 1:
            profile = profiles[0]
        else:
            saved_user = (
                self.settings.comfy_user_id
                if self.settings.comfy_user_installation_id
                == record.comfy_installation_id
                else record.workflow_user_id
            )
            selected_index = next(
                (
                    index
                    for index, candidate in enumerate(profiles)
                    if candidate.id == saved_user
                ),
                0,
            )
            duplicate_names = {
                candidate.name.casefold()
                for candidate in profiles
                if sum(
                    other.name.casefold() == candidate.name.casefold()
                    for other in profiles
                )
                > 1
            }
            profile_choices = [
                (
                    f"{candidate.name} — {candidate.id}"
                    if candidate.name.casefold() in duplicate_names
                    else candidate.name
                )
                for candidate in profiles
            ]
            selected_name, accepted = QInputDialog.getItem(
                self,
                "Choose a ComfyUI profile",
                "Which ComfyUI profile should receive this workflow?",
                profile_choices,
                selected_index,
                False,
            )
            if not accepted:
                self._workflow_publish_busy = False
                self.publish_workflow_button.setText("Add to ComfyUI")
                self.publish_workflow_button.setEnabled(True)
                self.workflow_handoff_status_label.setText(
                    "Nothing was changed. Choose Add to ComfyUI when you are ready."
                )
                self.studio_log.info(
                    "workflow.publish.cancelled",
                    f"project={record.project_name}",
                )
                return
            try:
                profile = profiles[profile_choices.index(selected_name)]
            except ValueError:
                self._workflow_publish_failed(
                    request,
                    setup_request,
                    record,
                    "The selected ComfyUI profile is no longer available.",
                )
                return

        if not self._workflow_publish_is_current(
            request,
            setup_request,
            record,
            host,
        ):
            return

        self.settings = replace(
            self._collect_settings(),
            comfy_user_installation_id=record.comfy_installation_id,
            comfy_user_id=profile.id,
        )
        self.save_current_settings(silent=True)
        self.publish_workflow_button.setText("Adding…")
        self.workflow_handoff_status_label.setText(
            f"Adding the verified workflow to {profile.name}…"
        )

        def publish() -> WorkflowPublication:
            if (
                request != self._workflow_publish_request
                or setup_request != self._setup_request
            ):
                raise RuntimeError(
                    "The ComfyUI connection changed before the upload started."
                )
            return ComfyClient(host=host, timeout=15.0).publish_workflow_bytes(
                workflow_name,
                workflow_data,
                user_id=profile.id,
                expected_fingerprint=record.workflow_fingerprint,
            )

        self._run_worker(
            publish,
            lambda publication: self._workflow_published(
                request,
                setup_request,
                record,
                host,
                publication,
            ),
            lambda message: self._workflow_publish_failed(
                request,
                setup_request,
                record,
                message,
            ),
            operation="workflow.publish",
        )

    def _workflow_published(
        self,
        request: int,
        setup_request: int,
        record: RunResult,
        host: str,
        result: object,
    ) -> None:
        if not self._workflow_publish_is_current(
            request,
            setup_request,
            record,
            host,
        ):
            return
        if not isinstance(result, WorkflowPublication):
            self._workflow_publish_failed(
                request,
                setup_request,
                record,
                "ComfyUI returned an unexpected publication result.",
            )
            return
        if result.fingerprint != record.workflow_fingerprint:
            self._workflow_publish_failed(
                request,
                setup_request,
                record,
                "The published workflow did not match the verified local JSON.",
            )
            return

        published_record = replace(
            record,
            comfy_installation_name=self._result_installation_name(record),
            published_workflow_path=result.relative_path,
            published_workflow_fingerprint=result.fingerprint,
            workflow_user_id=result.profile.id,
            workflow_user_name=result.profile.name,
        )
        verified = self._verified_result_identity
        if verified is not None and verified[0] == record:
            self._verified_result_identity = (
                published_record,
                verified[1],
                verified[2],
            )
        self._displayed_result_record = published_record
        self._published_workflow = (published_record, result)
        self._workflow_publish_busy = False
        self._workflow_publish_retry_requires_refresh = False
        self._pending_workflow_publish = None
        self._persist_run_result(published_record)
        self.publish_workflow_button.setText("Open ComfyUI")
        self.publish_workflow_button.setEnabled(True)
        workflow_name = (
            record.workflow_path.stem
            if record.workflow_path is not None
            else record.project_name
        )
        collision_note = (
            " A protected alternate name was used because the fingerprinted "
            "name had been changed in ComfyUI."
            if result.renamed
            else ""
        )
        if result.created:
            receipt = (
                f"Added {workflow_name} to Workflows → MIVIM for profile "
                f"{result.profile.name}."
            )
        else:
            receipt = (
                f"{workflow_name} is already available in Workflows → MIVIM for "
                "profile "
                f"{result.profile.name}; the verified copy matched exactly."
            )
        self.workflow_handoff_status_label.setText(
            f"{receipt}{collision_note} If ComfyUI was already open, click Refresh "
            "in Workflows."
        )
        self.studio_log.info(
            "workflow.publish.finished",
            (
                f"project={record.project_name}; host={host}; "
                f"path={result.relative_path}; created={result.created}"
            ),
        )

    def _workflow_publish_failed(
        self,
        request: int,
        setup_request: int,
        record: RunResult,
        message: str,
    ) -> None:
        if (
            request != self._workflow_publish_request
            or setup_request != self._setup_request
            or self._displayed_result_record != record
        ):
            return
        self._workflow_publish_busy = False
        self._workflow_publish_retry_requires_refresh = True
        self._pending_workflow_publish = None
        self.publish_workflow_button.setText("Retry add to ComfyUI")
        self.publish_workflow_button.setEnabled(
            self._result_comfy_context_issue(record) is None
        )
        self.workflow_handoff_status_label.setText(
            "LoRA ready; workflow not added yet. Retry, or use Show JSON file."
        )
        self.studio_log.warning(
            "workflow.publish.failed",
            f"project={record.project_name}; detail={message}",
        )
        self._show_error(
            "LoRA is ready; workflow was not added",
            f"{message}\n\nNothing was overwritten. Retry, or use Show JSON file "
            "and Workflow → Open in ComfyUI.",
        )

    def copy_result_setup(self) -> None:
        record = self._displayed_result_record
        if record is None:
            return
        QApplication.clipboard().setText(
            "\n".join(
                (
                    f"LoRA: {record.lora_name}",
                    f"Identity phrase: {record.trigger}",
                    f"Base checkpoint: {record.checkpoint}",
                    f"Starting strength: {record.starting_strength:g}",
                    (
                        f"Ready workflow: {record.workflow_path}"
                        if record.workflow_path is not None
                        else "Ready workflow: not recorded"
                    ),
                )
            )
        )
        self.copy_result_setup_button.setText("Copied")
        self._single_shot(
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

    def open_ready_workflow(self) -> None:
        record = self._displayed_result_record
        workflow = self._ready_workflow_for_handoff(record)
        if workflow is not None:
            self._show_local_file(workflow)

    def _show_local_file(self, path: Path) -> None:
        if not path.is_file():
            self._show_error("File not found", f"The file no longer exists:\n{path}")
            return
        resolved = path.resolve()
        if sys.platform == "win32":
            try:
                subprocess.Popen(
                    ["explorer.exe", f"/select,{resolved}"],
                    close_fds=True,
                )
            except OSError as exc:
                self._show_error("Could not show the file", str(exc))
            return
        self._open_local_folder(resolved.parent)

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
        self._result_identity_request += 1
        self._recovery_check_request += 1
        self._workflow_publish_request += 1
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
        self._run_settings_snapshot = None
        self._run_draft_snapshot = None
        self._latest_result = None
        self._displayed_result_record = None
        self._verified_result_identity = None
        self._result_identity_failure = None
        self._pending_workflow_publish = None
        self._published_workflow = None
        self._workflow_publish_busy = False
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
            # Capture a just-emitted ownership marker before deciding whether
            # closing can safely stop this run.
            self._read_pipeline_output()
            self._flush_pipeline_line_buffer()
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
            if self._owned_comfy_prompt is not None:
                prompt_id, client_id, host = self._owned_comfy_prompt
                try:
                    result = ComfyClient(
                        host=host,
                        timeout=2.5,
                    ).cancel_owned_prompt(prompt_id, client_id)
                    self.studio_log.info(
                        "comfy.prompt.cancel.on_close",
                        f"status={result.status}; prompt_id={prompt_id}",
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    self.studio_log.warning(
                        "comfy.prompt.cancel.on_close_failed",
                        str(exc),
                    )
                    self._warn_close_for_running_comfy_prompt(
                        "MIVIM could not verify that this ComfyUI job stopped."
                    )
                    event.ignore()
                    return
                if not result.requested and result.status != "not_found":
                    self.studio_log.warning(
                        "comfy.prompt.cancel.on_close_unconfirmed",
                        f"status={result.status}; prompt_id={prompt_id}",
                    )
                    self._warn_close_for_running_comfy_prompt(result.message)
                    event.ignore()
                    return
            self.run_cancelled = True
            self._stop_process_tree(force=True)
        self._closing = True
        self._training_elapsed_timer.stop()
        self.studio_log.info("session.closed", "MIVIM LoRA Studio closed.")
        self.studio_log.close()
        self._desktop_owner_lifetime.close()
        event.accept()

    def _warn_close_for_running_comfy_prompt(self, detail: str) -> None:
        message = (
            f"{detail}\n\nMIVIM will stay open and leave the worker running. "
            "Open ComfyUI, stop this job from its queue, then close MIVIM again."
        )
        self._comfy_cancel_detail = message
        self.run_stage_label.setText("ComfyUI job still needs attention")
        self.run_detail_label.setText(message)
        QMessageBox.warning(
            self,
            "Stop the ComfyUI job before closing",
            message,
        )


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
        "--screenshot-page",
        choices=("connect", "project", "run", "result"),
        default="connect",
        help="Page to capture with --screenshot (default: connect).",
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
    app.setApplicationVersion(__version__)
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
        page = {
            "connect": NAV_SETUP,
            "project": NAV_PROJECT,
            "run": NAV_RUN,
            "result": NAV_RESULTS,
        }[args.screenshot_page]

        def capture() -> None:
            if not window.nav_buttons[page].isEnabled():
                print(
                    f"Screenshot page is unavailable: {args.screenshot_page}",
                    file=sys.stderr,
                )
                app.exit(3)
                return
            window.show_page(page)
            app.processEvents()
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(destination)):
                print(
                    f"Could not save desktop screenshot: {destination}",
                    file=sys.stderr,
                )
                app.exit(2)
                return
            print(f"Desktop screenshot: {destination}")
            if args.screenshot_page == "result":
                action = window.publish_workflow_button.text()
                ready_actions = {"Add to ComfyUI", "Check in ComfyUI"}
                handoff_state = (
                    "ready"
                    if window.publish_workflow_button.isVisible()
                    and window.publish_workflow_button.isEnabled()
                    and action in ready_actions
                    else "blocked"
                )
                print(f"Desktop result handoff: {handoff_state}; action={action!r}")
            app.quit()

        QTimer.singleShot(max(200, args.screenshot_delay), capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
