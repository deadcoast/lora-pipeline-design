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
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from PySide6.QtCore import (
    QObject,
    QProcess,
    QProcessEnvironment,
    QRunnable,
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
from .app_models import AppSettings, ProjectDraft, TrainingPreset
from .app_service import (
    ApplicationService,
    PipelineCommand,
    ReadinessItem,
    ReadinessReport,
    inspect_source,
    stage_from_log,
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
from .desktop_theme import APP_STYLE

NAV_SETUP, NAV_PROJECT, NAV_RUN, NAV_RESULTS = range(4)


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
    failed = Signal(str)


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
            try:
                self.signals.failed.emit(str(exc))
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
        self.service = service or ApplicationService()
        self.discovery: ComfyDesktopDiscovery | None = None
        self.comfy_status: ComfyApiStatus | None = None
        self.draft: ProjectDraft | None = None
        self.readiness_report: ReadinessReport | None = None
        self.source_folder: Path | None = None
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
        self._validated_settings: AppSettings | None = None
        self._validated_draft: ProjectDraft | None = None
        self._setup_request = 0
        self._source_request = 0
        self._readiness_request = 0

        self.setWindowTitle("MIVIM LoRA Studio")
        self.setWindowIcon(_app_icon())
        self.setMinimumSize(1040, 690)
        self.resize(1180, 780)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        self._populate_settings()
        self._populate_last_project()
        self._restore_last_result()
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
        page.addWidget(hero)

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
        model_layout.addWidget(_hint("Local training model file"))
        self.base_model_edit = QLineEdit()
        self.base_model_edit.setObjectName("trainingBaseModelEdit")
        self.base_model_edit.setPlaceholderText(
            "Local checkpoint path is filled from Comfy Desktop"
        )
        self.base_model_edit.setAccessibleName("Local SDXL training checkpoint file")
        base_model_row = QHBoxLayout()
        base_model_row.addWidget(self.base_model_edit, 1)
        base_model_browse = _button("Browse")
        base_model_browse.setObjectName("browseBaseModelButton")
        base_model_browse.setAccessibleName("Browse for local SDXL checkpoint file")
        base_model_browse.clicked.connect(self.choose_base_model)
        base_model_row.addWidget(base_model_browse)
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
        toolkit_browse = _button("Browse")
        toolkit_browse.setObjectName("browseToolkitButton")
        toolkit_browse.setAccessibleName("Browse for ai-toolkit folder")
        toolkit_browse.clicked.connect(self.choose_toolkit_dir)
        toolkit_row.addWidget(toolkit_browse)
        trainer_layout.addLayout(toolkit_row)
        self.toolkit_python_edit = QLineEdit()
        self.toolkit_python_edit.setObjectName("toolkitPythonEdit")
        self.toolkit_python_edit.setPlaceholderText(
            "The trainer's venv Python is detected automatically"
        )
        self.toolkit_python_edit.setAccessibleName("ai-toolkit Python executable")
        python_row = QHBoxLayout()
        python_row.addWidget(self.toolkit_python_edit, 1)
        python_browse = _button("Browse")
        python_browse.setObjectName("browseToolkitPythonButton")
        python_browse.setAccessibleName("Browse for ai-toolkit Python executable")
        python_browse.clicked.connect(self.choose_toolkit_python)
        python_row.addWidget(python_browse)
        trainer_layout.addLayout(python_row)
        form_grid.addWidget(trainer_card, 0, 1)

        storage_card = _card()
        storage_layout = QVBoxLayout(storage_card)
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
        form_grid.addWidget(storage_card, 1, 0)

        advanced_card = _card()
        advanced_layout = QVBoxLayout(advanced_card)
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
        form_grid.addWidget(advanced_card, 1, 1)

        form_grid.setColumnStretch(0, 1)
        form_grid.setColumnStretch(1, 1)
        page.addLayout(form_grid)

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
        page.addWidget(self.photo_summary)

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
            "Balanced uses 1024 px photos and three comparison strengths."
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
            "Rebuild the prepared photos if this project already exists"
        )
        self.overwrite_checkbox.setObjectName("overwriteCheck")
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
        self.run_stack.addWidget(readiness_widget)
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

        actions = QHBoxLayout()
        self.open_result_button = _button("Open result folder")
        self.open_result_button.setObjectName("openResultButton")
        self.open_result_button.clicked.connect(self.open_result_folder)
        actions.addWidget(self.open_result_button)
        self.result_open_comfy_button = _button("Open ComfyUI", "primaryButton")
        self.result_open_comfy_button.setObjectName("resultOpenComfyButton")
        self.result_open_comfy_button.clicked.connect(self.open_comfy_ui)
        self.result_open_comfy_button.setEnabled(False)
        actions.addWidget(self.result_open_comfy_button)
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
    ) -> None:
        worker = FunctionWorker(function)
        self._workers.append(worker)

        def done(result: object) -> None:
            if worker in self._workers:
                self._workers.remove(worker)
            success(result)

        def failed(message: str) -> None:
            if worker in self._workers:
                self._workers.remove(worker)
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
        self.comfy_badge.setText("CHECKING")
        self.comfy_badge.setObjectName("statusWarning")
        self.comfy_badge.style().unpolish(self.comfy_badge)
        self.comfy_badge.style().polish(self.comfy_badge)
        self.comfy_summary_label.setText(
            "Checking the installed Desktop instances and local ComfyUI API…"
        )

        original = self._collect_settings()

        def work() -> tuple[AppSettings, ComfyDesktopDiscovery, ComfyApiStatus]:
            discovery = self.service.discover_comfy()
            applied, status = self.service.setup_snapshot(original, discovery)
            return applied, discovery, status

        self._run_worker(
            work,
            lambda result: self._setup_refreshed(request, original, result),
            lambda message: self._setup_failed_for(request, message),
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
        self.open_desktop_button.setEnabled(discovery.launcher.supported)
        self.open_comfy_button.setEnabled(status.reachable)
        self.result_open_comfy_button.setEnabled(status.reachable)
        local_installations = tuple(
            installation
            for installation in discovery.installations
            if installation.is_local
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
        self.save_current_settings(silent=True)

    def _setup_failed_for(self, request: int, message: str) -> None:
        if request == self._setup_request:
            self._setup_failed(message)

    def _setup_failed(self, message: str) -> None:
        self.retry_comfy_button.setEnabled(True)
        self.comfy_badge.setText("CHECK FAILED")
        self.comfy_badge.setObjectName("statusError")
        self.comfy_badge.style().unpolish(self.comfy_badge)
        self.comfy_badge.style().polish(self.comfy_badge)
        self.comfy_summary_label.setText(
            f"The automatic check could not finish: {message}"
        )

    def _populate_settings(self) -> None:
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
        if names:
            self.model_hint.setText(
                f"{len(names)} model(s) reported by the running ComfyUI instance."
            )
        else:
            self.model_hint.setText(
                "No checkpoint is currently visible. Add an SDXL model in ComfyUI."
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
        name = self.settings.last_project.project_name.strip()
        if not name:
            return
        try:
            config.validate_project_name(name)
        except ValueError:
            return
        candidates = (
            self.service.project_path(self.settings.grids_dir) / f"{name}_grid.png",
            self.service.project_path(self.settings.datasets_dir) / name,
        )
        result = next((path for path in candidates if path.exists()), None)
        if result is not None:
            self._show_result(result)

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
            comfy_lora_dir=optional_path(self.lora_dir_edit.text()),
            comfy_checkpoint=checkpoint,
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
        QTimer.singleShot(5000, self.refresh_setup)

    def open_comfy_ui(self) -> None:
        host = self.host_edit.text().strip() or self.settings.comfy_host
        cleaned = (
            host.strip().removeprefix("http://").removeprefix("https://").rstrip("/")
        )
        if not cleaned or not QDesktopServices.openUrl(QUrl(f"http://{cleaned}")):
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
        for editor in (
            self.host_edit,
            self.lora_dir_edit,
            self.toolkit_dir_edit,
            self.toolkit_python_edit,
            self.base_model_edit,
            self.workspace_dir_edit,
            self.project_name_edit,
            self.trigger_edit,
        ):
            editor.textEdited.connect(self._invalidate_readiness)
        self.checkpoint_combo.currentTextChanged.connect(self._checkpoint_changed)
        self.vision_checkbox.toggled.connect(self._invalidate_readiness)
        self.overwrite_checkbox.toggled.connect(self._invalidate_readiness)

    def _checkpoint_changed(self, checkpoint: str) -> None:
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
        self._invalidate_readiness()
        if not self.project_name_edit.text().strip():
            safe = "".join(
                char if char.isalnum() or char in "._-" else "_"
                for char in folder.name.strip()
            ).strip("._-")
            if safe:
                self.project_name_edit.setText(f"{safe}_v1")
        self.photo_summary.setText("Inspecting the selected photos…")

        preset = TrainingPreset(self.preset_combo.currentData())
        self._source_request += 1
        request = self._source_request
        self._run_worker(
            lambda: inspect_source(folder, target_size=preset.image_size),
            lambda result: self._source_inspected_for(request, folder, preset, result),
            lambda message: self._source_inspection_failed(
                request, folder, preset, message
            ),
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
            self.photo_summary.setText(f"The photos could not be inspected: {message}")

    def _source_inspected(self, result: object) -> None:
        summary = result
        detail = f"{summary.usable_count} usable photo(s)"  # type: ignore[attr-defined]
        if summary.unreadable_count:  # type: ignore[attr-defined]
            detail += f" • {summary.unreadable_count} unreadable"  # type: ignore[attr-defined]
        if summary.small_count:  # type: ignore[attr-defined]
            detail += f" • {summary.small_count} will be enlarged"  # type: ignore[attr-defined]
        if summary.duplicate_count:  # type: ignore[attr-defined]
            detail += f" • {summary.duplicate_count} exact duplicate(s)"  # type: ignore[attr-defined]
        self.photo_summary.setText(detail)

    def _preset_changed(self) -> None:
        preset = TrainingPreset(self.preset_combo.currentData())
        descriptions = {
            TrainingPreset.FAST: (
                "Fast draft uses 900 training steps, rank 8, 768 px preparation, "
                "and one validation strength."
            ),
            TrainingPreset.BALANCED: (
                "Balanced uses 2,000 steps, rank 16, 1024 px preparation, and "
                "three comparison strengths."
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
        self._run_worker(
            lambda: self.service.readiness(settings, draft),
            lambda result: self._readiness_finished_for(
                request, settings, draft, result
            ),
            lambda message: self._readiness_failed_for(request, message),
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

    def _readiness_failed_for(self, request: int, message: str) -> None:
        if request == self._readiness_request:
            self._readiness_failed(message)

    def _readiness_failed(self, message: str) -> None:
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
        if item.code in {"project", "collision"}:
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
            self.photo_drop._browse()
        elif item.code == "comfy":
            self.show_page(NAV_SETUP)
            self.open_comfy_desktop()
        elif item.code == "checkpoint":
            self.show_page(NAV_SETUP)
            self.checkpoint_combo.setFocus()
            self.checkpoint_combo.showPopup()
        elif item.code == "base_model":
            self.show_page(NAV_SETUP)
            self.choose_base_model()
        elif item.code == "lora_dir":
            self.show_page(NAV_SETUP)
            self.choose_lora_dir()
        elif item.code == "trainer":
            self.show_page(NAV_SETUP)
            if "Python" in item.action:
                self.choose_toolkit_python()
            else:
                self.choose_toolkit_dir()
        elif item.code in {"storage", "training_storage"}:
            self.show_page(NAV_SETUP)
            self.choose_workspace_dir()
        elif item.code == "internal":
            self.run_readiness_checks()

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
        self.run_cancelled = False
        self._run_terminal_handled = False
        self._result_before = _artifact_snapshot(command.expected_result)
        self.run_stack.setCurrentIndex(1)
        self.run_log.clear()
        self.run_stage_label.setText(stage_label)
        self.run_detail_label.setText(
            "This uses the real project pipeline. Technical output is shown below "
            "if you need support."
        )
        self.run_progress.setValue(3)
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
        self.run_log.moveCursor(QTextCursor.MoveOperation.End)
        self.run_log.insertPlainText(data)
        self.run_log.ensureCursorVisible()
        for line in data.splitlines():
            progress = stage_from_log(line)
            if progress is not None:
                value, label = progress
                self.run_progress.setValue(max(self.run_progress.value(), value))
                self.run_stage_label.setText(label)
            marker = "[pipeline] Copied LoRA -> "
            if marker in line:
                installed = Path(line.partition(marker)[2].strip()).name
                if installed.casefold().endswith(".safetensors"):
                    self._installed_lora_name = installed

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
        self.run_log.appendPlainText(
            f"\n[studio] Could not run the pipeline: {process.errorString()}"
        )
        if error == QProcess.ProcessError.FailedToStart:
            self._run_terminal_handled = True
            self.cancel_button.setEnabled(False)
            self.run_stage_label.setText("The pipeline could not start")
            self.run_detail_label.setText(
                "No work was started. Edit the setup or check again, then retry."
            )
            self.run_progress.setValue(0)
            self._restore_navigation()
            self._show_recovery_actions(True)

    def _pipeline_finished(
        self,
        exit_code: int,
        _exit_status: QProcess.ExitStatus,
    ) -> None:
        if self._run_terminal_handled:
            return
        self._run_terminal_handled = True
        self._read_pipeline_output()
        self.cancel_button.setEnabled(False)
        self._restore_navigation()
        command = self.pipeline_command
        if command is None:
            return
        if self.run_cancelled:
            self.run_stage_label.setText("Run stopped")
            self.run_detail_label.setText(
                "Completed files were kept. Return to the review when you are ready."
            )
            self.run_progress.setValue(0)
            self._show_recovery_actions(True)
            return
        if exit_code != 0:
            self.run_stage_label.setText("This run needs attention")
            self.run_detail_label.setText(
                "Completed photos or checkpoints were kept. The technical details "
                "below show the stage that failed; fix the setup item and retry."
            )
            self.run_progress.setValue(0)
            self._show_recovery_actions(True)
            return
        result_after = _artifact_snapshot(command.expected_result)
        if result_after is None or result_after == self._result_before:
            self.run_stage_label.setText("Result could not be verified")
            self.run_detail_label.setText(
                "The pipeline exited successfully, but it did not create or update "
                f"the expected result at {command.expected_result}."
            )
            self._show_recovery_actions(True)
            return
        self.run_progress.setValue(100)
        self.run_stage_label.setText("Complete")
        self.run_detail_label.setText(
            "The result was verified on disk and is ready to review."
        )
        self._show_result(command.expected_result)
        QTimer.singleShot(450, lambda: self.show_page(NAV_RESULTS))

    def _restore_navigation(self) -> None:
        self.nav_buttons[NAV_SETUP].setEnabled(True)
        self.nav_buttons[NAV_PROJECT].setEnabled(True)
        self.nav_buttons[NAV_RUN].setEnabled(True)
        self.nav_buttons[NAV_RESULTS].setEnabled(self._latest_result is not None)

    def _show_recovery_actions(self, visible: bool) -> None:
        self.edit_after_run_button.setVisible(visible)
        self.open_run_folder_button.setVisible(visible)
        self.retry_after_run_button.setVisible(visible)
        self.retry_validation_button.setVisible(
            visible and self._installed_lora_name is not None
        )

    def retry_pipeline(self) -> None:
        self._show_recovery_actions(False)
        self.run_stack.setCurrentIndex(0)
        self.run_readiness_checks()

    def retry_validation(self) -> None:
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
    def _show_result(self, result: Path) -> None:
        self._latest_result = result
        self.nav_buttons[NAV_RESULTS].setEnabled(True)
        self.result_path_label.setText(str(result))
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
                self.result_open_comfy_button.setVisible(True)
                self.result_open_comfy_button.setEnabled(
                    self.comfy_status is not None and self.comfy_status.reachable
                )
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
                "Preparation is complete. Connect any missing training tools when "
                "you are ready to create the LoRA."
            )
            self.result_image.setAccessibleName("Prepared photo dataset status")
            self.result_open_comfy_button.setVisible(False)
            self.result_image.setPixmap(QPixmap())
            self.result_image.setText(
                "✓\n\nYour photos are prepared.\n\n"
                "Connect the remaining training tools, then return to Review & run "
                "for the full LoRA."
            )
            self.result_image.setStyleSheet(
                "font-size: 19px; color: #9ee7d7; padding: 30px;"
            )
        else:
            self.result_image.setText("The result is ready.")

    def open_result_folder(self) -> None:
        result = self._latest_result
        if result is None:
            return
        folder = result if result.is_dir() else result.parent
        self._open_local_folder(folder)

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
