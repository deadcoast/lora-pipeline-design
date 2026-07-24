"""Concrete, filename-level review for source-photo quality checks."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .app_service import SourceIssue, SourceSummary

_ISSUE_NAMES = {
    "unreadable": "Cannot open",
    "small": "Low resolution",
    "duplicate": "Exact duplicate",
    "near_duplicate": "Very similar",
    "blurry": "May be blurry",
}


class PhotoIssueDialog(QDialog):
    """Show exactly which source files need attention, without changing them."""

    def __init__(self, summary: SourceSummary, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.summary = summary
        self.setWindowTitle("Review photo checks")
        self.setMinimumSize(980, 610)

        root = QVBoxLayout(self)
        heading = QLabel("Photo quality review")
        heading.setObjectName("pageTitle")
        root.addWidget(heading)
        text = QLabel(
            "These checks are local and never change your originals. Exact duplicates "
            "must be removed before training; the other items are recommendations."
        )
        text.setWordWrap(True)
        root.addWidget(text)

        content = QHBoxLayout()
        self.issue_table = QTableWidget(len(summary.issues), 3)
        self.issue_table.setObjectName("photoIssueTable")
        self.issue_table.setHorizontalHeaderLabels(("Check", "File", "What to review"))
        self.issue_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.issue_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.issue_table.verticalHeader().setVisible(False)
        header = self.issue_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for row, issue in enumerate(summary.issues):
            self.issue_table.setItem(
                row,
                0,
                QTableWidgetItem(_ISSUE_NAMES[issue.kind]),
            )
            self.issue_table.setItem(
                row,
                1,
                QTableWidgetItem(self._relative(issue.path)),
            )
            self.issue_table.setItem(row, 2, QTableWidgetItem(issue.detail))
        self.issue_table.currentCellChanged.connect(self._selection_changed)
        content.addWidget(self.issue_table, 3)

        preview_column = QVBoxLayout()
        self.preview = QLabel("Select a row to preview its photo.")
        self.preview.setObjectName("photoIssuePreview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(340, 360)
        self.preview.setWordWrap(True)
        preview_column.addWidget(self.preview, 1)
        self.preview_name = QLabel()
        self.preview_name.setWordWrap(True)
        preview_column.addWidget(self.preview_name)
        content.addLayout(preview_column, 2)
        root.addLayout(content, 1)

        actions = QHBoxLayout()
        open_folder = QPushButton("Open photo folder")
        open_folder.clicked.connect(self._open_folder)
        actions.addWidget(open_folder)
        actions.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        actions.addWidget(close)
        recheck = QPushButton("Check photos again")
        recheck.setObjectName("recheckPhotosButton")
        recheck.clicked.connect(self.accept)
        actions.addWidget(recheck)
        root.addLayout(actions)

        if summary.issues:
            self.issue_table.selectRow(0)
        else:
            self.issue_table.setVisible(False)
            self.preview.setText(
                "✓\n\nNo file-specific concerns were found in this photo set."
            )

    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.summary.folder).as_posix()
        except ValueError:
            return path.name

    def _selection_changed(
        self,
        current_row: int,
        _current_column: int,
        _previous_row: int,
        _previous_column: int,
    ) -> None:
        if not 0 <= current_row < len(self.summary.issues):
            return
        issue: SourceIssue = self.summary.issues[current_row]
        pixmap = QPixmap(str(issue.path))
        self.preview.setPixmap(QPixmap())
        if pixmap.isNull():
            self.preview.setText("This file cannot be previewed.")
        else:
            self.preview.setText("")
            self.preview.setPixmap(
                pixmap.scaled(
                    340,
                    360,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self.preview_name.setText(self._relative(issue.path))

    def _open_folder(self) -> None:
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.summary.folder))):
            QMessageBox.warning(
                self,
                "Folder could not be opened",
                f"Windows could not open:\n{self.summary.folder}",
            )
