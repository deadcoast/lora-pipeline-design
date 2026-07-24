"""User-facing review of prepared photos and their training captions."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .app_service import caption_contains_trigger
from .dataset_pairs import scan_dataset_pairs
from .io_utils import atomic_write_text


class PreparedDatasetDialog(QDialog):
    """Let a casual user inspect every generated pair without leaving the app."""

    def __init__(
        self,
        dataset: Path,
        required_trigger: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.dataset = dataset
        self.required_trigger = required_trigger.strip()
        if not self.required_trigger:
            raise ValueError("A required identity phrase must be provided.")
        self.inventory = scan_dataset_pairs(dataset)
        self.pairs = self.inventory.review_pairs
        self._current_row = -1
        self._loading = False

        self.setWindowTitle("Review prepared photos and captions")
        self.setMinimumSize(900, 620)
        root = QVBoxLayout(self)
        heading = QLabel("Review what the trainer will learn")
        heading.setObjectName("pageTitle")
        root.addWidget(heading)
        guidance = QLabel(
            "Select each photo, correct its description if needed, and save. "
            f"The identity phrase “{self.required_trigger}” must remain in every "
            "caption."
        )
        guidance.setWordWrap(True)
        root.addWidget(guidance)
        self.file_issues_label = QLabel()
        self.file_issues_label.setObjectName("preparedFileIssues")
        self.file_issues_label.setWordWrap(True)
        file_problems = list(self.inventory.problem_messages())
        for _image_path, caption_path in self.pairs:
            if not caption_path.is_file():
                continue
            try:
                caption = caption_path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                file_problems.append(f"{caption_path.name} cannot be read.")
            else:
                if not caption:
                    file_problems.append(f"{caption_path.name} is empty.")
                elif not caption_contains_trigger(caption, self.required_trigger):
                    file_problems.append(
                        f"{caption_path.name} is without the identity phrase."
                    )
        if file_problems:
            self.file_issues_label.setText(
                "Files that need attention:\n• "
                + "\n• ".join(file_problems)
                + "\nUse Open prepared folder for duplicates or unsupported files. "
                "Select a listed image to correct or create its caption."
            )
            root.addWidget(self.file_issues_label)

        content = QHBoxLayout()
        self.photo_list = QListWidget()
        self.photo_list.setObjectName("preparedPhotoList")
        self.photo_list.setAccessibleName("Prepared training photos")
        self.photo_list.addItems([image.name for image, _caption in self.pairs])
        self.photo_list.currentRowChanged.connect(self._row_changed)
        content.addWidget(self.photo_list, 1)

        editor = QVBoxLayout()
        self.preview = QLabel("No prepared photos were found.")
        self.preview.setObjectName("preparedPhotoPreview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(520, 360)
        self.preview.setAccessibleName("Selected prepared training photo")
        editor.addWidget(self.preview, 1)
        caption_label = QLabel("Training caption")
        caption_label.setObjectName("cardTitle")
        editor.addWidget(caption_label)
        self.caption_edit = QTextEdit()
        self.caption_edit.setObjectName("preparedCaptionEdit")
        self.caption_edit.setAccessibleName("Caption for selected training photo")
        self.caption_edit.setMaximumHeight(110)
        editor.addWidget(self.caption_edit)
        self.position_label = QLabel()
        editor.addWidget(self.position_label)
        content.addLayout(editor, 3)
        root.addLayout(content, 1)

        actions = QHBoxLayout()
        open_folder = QPushButton("Open prepared folder")
        open_folder.clicked.connect(self._open_folder)
        actions.addWidget(open_folder)
        actions.addStretch(1)
        previous = QPushButton("Previous")
        previous.clicked.connect(lambda: self._move(-1))
        actions.addWidget(previous)
        save = QPushButton("Save caption")
        save.setObjectName("savePreparedCaptionButton")
        save.clicked.connect(self._save_current)
        actions.addWidget(save)
        next_button = QPushButton("Save & next")
        next_button.setObjectName("nextPreparedCaptionButton")
        next_button.clicked.connect(lambda: self._move(1))
        actions.addWidget(next_button)
        done = QPushButton("Done")
        done.setObjectName("closePreparedReviewButton")
        done.clicked.connect(self._finish)
        actions.addWidget(done)
        root.addLayout(actions)

        if self.pairs:
            self.photo_list.setCurrentRow(0)
        else:
            self.caption_edit.setEnabled(False)

    def _row_changed(self, row: int) -> None:
        if self._loading or row == self._current_row:
            return
        if (
            self._current_row >= 0
            and self.caption_edit.document().isModified()
            and not self._save_current()
        ):
            with QSignalBlocker(self.photo_list):
                self.photo_list.setCurrentRow(self._current_row)
            return
        self._load_row(row)

    def _load_row(self, row: int) -> None:
        if not 0 <= row < len(self.pairs):
            return
        self._loading = True
        try:
            image_path, caption_path = self.pairs[row]
            pixmap = QPixmap(str(image_path))
            self.preview.setPixmap(
                pixmap.scaled(
                    520,
                    360,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            try:
                caption = (
                    caption_path.read_text(encoding="utf-8").strip()
                    if caption_path.is_file()
                    else ""
                )
            except (OSError, UnicodeError) as exc:
                QMessageBox.critical(
                    self,
                    "Caption could not be opened",
                    f"{caption_path.name}\n\n{exc}",
                )
                caption = ""
            self.caption_edit.setPlainText(caption)
            self.caption_edit.document().setModified(False)
            self.position_label.setText(
                f"{row + 1} of {len(self.pairs)}  •  {image_path.name}"
                + (
                    f"  •  Missing {caption_path.name}; save to create it"
                    if not caption_path.is_file()
                    else ""
                )
            )
            self._current_row = row
        finally:
            self._loading = False

    def _save_current(self) -> bool:
        if not 0 <= self._current_row < len(self.pairs):
            return True
        caption = self.caption_edit.toPlainText().strip()
        if not caption:
            QMessageBox.warning(
                self,
                "Caption cannot be empty",
                "Describe the photo or restore the identity phrase before continuing.",
            )
            self.caption_edit.setFocus()
            return False
        if not caption_contains_trigger(caption, self.required_trigger):
            answer = QMessageBox.question(
                self,
                "Identity phrase is required",
                f"Keep “{self.required_trigger}” in every caption so the trainer "
                "can learn when to use this identity.\n\n"
                "Add it to the start of this caption now?",
                (QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel),
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.caption_edit.setFocus()
                return False
            caption = f"{self.required_trigger}, {caption.lstrip(' ,')}"
            self.caption_edit.setPlainText(caption)
            self.caption_edit.document().setModified(True)
        caption_path = self.pairs[self._current_row][1]
        try:
            atomic_write_text(caption_path, f"{caption}\n")
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Caption could not be saved",
                f"{caption_path.name}\n\n{exc}",
            )
            return False
        self.caption_edit.document().setModified(False)
        return True

    def _move(self, offset: int) -> None:
        if not self._save_current():
            return
        target = min(max(self._current_row + offset, 0), len(self.pairs) - 1)
        self.photo_list.setCurrentRow(target)

    def _finish(self) -> None:
        if self._save_current():
            self.accept()

    def _confirm_unsaved_close(self) -> bool:
        if not self.caption_edit.document().isModified():
            return True
        answer = QMessageBox.question(
            self,
            "Save this caption?",
            "The current caption has unsaved changes.",
            (
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel
            ),
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self._save_current()
        if answer == QMessageBox.StandardButton.Discard:
            self.caption_edit.document().setModified(False)
            return True
        return False

    def reject(self) -> None:
        if self._confirm_unsaved_close():
            super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._confirm_unsaved_close():
            event.accept()
        else:
            event.ignore()

    def _open_folder(self) -> None:
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.dataset))):
            QMessageBox.warning(
                self,
                "Folder could not be opened",
                f"Windows could not open:\n{self.dataset}",
            )
