"""Desktop comparison of saved training previews, with an explicit weight choice."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .training_checkpoints import TrainingCheckpoint


class CheckpointReviewDialog(QDialog):
    def __init__(
        self, choices: tuple[TrainingCheckpoint, ...], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        if not choices:
            raise ValueError("No readable training checkpoints were found.")
        self.choices = choices
        self.setWindowTitle("Compare training checkpoints")
        self.resize(1060, 750)
        self.setMinimumSize(760, 560)
        root = QVBoxLayout(self)
        title = QLabel("Choose the training step that works best")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        guidance = QLabel(
            "Compare the same preview across training steps. Look for likeness, "
            "natural details, and variety in clothing and surroundings. "
            "An earlier step may keep more flexibility."
        )
        guidance.setWordWrap(True)
        root.addWidget(guidance)
        self.prompt_combo = QComboBox()
        self.prompt_combo.setObjectName("checkpointPreviewPrompt")
        self.prompt_combo.setAccessibleName("Training preview prompt")
        indexes = sorted({index for choice in choices for index, _ in choice.previews})
        for index in indexes:
            self.prompt_combo.addItem(f"Training preview {index + 1}", index)
        root.addWidget(self.prompt_combo)
        columns = QHBoxLayout()
        self.reference_combo = QComboBox()
        self.reference_combo.setObjectName("referenceCheckpoint")
        self.choice_combo = QComboBox()
        self.choice_combo.setObjectName("selectedCheckpoint")
        self.images: list[QLabel] = []
        for label, combo in (
            ("Compare with", self.reference_combo),
            ("Checkpoint to validate", self.choice_combo),
        ):
            column = QVBoxLayout()
            column.addWidget(QLabel(label))
            combo.setAccessibleName(label)
            combo.addItems([choice.label for choice in choices])
            column.addWidget(combo)
            preview = QLabel()
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview.setMinimumSize(300, 300)
            preview.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
            )
            preview.setWordWrap(True)
            self.images.append(preview)
            column.addWidget(preview, 1)
            columns.addLayout(column, 1)
        root.addLayout(columns, 1)
        self.reference_combo.setCurrentIndex(len(choices) - 1)
        self.choice_combo.setCurrentIndex(max(0, len(choices) - 2))
        note = QLabel(
            "These are the trainer's saved previews. Validate your choice to generate "
            "a fresh baseline and strength comparison in ComfyUI. Each choice gets "
            "its own result; your existing LoRA and grid stay available."
        )
        note.setWordWrap(True)
        root.addWidget(note)
        actions = QHBoxLayout()
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        actions.addWidget(close)
        actions.addStretch(1)
        self.validate_button = QPushButton("Validate this checkpoint")
        self.validate_button.setObjectName("validateCheckpointButton")
        self.validate_button.clicked.connect(self.accept)
        actions.addWidget(self.validate_button)
        root.addLayout(actions)
        for combo in (self.reference_combo, self.choice_combo, self.prompt_combo):
            combo.currentIndexChanged.connect(self._render_previews)
        self._render_previews()

    @property
    def selected_checkpoint(self) -> TrainingCheckpoint:
        return self.choices[self.choice_combo.currentIndex()]

    def _render_previews(self) -> None:
        prompt_index = self.prompt_combo.currentData()
        for combo, image in zip(
            (self.reference_combo, self.choice_combo), self.images, strict=True
        ):
            choice = self.choices[combo.currentIndex()]
            path = dict(choice.previews).get(prompt_index)
            pixmap = QPixmap(str(path)) if path is not None else QPixmap()
            image.setAccessibleName(f"{choice.label}, preview {prompt_index}")
            if pixmap.isNull():
                image.clear()
                image.setText("No saved preview for this prompt and training step.")
            else:
                image.setPixmap(
                    pixmap.scaled(
                        image.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._render_previews()
