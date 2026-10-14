"""Library page widgets: one-word tag fields, the tag dialog, and the run rows."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QStringListModel, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .library import Library, LibraryEntry, RunTags, entries_matching
from .model_profiles import model_profile
from .tags import TAG_RULE, normalize_tag, tag_error

THUMBNAIL_WIDTH = 132
THUMBNAIL_HEIGHT = 88

STATUS_TEXT: dict[str, tuple[str, str]] = {
    "ready": ("Ready in ComfyUI", "statusReady"),
    "lora_missing": ("LoRA missing", "statusWarning"),
    "result_missing": ("Result missing", "statusWarning"),
    "validation_pending": ("Validation pending", "statusWarning"),
    "prepared": ("Prepared only", "statusWarning"),
    "incomplete": ("Incomplete", "statusError"),
}


def style_label(style: str) -> str:
    """Display form of a style tag: ``anime`` becomes ``Anime``."""
    return style[:1].upper() + style[1:] if style else "Unlabelled"


def profile_label(profile_id: str) -> str:
    try:
        return model_profile(profile_id).label
    except ValueError:
        return profile_id or "Profile not recorded"


def _label(text: str, object_name: str, *, wrap: bool = False) -> QLabel:
    label = QLabel(text)
    label.setObjectName(object_name)
    label.setWordWrap(wrap)
    return label


def _clear_layout(layout: QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            continue
        child = item.layout()
        widget = item.widget()
        if child is not None:
            _clear_layout(child)
        if widget is not None:
            widget.deleteLater()


class TagField(QWidget):
    """A one-word tag editor with autocomplete and a hint that names new words."""

    changed = Signal(str)

    def __init__(self, kind: str, title: str, placeholder: str = "") -> None:
        super().__init__()
        self.kind = kind
        self._vocabulary: tuple[str, ...] = ()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.title = _label(title, "cardTitle")
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setAccessibleName(title)
        self.edit.setMaxLength(48)
        self._model = QStringListModel([])
        completer = QCompleter(self._model, self.edit)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.edit.setCompleter(completer)
        self.hint = _label(TAG_RULE, "hintText", wrap=True)
        layout.addWidget(self.title)
        layout.addWidget(self.edit)
        layout.addWidget(self.hint)
        self.edit.textChanged.connect(self._text_changed)

    def set_vocabulary(self, words: Iterable[str]) -> None:
        self._vocabulary = tuple(sorted(set(words)))
        self._model.setStringList(list(self._vocabulary))
        self._text_changed(self.edit.text())

    def value(self) -> str:
        return normalize_tag(self.edit.text())

    def set_value(self, text: str) -> None:
        self.edit.setText(normalize_tag(text))

    def error(self) -> str | None:
        return tag_error(self.value())

    def _text_changed(self, text: str) -> None:
        lowered = text.lower()
        if lowered != text:
            position = self.edit.cursorPosition()
            self.edit.setText(lowered)
            self.edit.setCursorPosition(position)
            return
        value = normalize_tag(text)
        if not value:
            hint = TAG_RULE
        elif (issue := tag_error(value)) is not None:
            hint = issue
        elif value in self._vocabulary:
            hint = f"Existing {self.kind}: {value}"
        else:
            hint = f"New {self.kind}: {value} will be added"
        self.hint.setText(hint)
        self.changed.emit(value)


class TagRunDialog(QDialog):
    """File one run under a character and an art style."""

    def __init__(
        self,
        project_name: str,
        characters: Iterable[str],
        styles: Iterable[str],
        current: RunTags | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tag this run")
        self.setMinimumWidth(420)
        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.addWidget(_label(project_name, "pageTitle"))
        root.addWidget(
            _label(
                "One word each. Pick an existing word when one fits so the "
                "library stays tidy.",
                "cardBody",
                wrap=True,
            )
        )
        self.character_field = TagField("character", "Character", "ruth")
        self.artstyle_field = TagField("style", "Art style", "anime")
        self.character_field.set_vocabulary(characters)
        self.artstyle_field.set_vocabulary(styles)
        if current is not None:
            self.character_field.set_value(current.character)
            self.artstyle_field.set_value(current.artstyle)
        root.addWidget(self.character_field)
        root.addWidget(self.artstyle_field)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)
        self.character_field.changed.connect(self._validate)
        self.artstyle_field.changed.connect(self._validate)
        self._validate()

    def _validate(self, *_args: object) -> None:
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(
            self.character_field.error() is None and self.artstyle_field.error() is None
        )

    def tags(self) -> RunTags:
        return RunTags(self.character_field.value(), self.artstyle_field.value())


class LibraryRow(QFrame):
    open_requested = Signal(object)
    next_requested = Signal(object)
    tag_requested = Signal(object)

    def __init__(self, entry: LibraryEntry) -> None:
        super().__init__()
        self.entry = entry
        self.setObjectName("libraryRow")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(14)

        self.thumbnail = QLabel("No grid")
        self.thumbnail.setObjectName("libraryThumbnail")
        self.thumbnail.setFixedSize(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)
        self.thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail.setAccessibleName(
            f"Validation grid preview for {entry.project_name}"
        )
        if entry.grid_path is not None:
            self.thumbnail.setText("…")
        layout.addWidget(self.thumbnail)

        text = QVBoxLayout()
        text.setSpacing(5)
        self.name_label = _label(entry.project_name, "cardTitle")
        text.addWidget(self.name_label)
        chips = QHBoxLayout()
        chips.setSpacing(8)
        self.style_chip = _label(
            entry.artstyle or "untagged",
            "styleChip" if entry.artstyle else "untaggedChip",
        )
        status_text, status_name = STATUS_TEXT[entry.status]
        self.status_chip = _label(status_text, status_name)
        for chip in (self.style_chip, self.status_chip):
            chip.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            chips.addWidget(chip)
        chips.addStretch(1)
        text.addLayout(chips)
        model = entry.checkpoint or "Base model not recorded"
        self.model_label = _label(
            f"{model} · {profile_label(entry.model_profile)}", "hintText", wrap=True
        )
        self.model_label.setToolTip(f"MODEL :: {style_label(entry.artstyle)}")
        text.addWidget(self.model_label)
        details = [
            f"Trigger {entry.trigger}" if entry.trigger else "Trigger not recorded"
        ]
        if entry.steps:
            details.append(f"{entry.steps:,} steps")
        if entry.completed_at:
            details.append(entry.completed_at[:16].replace("T", " "))
        self.detail_label = _label(" · ".join(details), "hintText", wrap=True)
        text.addWidget(self.detail_label)
        layout.addLayout(text, 1)

        actions = QVBoxLayout()
        actions.setSpacing(6)
        self.open_button = QPushButton("Open result")
        self.open_button.setObjectName("libraryOpenButton")
        self.open_button.setEnabled(entry.record is not None)
        self.open_button.clicked.connect(lambda: self.open_requested.emit(self.entry))
        self.next_button = QPushButton("Train next version")
        self.next_button.setObjectName("libraryNextButton")
        self.next_button.clicked.connect(lambda: self.next_requested.emit(self.entry))
        self.tag_button = QPushButton("Edit tags" if entry.tagged else "Tag")
        self.tag_button.setObjectName("libraryTagButton")
        self.tag_button.clicked.connect(lambda: self.tag_requested.emit(self.entry))
        for button in (self.open_button, self.next_button, self.tag_button):
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)

    def set_thumbnail(self, pixmap: QPixmap) -> None:
        self.thumbnail.setPixmap(
            pixmap.scaled(
                THUMBNAIL_WIDTH,
                THUMBNAIL_HEIGHT,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class LibraryPage(QWidget):
    """Every trained model, grouped by character, filtered by style."""

    open_requested = Signal(object)
    next_requested = Signal(object)
    tag_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._library = Library(())
        self._style = ""
        self._groups: list[tuple[str, list[str]]] = []
        self._unsorted: list[str] = []
        self.rows: dict[str, LibraryRow] = {}
        self.style_buttons: list[QPushButton] = []

        page = QVBoxLayout(self)
        page.setContentsMargins(34, 30, 34, 34)
        page.setSpacing(16)
        header = QVBoxLayout()
        header.setSpacing(5)
        header.addWidget(_label("LIBRARY", "eyebrow"))
        header.addWidget(_label("Your trained models", "pageTitle"))
        header.addWidget(
            _label(
                "Every run, filed by character and art style. Open a result, "
                "train the next version with the same setup, or tag older runs.",
                "pageSubtitle",
                wrap=True,
            )
        )
        page.addLayout(header)

        self.warning_label = _label("", "statusWarning", wrap=True)
        self.warning_label.setVisible(False)
        page.addWidget(self.warning_label)

        filters = QHBoxLayout()
        filters.setSpacing(8)
        self.style_bar = QHBoxLayout()
        self.style_bar.setSpacing(6)
        filters.addLayout(self.style_bar)
        filters.addStretch(1)
        self.search = QLineEdit()
        self.search.setObjectName("librarySearch")
        self.search.setPlaceholderText("Search name, character, or trigger")
        self.search.setClearButtonEnabled(True)
        self.search.setMaximumWidth(320)
        self.search.textChanged.connect(lambda _text: self._rebuild())
        filters.addWidget(self.search)
        page.addLayout(filters)

        self.empty_label = _label(
            "No trained models yet. Connect to Comfy Desktop, then start a new project.",
            "cardBody",
            wrap=True,
        )
        page.addWidget(self.empty_label)

        content = QWidget()
        content.setObjectName("scrollPageContent")
        self.groups_layout = QVBoxLayout(content)
        self.groups_layout.setContentsMargins(0, 0, 0, 0)
        self.groups_layout.setSpacing(14)
        self.groups_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setObjectName("libraryScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        page.addWidget(scroll, 1)

    # ---------------------------------------------------------------- state
    def set_library(self, library: Library) -> None:
        self._library = library
        styles = library.styles()
        if self._style and self._style not in styles:
            self._style = ""
        _clear_layout(self.style_bar)
        self.style_buttons = []
        for style in ("", *styles):
            button = QPushButton("All styles" if not style else style)
            button.setObjectName("filterChip")
            button.setCheckable(True)
            button.setChecked(style == self._style)
            button.clicked.connect(
                lambda _checked, chosen=style: self._choose_style(chosen)
            )
            self.style_bar.addWidget(button)
            self.style_buttons.append(button)
        self._rebuild()

    def set_warning(self, text: str) -> None:
        self.warning_label.setText(text)
        self.warning_label.setVisible(bool(text))

    def groups(self) -> list[tuple[str, list[str]]]:
        return [(character, list(names)) for character, names in self._groups]

    def unsorted_names(self) -> list[str]:
        return list(self._unsorted)

    def entries_needing_thumbnails(self) -> list[LibraryEntry]:
        return [
            row.entry
            for row in self.rows.values()
            if row.entry.grid_path is not None and row.thumbnail.pixmap().isNull()
        ]

    def set_thumbnail(self, project_name: str, pixmap: QPixmap) -> None:
        row = self.rows.get(project_name)
        if row is not None:
            row.set_thumbnail(pixmap)

    # -------------------------------------------------------------- building
    def _choose_style(self, style: str) -> None:
        self._style = style
        for button, chosen in zip(self.style_buttons, ("", *self._library.styles())):
            button.setChecked(chosen == style)
        self._rebuild()

    def _rebuild(self) -> None:
        stretch = self.groups_layout.takeAt(self.groups_layout.count() - 1)
        _clear_layout(self.groups_layout)
        self.rows = {}
        self._groups = []
        self._unsorted = []
        search = self.search.text()
        for character, entries in self._library.characters():
            shown = entries_matching(entries, style=self._style, search=search)
            if not shown:
                continue
            self._groups.append((character, [entry.project_name for entry in shown]))
            self.groups_layout.addWidget(
                self._group_card(
                    character,
                    f"{len(entries)} run" + ("" if len(entries) == 1 else "s"),
                    shown,
                )
            )
        unsorted = entries_matching(
            self._library.unsorted(), style=self._style, search=search
        )
        if unsorted:
            self._unsorted = [entry.project_name for entry in unsorted]
            self.groups_layout.addWidget(
                self._group_card(
                    "Unsorted",
                    "Tag these runs to file them by character and style.",
                    unsorted,
                )
            )
        if stretch is not None:
            self.groups_layout.addItem(stretch)
        else:
            self.groups_layout.addStretch(1)
        self.empty_label.setVisible(not self._library.entries)

    def _group_card(
        self, title: str, subtitle: str, entries: Iterable[LibraryEntry]
    ) -> QFrame:
        card = QFrame()
        card.setObjectName("libraryGroup")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)
        heading = QHBoxLayout()
        heading.addWidget(_label(title, "libraryGroupTitle"))
        heading.addWidget(_label(subtitle, "hintText"), 1)
        layout.addLayout(heading)
        for entry in entries:
            row = LibraryRow(entry)
            row.open_requested.connect(self.open_requested)
            row.next_requested.connect(self.next_requested)
            row.tag_requested.connect(self.tag_requested)
            layout.addWidget(row)
            self.rows[entry.project_name] = row
        return card
