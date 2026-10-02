from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

from src import desktop
from src.app_models import AppSettings, ProjectDraft, RunResult, TrainingPreset
from src.dataset_review import PreparedDatasetDialog
from src.desktop import NAV_RESULTS, MainWindow
from tests.test_desktop import _OfflineLocalService, _wait_for_initial_setup
from tests.test_run_guardian import _start_holder, _stop_holder


def _review_window(qtbot, tmp_path: Path) -> tuple[MainWindow, Path]:
    dataset = tmp_path / "original-workspace" / "datasets" / "portrait_v1"
    dataset.mkdir(parents=True)
    Image.new("RGB", (64, 64), "blue").save(dataset / "001.png")
    (dataset / "001.txt").write_text("ohwx person, original\n", encoding="utf-8")
    window = MainWindow(
        settings=AppSettings(
            # The displayed result, not today's workspace/project form, owns
            # the dataset being reviewed.
            datasets_dir=tmp_path / "new-workspace" / "datasets",
            last_project=ProjectDraft(
                project_name="different_v2", trigger="new person"
            ),
        ),
        settings_path=tmp_path / "settings.json",
        service=_OfflineLocalService(root=Path(__file__).resolve().parents[1]),
    )
    qtbot.addWidget(window)
    window.show()
    _wait_for_initial_setup(qtbot, window)
    window._show_result(
        dataset,
        RunResult(
            project_name="portrait_v1",
            mode="prepare",
            result_path=dataset,
            completed_at="2026-09-16T12:00:00+00:00",
            trigger="ohwx person",
            preset=TrainingPreset.BALANCED,
        ),
    )
    window.show_page(NAV_RESULTS)
    return window, dataset


def _try_training(dataset: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Exercise the public trainer's real leases, replacing only its GPU body."""
    toolkit = tmp_path / "trainer"
    toolkit.mkdir(exist_ok=True)
    (toolkit / "run.py").touch()
    return subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
from src import config, train
from src.run_guardian import RunLeaseBusyError

dataset, toolkit = map(Path, sys.argv[1:])
config.AI_TOOLKIT_DIR = str(toolkit)
config.RUN_CONFIGS_DIR = toolkit / "configs"
train.check_orphaned_comfy_prompts = lambda: None
train._train_with_guard = lambda *_args, **_kwargs: 0
try:
    result = train.train(
        toolkit / "template.yaml",
        run_name="another_project_v1",
        dataset_dir=dataset,
        output_dir=toolkit / "checkpoints",
    )
except RunLeaseBusyError as exc:
    print(exc)
    sys.exit(23)
sys.exit(result)
""",
            str(dataset),
            str(toolkit),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def test_caption_review_refuses_a_dataset_owned_by_another_process(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard_root = tmp_path / "guard"
    monkeypatch.setenv("MIVIM_RUN_GUARD_DIR", str(guard_root))
    window, dataset = _review_window(qtbot, tmp_path)
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, title, message: errors.append((title, message)),
    )
    opened: list[bool] = []

    def edit_if_open() -> None:
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, PreparedDatasetDialog):
            opened.append(True)
            dialog.caption_edit.setPlainText("ohwx person, edited")
            qtbot.mouseClick(
                dialog.findChild(QPushButton, "closePreparedReviewButton"),
                Qt.MouseButton.LeftButton,
            )

    holder, _pid = _start_holder(
        tmp_path / "other-workspace",
        guard_root,
        "different_training_project",
        use_accelerator=True,
        resource_paths=(dataset,),
    )
    try:
        QTimer.singleShot(0, edit_if_open)
        qtbot.mouseClick(window.review_prepared_button, Qt.MouseButton.LeftButton)
        QApplication.processEvents()
        assert not opened
        assert errors and errors[0][0] == "Prepared photos are in use"
        assert "Wait" in errors[0][1]
        assert (dataset / "001.txt").read_text(
            encoding="utf-8"
        ) == "ohwx person, original\n"
    finally:
        _stop_holder(holder)

    QTimer.singleShot(0, edit_if_open)
    qtbot.mouseClick(window.review_prepared_button, Qt.MouseButton.LeftButton)
    assert opened == [True]
    assert (dataset / "001.txt").read_text(encoding="utf-8") == "ohwx person, edited\n"


@pytest.mark.parametrize("finish", ["done", "escape", "close"])
def test_caption_review_holds_the_dataset_until_the_user_finishes(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finish: str
) -> None:
    monkeypatch.setenv("MIVIM_RUN_GUARD_DIR", str(tmp_path / "guard"))
    window, dataset = _review_window(qtbot, tmp_path)
    attempts: list[subprocess.CompletedProcess[str]] = []
    stayed_open: list[bool] = []
    closed: list[bool] = []
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Cancel
    )

    def edit_and_finish() -> None:
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, PreparedDatasetDialog)
        try:
            attempts.append(_try_training(dataset, tmp_path))
            attempts.append(_try_training(tmp_path / "unrelated-dataset", tmp_path))
            dialog.caption_edit.setPlainText("ohwx person, edited")
            dialog.caption_edit.document().setModified(True)
            dialog.close()  # Cancel the unsaved-changes prompt.
            stayed_open.append(dialog.isVisible())
            attempts.append(_try_training(dataset, tmp_path))
            qtbot.mouseClick(
                dialog.findChild(QPushButton, "savePreparedCaptionButton"),
                Qt.MouseButton.LeftButton,
            )
            attempts.append(_try_training(dataset, tmp_path))
            if finish == "done":
                qtbot.mouseClick(
                    dialog.findChild(QPushButton, "closePreparedReviewButton"),
                    Qt.MouseButton.LeftButton,
                )
            elif finish == "escape":
                qtbot.keyClick(dialog, Qt.Key.Key_Escape)
            else:
                dialog.close()
            closed.append(not dialog.isVisible())
        finally:
            dialog.caption_edit.document().setModified(False)
            dialog.reject()

    QTimer.singleShot(0, edit_and_finish)
    qtbot.mouseClick(window.review_prepared_button, Qt.MouseButton.LeftButton)

    assert stayed_open == [True]
    assert closed == [True]
    assert [attempt.returncode for attempt in attempts] == [23, 0, 23, 23], [
        (attempt.stdout, attempt.stderr) for attempt in attempts
    ]
    assert "Close the photo and caption review" in attempts[0].stdout
    assert (dataset / "001.txt").read_text(encoding="utf-8") == "ohwx person, edited\n"
    after_close = _try_training(dataset, tmp_path)
    assert after_close.returncode == 0, after_close.stderr


def test_caption_review_releases_the_dataset_when_opening_fails(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MIVIM_RUN_GUARD_DIR", str(tmp_path / "guard"))
    window, dataset = _review_window(qtbot, tmp_path)
    errors: list[str] = []

    def fail_to_open(*_args):
        raise OSError("Prepared folder became unreadable")

    monkeypatch.setattr(desktop, "PreparedDatasetDialog", fail_to_open)
    monkeypatch.setattr(
        QMessageBox, "critical", lambda _parent, _title, message: errors.append(message)
    )
    qtbot.mouseClick(window.review_prepared_button, Qt.MouseButton.LeftButton)

    assert errors and "Prepared folder became unreadable" in errors[0]
    attempt = _try_training(dataset, tmp_path)
    assert attempt.returncode == 0, attempt.stderr
