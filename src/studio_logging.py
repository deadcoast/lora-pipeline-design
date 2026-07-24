"""Small, local-only rotating log for the desktop application's support trail."""

from __future__ import annotations

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4


class StudioLog:
    """Persist a bounded support trail without making logging an app dependency.

    The log intentionally records application events and pipeline text only. It
    never records environment variables, photo contents, or sends data over the
    network. Logging is best-effort so a locked or unwritable support directory
    cannot prevent a casual user from opening the studio.
    """

    def __init__(
        self,
        directory: Path,
        *,
        max_bytes: int = 1_000_000,
        backup_count: int = 4,
    ) -> None:
        self.directory = directory.expanduser()
        self.path = self.directory / "studio.log"
        self.error: str | None = None
        self._closed = False
        self._handler: RotatingFileHandler | None = None
        self._logger = logging.getLogger(f"mivim.studio.{uuid4().hex}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                self.path,
                maxBytes=max(1, max_bytes),
                backupCount=max(1, backup_count),
                encoding="utf-8",
                delay=True,
            )
        except OSError as exc:
            self.error = str(exc)
            return
        formatter = logging.Formatter(
            "%(asctime)sZ | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        self._handler = handler
        self._logger.addHandler(handler)

    @property
    def available(self) -> bool:
        return self._handler is not None and not self._closed

    def info(self, event: str, message: str) -> None:
        self._write(logging.INFO, event, message)

    def warning(self, event: str, message: str) -> None:
        self._write(logging.WARNING, event, message)

    def error_event(self, event: str, message: str) -> None:
        self._write(logging.ERROR, event, message)

    def exception_details(
        self,
        event: str,
        message: str,
        traceback_text: str,
    ) -> None:
        """Write a plain failure summary followed by its local stack context."""

        self._write(logging.ERROR, event, message)
        for line in traceback_text.splitlines():
            self._write(logging.ERROR, f"{event}.trace", line)

    def pipeline_output(self, text: str) -> None:
        """Persist the real merged worker output one bounded log record per line."""

        for line in text.splitlines():
            self._write(logging.INFO, "pipeline.output", line)

    def flush(self) -> None:
        if self._handler is not None:
            self._handler.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        handler = self._handler
        self._handler = None
        if handler is not None:
            handler.flush()
            self._logger.removeHandler(handler)
            handler.close()

    def _write(self, level: int, event: str, message: str) -> None:
        if not self.available:
            return
        clean_event = (
            "".join(
                character
                for character in event.strip().lower()
                if character.isalnum() or character in "._-"
            )
            or "studio"
        )
        clean_message = " ".join(
            str(message).replace("\x00", "�").replace("\r", "").splitlines()
        )
        self._logger.log(level, "%s | %s", clean_event, clean_message)


def log_directory_for(settings_path: Path) -> Path:
    """Keep support logs beside the user's versioned settings, never in the repo."""

    return settings_path.expanduser().parent / "logs"
