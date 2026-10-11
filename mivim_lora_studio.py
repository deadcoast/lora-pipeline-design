no """Deployment entry point for MIVIM LoRA Studio."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def _report_startup_failure(error: Exception) -> None:
    root = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    log_path = root / "MIVIM" / "LoRA Pipeline" / "startup-error.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
    except OSError:
        pass
    message = (
        "MIVIM LoRA Studio could not start.\n\n"
        f"{error}\n\n"
        f"Technical details: {log_path}"
    )
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(  # type: ignore[attr-defined]
                0,
                message,
                "MIVIM LoRA Studio",
                0x10,
            )
            return
        except (AttributeError, OSError):
            pass
    print(message, file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pipeline-worker":
        from src.pipeline import main as pipeline_main

        pipeline_main(sys.argv[2:])
    else:
        if "__compiled__" in globals() or getattr(sys, "frozen", False):
            os.environ["MIVIM_PACKAGED_APP"] = "1"
        try:
            from src.desktop import main as desktop_main

            raise SystemExit(desktop_main(sys.argv[1:]))
        except Exception as exc:
            _report_startup_failure(exc)
            raise SystemExit(1) from exc
