from __future__ import annotations

from pathlib import Path

from src.studio_logging import StudioLog


def test_studio_log_rotates_and_stays_inside_its_local_directory(
    tmp_path: Path,
) -> None:
    studio_log = StudioLog(tmp_path / "support", max_bytes=320, backup_count=2)

    for index in range(30):
        studio_log.info("rotation.test", f"record={index}; detail={'x' * 80}")
    studio_log.flush()

    assert studio_log.available
    assert studio_log.path.parent == tmp_path / "support"
    assert studio_log.path.is_file()
    assert (tmp_path / "support" / "studio.log.1").is_file()
    assert len(list((tmp_path / "support").glob("studio.log*"))) <= 3
    assert "rotation.test" in studio_log.path.read_text(encoding="utf-8")
    studio_log.close()
