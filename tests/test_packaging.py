from pathlib import Path

from PIL import Image

from scripts.generate_windows_icon import create_icon


def test_windows_icon_is_a_real_multiresolution_ico(tmp_path: Path) -> None:
    destination = create_icon(tmp_path / "mivim.ico")

    with Image.open(destination) as icon:
        assert icon.format == "ICO"
        assert icon.size == (256, 256)
        sizes = icon.info.get("sizes")

    assert isinstance(sizes, set)
    assert {(16, 16), (32, 32), (48, 48), (256, 256)} <= sizes
