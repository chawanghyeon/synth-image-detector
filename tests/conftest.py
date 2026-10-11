import io

import pytest
from PIL import Image


def noise(size: tuple[int, int], sigma: int = 40) -> Image.Image:
    return Image.effect_noise(size, sigma).convert("RGB")


def png_bytes(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    noise(size).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture
def crops(tmp_path):
    """Eight 64x64 crops on disk and their labels, alternating AI (1) / real (0)."""
    files = []
    for i in range(8):
        path = tmp_path / f"crop{i}.png"
        noise((64, 64), 30 + i).save(path)
        files.append(str(path))
    return files, [i % 2 for i in range(8)]
