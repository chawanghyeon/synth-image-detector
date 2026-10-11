"""JPEG / resize degradations: fixed evaluation conditions and the random training augmentation."""

import io
import random
from collections.abc import Callable

from PIL import Image


def jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality)
    return Image.open(buffer).convert("RGB")


def rescale(image: Image.Image, factor: float) -> Image.Image:
    size = (max(64, int(image.width * factor)), max(64, int(image.height * factor)))
    return image.resize(size, Image.Resampling.BICUBIC)


def sns(image: Image.Image, factor: float = 0.75, quality: int = 80) -> Image.Image:
    """SNS-style copy: downscale, then JPEG."""
    return jpeg(rescale(image, factor), quality)


CONDITIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "original": lambda image: image,
    "jpeg75": lambda image: jpeg(image, 75),
    "jpeg50": lambda image: jpeg(image, 50),
    "half": lambda image: rescale(image, 0.5),
    "sns": sns,  # 0.75x + JPEG 80
}


def degrade(image: Image.Image, condition: str) -> Image.Image:
    return CONDITIONS[condition](image)


def augment(image: Image.Image, rng: random.Random, always: bool = False) -> Image.Image:
    """Training copy: 25% original, 35% JPEG 50-95, 40% SNS chain (0.5-1x + JPEG 60-90).
    `always` never returns the original."""
    r = rng.uniform(0.25 if always else 0, 1)
    if r < 0.25:
        return image
    if r < 0.6:
        return jpeg(image, rng.randint(50, 95))
    return sns(image, rng.uniform(0.5, 1.0), rng.randint(60, 90))
