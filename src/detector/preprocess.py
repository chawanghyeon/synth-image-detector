"""Image -> model input, shared by the server and by training/scoring.

Training images are native-resolution crops of at most 512 px (detector.data.factory). The server
cuts the same kind of crop before any resizing, so large images keep their fine detail; small
images are used whole and never upscaled before cropping. A crop policy returns the regions to
score; `to_array` turns one region into the model input.
"""

from typing import Protocol

import numpy as np
from PIL import Image

CROP = 512
MIN_SIDE = 256


class CropPolicy(Protocol):
    def __call__(self, image: Image.Image) -> list[Image.Image]: ...


class CenterCrop:
    """One centred square of min(crop, width, height) px (even), at native resolution."""

    def __init__(self, crop: int = CROP, min_side: int = MIN_SIDE) -> None:
        self.crop, self.min_side = crop, min_side

    def __call__(self, image: Image.Image) -> list[Image.Image]:
        width, height = image.size
        if min(width, height) < self.min_side:
            raise ValueError(f"Image must be at least {self.min_side}x{self.min_side} pixels")
        side = min(self.crop, width, height) // 2 * 2
        left, top = (width - side) // 2, (height - side) // 2
        return [image.convert("RGB").crop((left, top, left + side, top + side))]


def to_array(image: Image.Image, size: int) -> np.ndarray:
    """(3, size, size) float32 in [-1, 1]: the whole image, bicubic-resized."""
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    a = np.asarray(resized, dtype=np.float32) / 127.5 - 1
    return np.ascontiguousarray(a.transpose(2, 0, 1))


def model_input(image: Image.Image, size: int, policy: CropPolicy | None = None) -> np.ndarray:
    """(n_crops, 3, size, size) for every region the policy picks (default: CenterCrop)."""
    return np.stack([to_array(crop, size) for crop in (policy or CenterCrop())(image)])
