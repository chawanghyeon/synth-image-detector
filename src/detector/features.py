"""Input for the student: one native-resolution crop, resized to the model size.

The crop is a centred square of at most 512 px, cut before any resizing, as in training
(scripts/prepare_data.py cuts the same kind of crop). Smaller images are used whole, never upscaled
before cropping.
"""

import numpy as np
from PIL import Image

CROP = 512
MIN_SIDE = 256


def student_crop(image: Image.Image, size: int) -> np.ndarray:
    """(1, 3, size, size) float32 in [-1, 1]."""
    width, height = image.size
    if min(width, height) < MIN_SIDE:
        raise ValueError(f"Image must be at least {MIN_SIDE}x{MIN_SIDE} pixels")
    side = min(CROP, width, height) // 2 * 2
    left, top = (width - side) // 2, (height - side) // 2
    crop = image.convert("RGB").crop((left, top, left + side, top + side))
    a = (
        np.asarray(crop.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32) / 127.5
        - 1
    )
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])
