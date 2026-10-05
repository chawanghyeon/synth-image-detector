"""Inputs for the two detectors. Images are never resized, so pixel-level traces survive.

SPAM (Pevny et al. 2010): camera pipelines and image generators leave different local noise
dependencies. Each feature is a second-order Markov transition probability of truncated pixel
differences, so it is independent of image size and content is largely removed.

Patches: 128px native-resolution tiles for the noise-residual CNN.
"""

import numpy as np
from PIL import Image

T = 3  # differences are clipped to [-T, T]
CROP = 512  # SPAM training used native-resolution crops of at most this size
MIN_SIDE = 256
PATCH = 128


def _transitions(d: np.ndarray, axis: int) -> np.ndarray:
    d = np.clip(np.round(d), -T, T).astype(np.int64) + T
    n = 2 * T + 1
    a, b, c = (np.take(d, range(i, d.shape[axis] - 2 + i), axis=axis) for i in range(3))
    joint = np.bincount(((a * n + b) * n + c).ravel(), minlength=n**3).astype(np.float64)
    pair = np.bincount((a * n + b).ravel(), minlength=n**2).astype(np.float64)
    return joint / np.maximum(np.repeat(pair, n), 1)


def _spam(x: np.ndarray) -> np.ndarray:
    straight = []
    for axis in (0, 1):
        forward = np.diff(x, axis=axis)
        straight += [_transitions(forward, axis), _transitions(-np.flip(forward, axis=axis), axis)]
    diag = x[1:, 1:] - x[:-1, :-1]
    anti = x[1:, :-1] - x[:-1, 1:]
    diagonal = [
        _transitions(diag, 1),
        _transitions(-diag[:, ::-1], 1),
        _transitions(anti, 1),
        _transitions(-anti[:, ::-1], 1),
    ]
    return np.concatenate([np.mean(straight, 0), np.mean(diagonal, 0)])


def spam_features(image: Image.Image) -> np.ndarray:
    """1372 features: SPAM of luminance and of the R-G colour difference."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    luma = rgb @ [0.299, 0.587, 0.114]
    chroma = rgb[..., 0] - rgb[..., 1]
    return np.concatenate([_spam(luma), _spam(chroma)]).astype(np.float32)


def crops(image: Image.Image, limit: int = 4) -> list[Image.Image]:
    """Up to `limit` native-resolution crops on an even grid; the image is never resized."""
    width, height = image.size
    if min(width, height) < MIN_SIDE:
        raise ValueError(f"Image must be at least {MIN_SIDE}x{MIN_SIDE} pixels")
    size = min(CROP, width, height)
    lefts = np.linspace(0, width - size, min(2, width // size)).astype(int)
    tops = np.linspace(0, height - size, min(2, height // size)).astype(int)
    boxes = [(x, y, x + size, y + size) for y in tops for x in lefts][:limit]
    return [image.crop(box) for box in boxes]


def patches(image: Image.Image, limit: int = 32) -> np.ndarray:
    """Up to `limit` non-overlapping 128px tiles on an even grid, as (N, 3, 128, 128) in [0, 1]."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    height, width = array.shape[:2]
    if height < PATCH or width < PATCH:
        raise ValueError(f"Image must be at least {PATCH}x{PATCH} pixels")
    rows = np.linspace(0, height - PATCH, max(1, min(height // PATCH, 8))).astype(int)
    cols = np.linspace(0, width - PATCH, max(1, min(width // PATCH, 8))).astype(int)
    boxes = [(r, c) for r in rows for c in cols]
    if len(boxes) > limit:
        boxes = [boxes[i] for i in np.linspace(0, len(boxes) - 1, limit).astype(int)]
    batch = np.stack([array[r : r + PATCH, c : c + PATCH] for r, c in boxes])
    return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))


SEMANTIC = 224
SEMANTIC_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SEMANTIC_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def semantic_crops(image: Image.Image) -> np.ndarray:
    """Up to four native-resolution 224px crops around the centre, normalised, (N, 3, 224, 224)."""
    a = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = a.shape[:2]
    if h < SEMANTIC or w < SEMANTIC:
        raise ValueError(f"Image must be at least {SEMANTIC}x{SEMANTIC} pixels")
    s, cy, cx = SEMANTIC, h // 2, w // 2
    ys = [max(0, cy - s), min(h - s, cy)] if h >= 2 * s else [(h - s) // 2]
    xs = [max(0, cx - s), min(w - s, cx)] if w >= 2 * s else [(w - s) // 2]
    crops = np.stack([a[y : y + s, x : x + s] for y in ys for x in xs]).astype(np.float32) / 255
    return np.ascontiguousarray(((crops - SEMANTIC_MEAN) / SEMANTIC_STD).transpose(0, 3, 1, 2))
