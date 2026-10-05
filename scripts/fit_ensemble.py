"""Fit how the two detectors are combined, on full-size validation images scored like the server.

Usage: python scripts/fit_ensemble.py
Writes models/ensemble.json (normalisation, weight, threshold) and reports/ensemble.json
(OpenFake test, original files and JPEG re-saves, at the fixed threshold ai_score > 0.5).
"""

import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score
from train_spam import jpeg, operating_point, threshold_at_fpr

from detector.detector import Detector

VALIDATION = "data/raw/core/validation-00000-of-00015.parquet"
TEST = "data/raw/core/test-00000-of-00013.parquet"
CACHE = Path("data/features")
_detector = None


def _raw(job: tuple[bytes, int | None]) -> tuple[float, float] | None:
    global _detector
    if _detector is None:
        os.environ["ORT_THREADS"] = "1"
        _detector = Detector(fitted=False)
    data, quality = job
    try:
        with Image.open(io.BytesIO(data)) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        return _detector.raw_scores(jpeg(image, quality) if quality else image)
    except (OSError, ValueError):
        return None  # too small or unreadable; the server answers 422 for these


def raw_scores(shard: str, quality: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Labels and (spam, cnn) raw scores for every usable image in a shard."""
    path = CACHE / f"server_{Path(shard).name.split('-')[0]}_{quality or 'original'}.npz"
    if path.exists():
        cached = np.load(path)
        return cached["labels"], cached["scores"]
    labels, scores = [], []
    parquet = pq.ParquetFile(shard)
    with ProcessPoolExecutor(int(os.getenv("WORKERS", "5"))) as pool:
        for group in range(parquet.num_row_groups):
            rows = parquet.read_row_group(group, columns=["image", "label"]).to_pylist()
            jobs = [(r["image"]["bytes"], quality) for r in rows]
            for row, result in zip(rows, pool.map(_raw, jobs)):
                if result is not None:
                    labels.append(int(row["label"] == "fake"))
                    scores.append(result)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, labels=np.array(labels), scores=np.array(scores))
    return np.array(labels), np.array(scores)


def main() -> None:
    y_val, raw_val = raw_scores(VALIDATION, None)
    real = raw_val[y_val == 0]
    mean, std = real.mean(0), real.std(0)

    def combine(raw: np.ndarray, weight: float) -> np.ndarray:
        z = (raw - mean) / std
        return weight * z[:, 0] + (1 - weight) * z[:, 1]

    weight = max((0.3, 0.5, 0.7), key=lambda w: roc_auc_score(y_val, combine(raw_val, w)))
    threshold = threshold_at_fpr(y_val, combine(raw_val, weight), 0.01)
    config = {
        "weight": weight,
        "spam_mean": float(mean[0]),
        "spam_std": float(std[0]),
        "cnn_mean": float(mean[1]),
        "cnn_std": float(std[1]),
        "threshold": threshold,
    }
    Path("models/ensemble.json").write_text(json.dumps(config, indent=2) + "\n")
    report: dict = {"ensemble": config, "test": {}}
    single = {
        name: threshold_at_fpr(y_val, raw_val[:, k], 0.01) for k, name in enumerate(["spam", "cnn"])
    }
    for quality in (None, 90, 75):
        y, raw = raw_scores(TEST, quality)
        combined = combine(raw, weight)
        report["test"][f"jpeg{quality}" if quality else "original"] = {
            "n": len(y),
            "ensemble": {
                "auroc": float(roc_auc_score(y, combined)),
                **operating_point(y, combined, threshold),
            },
            **{
                name: {
                    "auroc": float(roc_auc_score(y, raw[:, k])),
                    **operating_point(y, raw[:, k], single[name]),
                }
                for k, name in enumerate(["spam", "cnn"])
            },
        }
        print(quality or "original", report["test"], flush=True)
    Path("reports/ensemble.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
