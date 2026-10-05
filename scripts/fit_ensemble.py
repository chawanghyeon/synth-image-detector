"""Fit how the three detectors are combined, on full-size validation images scored like the server.

Usage: python scripts/fit_ensemble.py
Writes models/ensemble.json (normalisation, weight, threshold) and reports/ensemble.json.
Normalisation uses real validation photos only; the threshold is the 1% false-positive point.
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
from train_spam import threshold_at_fpr

from detector.detector import Detector

VALIDATION = "data/raw/core/validation-00000-of-00015.parquet"
CACHE = Path("data/features/server_validation.npz")
_detector = None


def _raw(data: bytes) -> tuple[float, float, float] | None:
    global _detector
    if _detector is None:
        os.environ["ORT_THREADS"] = "1"
        _detector = Detector(fitted=False)
    try:
        with Image.open(io.BytesIO(data)) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        return _detector.raw_scores(image)
    except (OSError, ValueError):
        return None  # too small or unreadable; the server answers 422 for these


def raw_scores() -> tuple[np.ndarray, np.ndarray]:
    """Labels and (spam, cnn, semantic) raw scores for every usable validation image."""
    if CACHE.exists():
        cached = np.load(CACHE)
        return cached["labels"], cached["scores"]
    labels, scores = [], []
    parquet = pq.ParquetFile(VALIDATION)
    with ProcessPoolExecutor(int(os.getenv("WORKERS", "4"))) as pool:
        for group in range(parquet.num_row_groups):
            rows = parquet.read_row_group(group, columns=["image", "label"]).to_pylist()
            for row, result in zip(rows, pool.map(_raw, [r["image"]["bytes"] for r in rows])):
                if result is not None:
                    labels.append(int(row["label"] == "fake"))
                    scores.append(result)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, labels=np.array(labels), scores=np.array(scores))
    return np.array(labels), np.array(scores)


def main() -> None:
    y, raw = raw_scores()
    real = raw[y == 0]
    mean, std = real.mean(0), real.std(0)
    z = (raw - mean) / std

    def noise(w: float) -> np.ndarray:
        return w * z[:, 0] + (1 - w) * z[:, 1]

    weight = max((0.3, 0.5, 0.7), key=lambda w: roc_auc_score(y, noise(w)))
    n = noise(weight)
    config = {
        "weight": weight,
        "spam_mean": float(mean[0]),
        "spam_std": float(std[0]),
        "cnn_mean": float(mean[1]),
        "cnn_std": float(std[1]),
        "noise_mean": float(n[y == 0].mean()),
        "noise_std": float(n[y == 0].std()),
        "semantic_mean": float(mean[2]),
        "semantic_std": float(std[2]),
    }
    combined = (n - config["noise_mean"]) / config["noise_std"] + z[:, 2]
    config["threshold"] = threshold_at_fpr(y, combined, 0.01)
    Path("models/ensemble.json").write_text(json.dumps(config, indent=2) + "\n")
    report = {
        "validation_n": int(len(y)),
        "validation_auroc": {
            "spam": float(roc_auc_score(y, raw[:, 0])),
            "cnn": float(roc_auc_score(y, raw[:, 1])),
            "semantic": float(roc_auc_score(y, raw[:, 2])),
            "noise_pair": float(roc_auc_score(y, n)),
            "combined": float(roc_auc_score(y, combined)),
        },
        "validation_tpr_at_threshold": float((combined[y == 1] > config["threshold"]).mean()),
        "ensemble": config,
    }
    Path("reports/ensemble.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["validation_auroc"]), flush=True)


if __name__ == "__main__":
    main()
