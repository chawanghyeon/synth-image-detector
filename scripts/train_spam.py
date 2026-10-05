"""Train the SPAM + gradient-boosting detector and report held-out results.

Usage: python scripts/train_spam.py
Train: OpenFake train shards + extra real photos (COCO, Flickr30k), each with two augmented copies.
Threshold selection: OpenFake validation. Test: OpenFake test (unseen generators and real sources).
"""

import csv
import io
import json
import os
import random
import sys
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from detector.features import spam_features

ROOT = Path("data/crops")
CACHE = Path("data/features")
TRAIN_SOURCES = ["train", "coco", "flickr30k"]
WORKERS = int(os.getenv("WORKERS", "4"))
CONDITIONS = {
    "original": None,
    "jpeg95": ("jpeg", 95),
    "jpeg75": ("jpeg", 75),
    "jpeg50": ("jpeg", 50),
    "half_size": ("resize", 0.5),
}


def jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def degrade(image: Image.Image, condition: tuple | None) -> Image.Image:
    if condition is None:
        return image
    kind, value = condition
    if kind == "jpeg":
        return jpeg(image, value)
    return image.resize([int(side * value) for side in image.size], Image.Resampling.BICUBIC)


def augment(image: Image.Image, seed: int) -> Image.Image:
    """Same random resize/JPEG for both classes, so file history cannot be the shortcut."""
    rng = random.Random(seed)
    if rng.random() < 0.3:
        scale = rng.uniform(0.5, 1.5)
        size = [max(256, int(side * scale)) for side in image.size]
        image = image.resize(size, rng.choice([Image.BILINEAR, Image.BICUBIC]))
    if rng.random() < 0.8:
        image = jpeg(image, rng.randint(60, 100))
    return image


def _job(job: tuple[str, tuple | None, int | None]) -> np.ndarray:
    file, condition, seed = job
    image = Image.open(ROOT / file).convert("RGB")
    image = augment(image, seed) if seed is not None else degrade(image, condition)
    return spam_features(image)


def features(rows: list[dict], tag: str, condition=None, copy: int | None = None) -> np.ndarray:
    path = CACHE / f"spam_{tag}.npy"
    if path.exists():
        return np.load(path)
    seeds = [None if copy is None else zlib.crc32(f"{copy}{r['file']}".encode()) for r in rows]
    with ProcessPoolExecutor(WORKERS) as pool:
        array = np.stack(
            list(
                pool.map(
                    _job, [(r["file"], condition, s) for r, s in zip(rows, seeds)], chunksize=16
                )
            )
        )
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return array


def threshold_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr: float) -> float:
    real = np.sort(scores[labels == 0])[::-1]
    return float(real[min(int(fpr * len(real)), len(real) - 1)])


def operating_point(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    flagged = scores > threshold
    return {"fpr": float(flagged[labels == 0].mean()), "tpr": float(flagged[labels == 1].mean())}


def main() -> None:
    split = defaultdict(list)
    for row in csv.DictReader((ROOT / "index.csv").open()):
        split[row["split"]].append(row)
        if row["split"] == "extra":
            split[row["model"]].append(row)
    x, y = [], []
    for source in TRAIN_SOURCES:
        rows = split[source]
        tag = "train" if source == "train" else f"extra_{source}"
        for name, copy in (("orig", None), ("aug0", 0), ("aug1", 1)):
            x.append(features(rows, f"{tag}_{name}", copy=copy))
            y.append([int(r["label"]) for r in rows])
    x, y = np.concatenate(x), np.concatenate(y)
    model = HistGradientBoostingClassifier(
        max_iter=1500,
        learning_rate=0.05,
        max_leaf_nodes=63,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=0,
    ).fit(x, y)
    val, test = split["validation"], split["test"]
    y_val = np.array([int(r["label"]) for r in val])
    y_test = np.array([int(r["label"]) for r in test])
    val_scores = model.predict_proba(features(val, "validation_original"))[:, 1]
    thresholds = {f"fpr_{p}": threshold_at_fpr(y_val, val_scores, p) for p in (0.01, 0.05)}
    report: dict = {
        "train_n": int(len(y) // 3),
        "validation_auroc": float(roc_auc_score(y_val, val_scores)),
        "thresholds_from_validation": thresholds,
        "test": {},
    }
    for name, condition in CONDITIONS.items():
        scores = model.predict_proba(features(test, f"test_{name}", condition))[:, 1]
        report["test"][name] = {
            "auroc": float(roc_auc_score(y_test, scores)),
            **{k: operating_point(y_test, scores, t) for k, t in thresholds.items()},
        }
        if name == "original":
            by_source = defaultdict(list)
            for row, score in zip(test, scores):
                by_source[row["model"]].append(score > thresholds["fpr_0.01"])
            report["test_flag_rate_by_source_at_val_fpr_1pct"] = {
                k: [round(float(np.mean(v)), 3), len(v)]
                for k, v in sorted(by_source.items(), key=lambda kv: -np.mean(kv[1]))
            }
        print(name, report["test"][name], flush=True)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/spam_hgb.joblib", compress=3)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/spam_hgb.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
