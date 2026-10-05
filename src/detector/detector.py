"""Ensemble of two light noise detectors, no torch: SPAM + boosting, and a small CNN in ONNX."""

import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from PIL import Image

from detector.features import crops, patches, spam_features


class Detector:
    def __init__(self, folder: str | Path = "models", fitted: bool = True) -> None:
        folder = Path(folder)
        self.spam = joblib.load(folder / "spam_hgb.joblib")
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("ORT_THREADS", "0"))  # 0 = all cores
        self.cnn = ort.InferenceSession(
            str(folder / "noisenet.onnx"), options, providers=["CPUExecutionProvider"]
        )
        # Combination weights and threshold, written by scripts/fit_ensemble.py.
        self.ensemble = json.loads((folder / "ensemble.json").read_text()) if fitted else {}

    def raw_scores(self, image: Image.Image) -> tuple[float, float]:
        """Mean SPAM log-odds over 512px crops and mean CNN logit over 128px patches."""
        spam = self.spam.decision_function(np.stack([spam_features(c) for c in crops(image)]))
        cnn = self.cnn.run(None, {"patches": patches(image)})[0]
        return float(spam.mean()), float(cnn.mean())

    def score(self, image: Image.Image) -> float:
        """0.5 = validation threshold at 1% false positives on real photos. Not a probability."""
        e = self.ensemble
        spam, cnn = self.raw_scores(image)
        combined = (
            e["weight"] * (spam - e["spam_mean"]) / e["spam_std"]
            + (1 - e["weight"]) * (cnn - e["cnn_mean"]) / e["cnn_std"]
        )
        return 1 / (1 + math.exp(-(combined - e["threshold"])))
