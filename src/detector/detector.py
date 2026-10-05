"""Three detectors combined, no torch at serve time.

- SPAM + boosting: pixel-difference noise statistics (strong on original files).
- NoiseNet (ONNX): small CNN on noise residuals.
- Semantic (ONNX): frozen DINOv3-B features + logistic regression (survives compression/resizing).
"""

import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from PIL import Image

from detector.features import crops, patches, semantic_crops, spam_features


class Detector:
    def __init__(self, folder: str | Path = "models", fitted: bool = True) -> None:
        folder = Path(folder)
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("ORT_THREADS", "0"))  # 0 = all cores

        def session(name: str) -> ort.InferenceSession:
            return ort.InferenceSession(
                str(folder / name), options, providers=["CPUExecutionProvider"]
            )

        self.spam = joblib.load(folder / "spam_hgb.joblib")
        self.cnn = session("noisenet.onnx")
        self.semantic = session("semantic.onnx")
        self.semantic_lr = joblib.load(folder / "semantic_lr.joblib")
        # Normalisation, weight and threshold, written by scripts/fit_ensemble.py.
        self.ensemble = json.loads((folder / "ensemble.json").read_text()) if fitted else {}

    def raw_scores(self, image: Image.Image) -> tuple[float, float, float]:
        """SPAM log-odds, CNN logit and semantic log-odds, each averaged over its crops."""
        spam = self.spam.decision_function(np.stack([spam_features(c) for c in crops(image)]))
        cnn = self.cnn.run(None, {"patches": patches(image)})[0]
        features = self.semantic.run(None, {"crops": semantic_crops(image)})[0].mean(0)
        semantic = self.semantic_lr.decision_function(features.reshape(1, -1))
        return float(spam.mean()), float(cnn.mean()), float(semantic[0])

    def combine(self, spam: float, cnn: float, semantic: float) -> float:
        """Noise pair and semantic score, each normalised on real validation photos, then added."""
        e = self.ensemble
        noise = e["weight"] * (spam - e["spam_mean"]) / e["spam_std"]
        noise += (1 - e["weight"]) * (cnn - e["cnn_mean"]) / e["cnn_std"]
        return (noise - e["noise_mean"]) / e["noise_std"] + (semantic - e["semantic_mean"]) / e[
            "semantic_std"
        ]

    def score(self, image: Image.Image) -> float:
        """0.5 = validation threshold at 1% false positives on real photos. Not a probability."""
        combined = self.combine(*self.raw_scores(image))
        return 1 / (1 + math.exp(-(combined - self.ensemble["threshold"])))
