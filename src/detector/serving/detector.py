"""The exported student (ONNX) behind the API.

The student was trained on native-resolution crops (at most 512 px, never upscaled) resized to
336 px, including JPEG and SNS-style copies, so it keeps working after compression and resizing.
"""

import math
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from detector.model_card import WEIGHTS, ModelCard
from detector.preprocess import CenterCrop, CropPolicy, model_input


class Detector:
    def __init__(self, folder: str | Path = "models", policy: CropPolicy | None = None) -> None:
        folder = Path(folder)
        self.card = ModelCard.load(folder)  # written by `detector export`
        self.policy = policy or CenterCrop()
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("ORT_THREADS", "0"))  # 0 = all cores
        self.session = ort.InferenceSession(
            str(folder / WEIGHTS), options, providers=["CPUExecutionProvider"]
        )

    def logits(self, image: Image.Image) -> np.ndarray:
        """One logit per crop the policy picks."""
        x = model_input(image, self.card.size, self.policy)
        return self.session.run(None, {"images": x})[0]

    def logit(self, image: Image.Image) -> float:
        return float(self.logits(image).mean())

    def score(self, image: Image.Image) -> float:
        """0.5 = the threshold (1% false positives on the card's reals). Not a probability."""
        return 1 / (1 + math.exp(-(self.logit(image) - self.card.threshold)))
