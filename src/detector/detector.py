"""PE-Core-L student (ONNX), fine-tuned and distilled from a PE-Core-G probe.
No torch at serve time.

The student was trained on native-resolution crops (at most 512 px, never upscaled) resized to
336 px, including JPEG and SNS-style copies, so it keeps working after compression and resizing.
"""

import json
import math
import os
from pathlib import Path

import onnxruntime as ort
from PIL import Image

from detector.features import student_crop


class Detector:
    def __init__(self, folder: str | Path = "models") -> None:
        folder = Path(folder)
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("ORT_THREADS", "0"))  # 0 = all cores
        # Written by scripts/export_student.py: input size and the 1%-FPR threshold.
        self.config = json.loads((folder / "student.json").read_text())
        self.session = ort.InferenceSession(
            str(folder / "student.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def logit(self, image: Image.Image) -> float:
        crop = student_crop(image, self.config["size"])
        return float(self.session.run(None, {"images": crop})[0][0])

    def score(self, image: Image.Image) -> float:
        """0.5 = threshold at 1% false positives on held-aside real images. Not a probability."""
        return 1 / (1 + math.exp(-(self.logit(image) - self.config["threshold"])))
