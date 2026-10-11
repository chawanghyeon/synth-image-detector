"""Laptop benchmark: GPT-Image images made with Codex (never trained on) and real test photos,
each as the original and as an SNS copy (0.75x + JPEG 80), through the server's crop.

data/codex_test/*.png   50 AI images (prompts in data/codex_test/prompts.txt)
data/real_photos/*.png  100 real OpenFake test photos (short side >= 400 px, 512 px crops)
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from detector.data.degrade import degrade
from detector.preprocess import model_input

Logits = Callable[[np.ndarray], np.ndarray]  # (n, 3, size, size) -> (n,)


@dataclass(frozen=True)
class BenchResult:
    name: str
    ai: tuple[int, int, int]  # detected (original, sns), total
    real: tuple[int, int, int]  # flagged (original, sns), total

    def __str__(self) -> str:
        (a0, a1, n), (r0, r1, m) = self.ai, self.real
        return (
            f"{self.name}: Codex {a0}/{n} original, {a1}/{n} SNS; "
            f"real photos flagged {r0}/{m}, {r1}/{m}"
        )


class CodexBench:
    def __init__(
        self, ai: Path = Path("data/codex_test"), real: Path = Path("data/real_photos")
    ) -> None:
        self.ai = [Image.open(p).convert("RGB") for p in sorted(ai.glob("*.png"))]
        self.real = [Image.open(p).convert("RGB") for p in sorted(real.glob("*.png"))]

    def margins(self, logits: Logits, threshold: float, size: int) -> dict[str, np.ndarray]:
        """logit - threshold for {ai,real}_{original,sns}."""
        out = {}
        for kind, images in (("ai", self.ai), ("real", self.real)):
            for condition in ("original", "sns"):
                x = [logits(model_input(degrade(im, condition), size)).mean() for im in images]
                out[f"{kind}_{condition}"] = np.array(x) - threshold
        return out

    def run(self, name: str, logits: Logits, threshold: float, size: int) -> BenchResult:
        m = self.margins(logits, threshold, size)
        ai = (int((m["ai_original"] > 0).sum()), int((m["ai_sns"] > 0).sum()), len(self.ai))
        real = (int((m["real_original"] > 0).sum()), int((m["real_sns"] > 0).sum()), len(self.real))
        return BenchResult(name, ai, real)
