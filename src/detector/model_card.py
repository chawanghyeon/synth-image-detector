"""models/student.json: what the server needs to know about the exported model."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CARD = "student.json"
WEIGHTS = "student.onnx"


@dataclass(frozen=True)
class ModelCard:
    size: int  # input side in pixels
    threshold: float  # logit at 1% FPR on `threshold_reals`; ai_score = sigmoid(logit - threshold)
    arch: str = ""
    weights: str = ""  # training checkpoint the ONNX file was exported from
    threshold_reals: str = ""  # "reddit3" (served) or "cal" (held-aside training reals)

    @classmethod
    def load(cls, folder: str | Path) -> "ModelCard":
        return cls(**json.loads((Path(folder) / CARD).read_text()))

    def save(self, folder: str | Path) -> None:
        (Path(folder) / CARD).write_text(json.dumps(asdict(self), indent=2) + "\n")
