"""Student checkpoint -> models/student.onnx + models/student.json for the server."""

from pathlib import Path

import numpy as np
import torch

from detector.model_card import WEIGHTS, ModelCard
from detector.training.model import Student


def export_onnx(student: Student, card: ModelCard, folder: Path = Path("models")) -> float:
    """Write the ONNX graph ((n, 3, size, size) "images" -> (n,) "logits") and the card; returns
    the largest |onnx - torch| difference on random input."""
    import onnxruntime as ort

    folder.mkdir(parents=True, exist_ok=True)
    student = student.float().eval()
    x = torch.rand(2, 3, card.size, card.size) * 2 - 1
    torch.onnx.export(
        student,
        x,
        str(folder / WEIGHTS),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "n"}, "logits": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    with torch.inference_mode():
        expected = student(x).numpy()
    got = ort.InferenceSession(str(folder / WEIGHTS)).run(None, {"images": x.numpy()})[0]
    card.save(folder)
    return float(np.abs(got - expected).max())
