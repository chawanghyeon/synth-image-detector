"""Export a PE-Core-L student (kaggle/distill or kaggle/train) to ONNX for the server.

Usage: python scripts/export_student.py WEIGHTS.pt SCORES.npz [THRESHOLD_SET]
  e.g. data/exp27/student_v3r.pt data/exp27/scores_v3r.npz reddit3
The saved weights (last 8 blocks, attention pooling, norm, head; fp16) go on top of the public
PE-Core-L14-336 weights. The ONNX graph takes (N, 3, 336, 336) images scaled to [-1, 1] and returns
one logit per image. models/student.json records the threshold at 1% FPR so that
ai_score = sigmoid(logit - threshold) puts that boundary at 0.5. Its reals are the held-aside
training reals ("cal_original"), or, with THRESHOLD_SET, the real images of that evaluation set
(original files). The served model uses reddit3: images posted online, never trained on
(experiment 27).
Needs torch and timm (extra "train").
"""

import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch

WEIGHTS = Path(sys.argv[1])
SCORES = Path(sys.argv[2])
THRESHOLD_SET = sys.argv[3] if len(sys.argv) > 3 else None
ARCH = "vit_pe_core_large_patch14_336.fb"
SIZE = 336


class Student(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(ARCH, pretrained=True, num_classes=0)
        saved = torch.load(WEIGHTS, map_location="cpu")
        _, unexpected = self.backbone.load_state_dict(
            {k: v.float() for k, v in saved["model"].items()}, strict=False
        )
        assert not unexpected, unexpected
        self.head = torch.nn.Linear(self.backbone.num_features, 1)
        self.head.load_state_dict({k: v.float() for k, v in saved["head"].items()})

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images)).squeeze(1)


def main() -> None:
    model = Student().eval()
    x = torch.rand(2, 3, SIZE, SIZE) * 2 - 1
    Path("models").mkdir(exist_ok=True)
    torch.onnx.export(
        model,
        x,
        "models/student.onnx",
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "n"}, "logits": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort

    with torch.inference_mode():
        expected = model(x).numpy()
    got = ort.InferenceSession("models/student.onnx").run(None, {"images": x.numpy()})[0]
    print("max |onnx - torch|:", float(np.abs(got - expected).max()))
    scores = np.load(SCORES)
    if THRESHOLD_SET:
        reals = scores["original"][(scores["set"] == THRESHOLD_SET) & (scores["label"] == 0)]
    else:
        reals = scores["cal_original"]
    threshold = float(np.quantile(reals, 0.99))
    config = {"arch": ARCH, "size": SIZE, "threshold": threshold, "weights": WEIGHTS.name}
    config["threshold_reals"] = THRESHOLD_SET or "held-aside training reals"
    Path("models/student.json").write_text(json.dumps(config, indent=2) + "\n")
    print("threshold", threshold)


if __name__ == "__main__":
    main()
