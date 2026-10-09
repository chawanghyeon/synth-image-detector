"""Export a distilled PE-Core-L student (kaggle/distill) to ONNX for the server.

Usage: python scripts/export_student.py [student_distill.pt] [scores_distill.npz]
  (defaults: the files of the last kaggle/distill run in data/distill/)
The saved weights (last 8 blocks, attention pooling, norm, head; fp16) go on top of the public
PE-Core-L14-336 weights. The ONNX graph takes (N, 3, 336, 336) images scaled to [-1, 1] and returns
one logit per image. models/student.json records the threshold (1% FPR on the held-aside training
reals) so that ai_score = sigmoid(logit - threshold) puts the 1%-FPR boundary at 0.5.
Needs torch and timm (extra "train").
"""

import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch

WEIGHTS = Path(sys.argv[1] if len(sys.argv) > 1 else "data/distill/student_distill.pt")
SCORES = Path(sys.argv[2] if len(sys.argv) > 2 else "data/distill/scores_distill.npz")
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
    threshold = float(np.quantile(np.load(SCORES)["cal_original"], 0.99))
    Path("models/student.json").write_text(
        json.dumps({"arch": ARCH, "size": SIZE, "threshold": threshold}, indent=2) + "\n"
    )
    print("threshold", threshold)


if __name__ == "__main__":
    main()
