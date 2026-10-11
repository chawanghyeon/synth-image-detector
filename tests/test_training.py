"""Training-side contracts with a tiny stand-in backbone (no pretrained download)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from PIL import Image  # noqa: E402

from detector.model_card import ModelCard  # noqa: E402
from detector.serving.detector import Detector  # noqa: E402
from detector.training.export import export_onnx  # noqa: E402
from detector.training.model import Student  # noqa: E402
from detector.training.recipe import Recipe  # noqa: E402
from detector.training.trainer import Trainer, score_files  # noqa: E402


class TinyBackbone(torch.nn.Module):
    num_features = 4

    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Conv2d(3, 4, 3)
        self.tuned = torch.nn.Linear(4, 4)
        self.frozen.requires_grad_(False)

    def forward(self, x):
        return self.tuned(self.frozen(x).mean((2, 3)))


def tiny_student():
    return Student(TinyBackbone(), torch.nn.Linear(4, 1))


def test_fit_updates_only_trainable_params_and_scores(crops):
    files, labels = crops
    student = tiny_student()
    frozen = student.backbone.frozen.weight.clone()
    tuned = student.backbone.tuned.weight.clone()
    recipe = Recipe("tiny", size=32, steps=3, batch=4)
    Trainer(student, recipe, "cpu", log=lambda _: None).fit(files, np.array(labels))
    assert torch.equal(student.backbone.frozen.weight, frozen)
    assert not torch.equal(student.backbone.tuned.weight, tuned)
    logits = score_files(student, files, "sns", 32, "cpu", batch=3)
    assert logits.shape == (8,) and np.isfinite(logits).all()


def test_checkpoint_holds_only_trained_weights(tmp_path):
    student = tiny_student()
    student.save(tmp_path / "student.pt")
    saved = torch.load(tmp_path / "student.pt")
    assert set(saved["model"]) == {"tuned.weight", "tuned.bias"}
    assert saved["model"]["tuned.weight"].dtype == torch.float16
    assert set(saved["head"]) == {"weight", "bias"}


def test_export_matches_what_the_server_loads(tmp_path):
    student = tiny_student().eval()
    card = ModelCard(size=32, threshold=0.25)
    assert export_onnx(student, card, tmp_path) < 1e-4
    detector = Detector(tmp_path)
    image = Image.effect_noise((400, 300), 40).convert("RGB")
    from detector.preprocess import model_input

    with torch.inference_mode():
        expected = student(torch.from_numpy(model_input(image, 32))).item()
    assert detector.logit(image) == pytest.approx(expected, abs=1e-4)
    assert detector.card == card
