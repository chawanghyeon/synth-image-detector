import json

import numpy as np
import onnx
import pytest
from conftest import png_bytes
from fastapi.testclient import TestClient
from onnx import TensorProto, helper
from PIL import Image

from detector.model_card import ModelCard
from detector.preprocess import CenterCrop, model_input, to_array
from detector.serving.detector import Detector


@pytest.fixture
def models(tmp_path):
    """Tiny stand-in with the student's interface: images (n, 3, 336, 336) -> logits (n)."""
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["images"], ["logits"], axes=[1, 2, 3], keepdims=0)],
        "stand_in",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["n", 3, 336, 336])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, tmp_path / "student.onnx")
    ModelCard(size=336, threshold=0.0).save(tmp_path)
    return tmp_path


def test_center_crop_is_native_resolution_then_resized():
    image = Image.new("RGB", (1200, 700), (255, 0, 0))
    [crop] = CenterCrop()(image)
    assert crop.size == (512, 512)
    x = model_input(image, 336)
    assert x.shape == (1, 3, 336, 336) and x.dtype == np.float32
    assert np.allclose(x[0, 0], 1) and np.allclose(x[0, 1], -1)
    assert CenterCrop()(Image.new("RGB", (300, 401)))[0].size == (300, 300)
    with pytest.raises(ValueError):
        CenterCrop()(Image.new("RGB", (200, 500)))


def test_to_array_range():
    x = to_array(Image.new("RGB", (40, 30), (0, 128, 255)), 16)
    assert x.shape == (3, 16, 16)
    assert np.allclose(x[0], -1) and np.allclose(x[2], 1)


def test_model_card_round_trip(tmp_path):
    card = ModelCard(336, 2.07, "arch", "student_v4.pt", "reddit3")
    card.save(tmp_path)
    assert ModelCard.load(tmp_path) == card
    assert json.loads((tmp_path / "student.json").read_text())["threshold"] == 2.07


def test_detector_score_is_sigmoid_of_margin(models):
    detector = Detector(models)
    image = Image.new("RGB", (400, 300), (255, 255, 255))
    assert detector.logit(image) == pytest.approx(1.0, abs=1e-4)
    assert detector.score(image) == pytest.approx(1 / (1 + np.exp(-1)), abs=1e-4)


def test_predict(models, monkeypatch):
    import detector.serving.app as api

    monkeypatch.setattr(api, "MODELS", str(models))
    with TestClient(api.app) as client:
        response = client.post("/predict", files={"file": ("x.png", png_bytes((300, 260)))})
        assert response.status_code == 200
        assert 0 <= response.json()["ai_score"] <= 1
        assert client.post("/predict", files={"file": ("x", b"junk")}).status_code == 422
        tiny = client.post("/predict", files={"file": ("x.png", png_bytes((64, 64)))})
        assert tiny.status_code == 422


def test_missing_model_fails_startup(models, monkeypatch):
    import detector.serving.app as api

    (models / "student.json").unlink()
    monkeypatch.setattr(api, "MODELS", str(models))
    with pytest.raises(FileNotFoundError), TestClient(api.app):
        pass
