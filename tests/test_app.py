import io
import json

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper
from PIL import Image

from detector.features import student_crop


def png(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.effect_noise(size, 40).convert("RGB").save(buffer, "PNG")
    return buffer.getvalue()


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
    (tmp_path / "student.json").write_text(json.dumps({"size": 336, "threshold": 0.0}))
    return tmp_path


def test_crop_is_native_resolution_then_resized():
    image = Image.new("RGB", (1200, 700), (255, 0, 0))
    x = student_crop(image, 336)
    assert x.shape == (1, 3, 336, 336) and x.dtype == np.float32
    assert np.allclose(x[0, 0], 1) and np.allclose(x[0, 1], -1)
    assert student_crop(Image.new("RGB", (300, 400)), 336).shape == (1, 3, 336, 336)
    with pytest.raises(ValueError):
        student_crop(Image.new("RGB", (200, 500)), 336)


def test_predict(models, monkeypatch):
    import detector.app as api

    monkeypatch.setattr(api, "MODELS", str(models))
    with TestClient(api.app) as client:
        response = client.post("/predict", files={"file": ("x.png", png((300, 260)))})
        assert response.status_code == 200
        assert 0 <= response.json()["ai_score"] <= 1
        assert client.post("/predict", files={"file": ("x", b"junk")}).status_code == 422
        tiny = client.post("/predict", files={"file": ("x.png", png((64, 64)))})
        assert tiny.status_code == 422


def test_missing_model_fails_startup(models, monkeypatch):
    import detector.app as api

    (models / "student.json").unlink()
    monkeypatch.setattr(api, "MODELS", str(models))
    with pytest.raises(FileNotFoundError):
        with TestClient(api.app):
            pass
