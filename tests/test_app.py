import io
import json

import joblib
import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier

from detector.features import crops, patches, spam_features


def png(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.effect_noise(size, 40).convert("RGB").save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture
def models(tmp_path):
    """Tiny stand-ins with the real interfaces: boosting on 1372 features, ONNX patches->logits."""
    rng = np.random.default_rng(0)
    spam = HistGradientBoostingClassifier(max_iter=5).fit(rng.random((20, 1372)), [0, 1] * 10)
    joblib.dump(spam, tmp_path / "spam_hgb.joblib")
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["patches"], ["logits"], axes=[1, 2, 3], keepdims=0)],
        "mean",
        [helper.make_tensor_value_info("patches", TensorProto.FLOAT, ["n", 3, 128, 128])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, tmp_path / "noisenet.onnx")
    config = dict(weight=0.5, spam_mean=0, spam_std=1, cnn_mean=0, cnn_std=1, threshold=0)
    (tmp_path / "ensemble.json").write_text(json.dumps(config))
    return tmp_path


def test_inputs_are_never_resized():
    assert [c.size for c in crops(Image.new("RGB", (1200, 700)))] == [(512, 512)] * 2
    assert [c.size for c in crops(Image.new("RGB", (300, 400)))] == [(300, 300)]
    with pytest.raises(ValueError):
        crops(Image.new("RGB", (200, 500)))
    assert patches(Image.new("RGB", (1000, 300))).shape == (14, 3, 128, 128)


def test_spam_features_are_transition_probabilities():
    features = spam_features(Image.effect_noise((300, 300), 30).convert("RGB"))
    assert features.shape == (1372,)
    assert np.isfinite(features).all() and features.min() >= 0


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

    (models / "ensemble.json").unlink()
    monkeypatch.setattr(api, "MODELS", str(models))
    with pytest.raises(FileNotFoundError):
        with TestClient(api.app):
            pass
