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
from sklearn.linear_model import LogisticRegression

from detector.features import crops, patches, semantic_crops, spam_features


def png(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.effect_noise(size, 40).convert("RGB").save(buffer, "PNG")
    return buffer.getvalue()


def save_onnx(nodes, inp, out, path, initializers=()):
    graph = helper.make_graph(
        nodes,
        "stand_in",
        [helper.make_tensor_value_info(inp[0], TensorProto.FLOAT, inp[1])],
        [helper.make_tensor_value_info(out[0], TensorProto.FLOAT, out[1])],
        initializer=list(initializers),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


@pytest.fixture
def models(tmp_path):
    """Tiny stand-ins with the real interfaces of all three detectors."""
    rng = np.random.default_rng(0)
    spam = HistGradientBoostingClassifier(max_iter=5).fit(rng.random((20, 1372)), [0, 1] * 10)
    joblib.dump(spam, tmp_path / "spam_hgb.joblib")
    save_onnx(
        [helper.make_node("ReduceMean", ["patches"], ["logits"], axes=[1, 2, 3], keepdims=0)],
        ("patches", ["n", 3, 128, 128]),
        ("logits", ["n"]),
        tmp_path / "noisenet.onnx",
    )
    # semantic stand-in: crops (n, 3, 224, 224) -> features (n, 3 depths, 1)
    reshape = helper.make_tensor("shape", TensorProto.INT64, [3], [-1, 3, 1])
    save_onnx(
        [
            helper.make_node("ReduceMean", ["crops"], ["pooled"], axes=[2, 3], keepdims=0),
            helper.make_node("Reshape", ["pooled", "shape"], ["features"]),
        ],
        ("crops", ["n", 3, 224, 224]),
        ("features", ["n", 3, 1]),
        tmp_path / "semantic.onnx",
        initializers=[reshape],
    )
    lr = LogisticRegression().fit(rng.random((20, 3)), [0, 1] * 10)
    joblib.dump(lr, tmp_path / "semantic_lr.joblib")
    config = dict(
        weight=0.5,
        spam_mean=0,
        spam_std=1,
        cnn_mean=0,
        cnn_std=1,
        noise_mean=0,
        noise_std=1,
        semantic_mean=0,
        semantic_std=1,
        threshold=0,
    )
    (tmp_path / "ensemble.json").write_text(json.dumps(config))
    return tmp_path


def test_inputs_are_never_resized():
    assert [c.size for c in crops(Image.new("RGB", (1200, 700)))] == [(512, 512)] * 2
    assert [c.size for c in crops(Image.new("RGB", (300, 400)))] == [(300, 300)]
    with pytest.raises(ValueError):
        crops(Image.new("RGB", (200, 500)))
    assert patches(Image.new("RGB", (1000, 300))).shape == (14, 3, 128, 128)
    assert semantic_crops(Image.new("RGB", (1000, 300))).shape == (2, 3, 224, 224)


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
