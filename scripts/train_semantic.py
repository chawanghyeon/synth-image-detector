"""Train the semantic detector: frozen DINOv3-B features + logistic regression, export to ONNX.

Usage: python scripts/train_semantic.py   (about 10 min on an M3 Pro)
Features: CLS, patch mean and patch std at three depths of DINOv3 ViT-B/16, averaged over up to
four native-resolution 224px crops. The classifier also sees a JPEG/resized copy of every image,
so it learns cues that survive compression. Writes models/semantic.onnx and
models/semantic_lr.joblib.
"""

import csv
import io
import random
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from train_spam import ROOT

from detector.features import SEMANTIC, semantic_crops

REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # gated: accept the licence on Hugging Face
PREFIX = 5  # CLS + 4 register tokens
CACHE = Path("data/semantic")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class Features(nn.Module):
    """(N, 3, 224, 224) normalised crops -> (N, 3 depths, CLS | patch mean | patch std)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.model(pixel_values=x, output_hidden_states=True).hidden_states
        n = len(hidden) - 1
        out = []
        for layer in (n // 3, 2 * n // 3, n):
            tokens = hidden[layer]
            patches = tokens[:, PREFIX:]
            out.append(torch.cat([tokens[:, 0], patches.mean(1), patches.std(1)], 1))
        return torch.stack(out, 1)


def augment(image: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    if rng.random() < 0.3:
        s = rng.uniform(0.5, 1.0)
        size = (max(SEMANTIC, int(image.width * s)), max(SEMANTIC, int(image.height * s)))
        image = image.resize(size, Image.Resampling.BICUBIC)
    if rng.random() < 0.8:
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=rng.randint(50, 95))
        image = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return image


def _job(args: tuple[str, int | None]) -> np.ndarray:
    file, seed = args
    image = Image.open(ROOT / file).convert("RGB")
    return semantic_crops(augment(image, seed) if seed is not None else image)


def dev_rows() -> list[dict]:
    """4,000 training crops with real and generated images balanced per image-size bin."""
    rows = [
        r
        for r in csv.DictReader((ROOT / "index.csv").open())
        if r["split"] == "train" or r["model"] in ("coco", "flickr30k")
    ]
    y = np.array([int(r["label"]) for r in rows])
    pixels = np.array([int(r["width"]) * int(r["height"]) for r in rows])
    bins = np.floor(np.log2(pixels) / 0.5).astype(int)
    rng = np.random.default_rng(1)
    keep = []
    for b in np.unique(bins):
        real, fake = (np.flatnonzero((bins == b) & (y == c)) for c in (0, 1))
        k = min(len(real), len(fake))
        keep += list(rng.choice(real, k, replace=False)) + list(rng.choice(fake, k, replace=False))
    idx = np.array(sorted(keep))
    rng = np.random.default_rng(9)
    pick = np.r_[rng.permutation(idx[y[idx] == 0])[:2000], rng.permutation(idx[y[idx] == 1])[:2000]]
    return [rows[i] for i in pick]


@torch.inference_mode()
def features(net: Features, rows: list[dict], tag: str, augmented: bool) -> np.ndarray:
    path = CACHE / f"dinov3_b_dev_{tag}.npz"
    if path.exists():
        return np.load(path)["x"]
    seeds = [zlib.crc32(f"sem{r['file']}".encode()) if augmented else None for r in rows]
    out = []
    with ProcessPoolExecutor(4) as pool:
        for crops in pool.map(_job, [(r["file"], s) for r, s in zip(rows, seeds)], chunksize=8):
            out.append(net(torch.from_numpy(crops).to(DEVICE)).mean(0).cpu().numpy())
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=np.array(out))
    return np.array(out)


def main() -> None:
    from transformers import AutoModel

    net = Features(AutoModel.from_pretrained(REPO)).eval().to(DEVICE)
    rows = dev_rows()
    y = np.array([int(r["label"]) for r in rows])
    x = np.concatenate([features(net, rows, "orig", False), features(net, rows, "aug", True)])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=5000))
    clf.fit(x.reshape(len(x), -1), np.r_[y, y])
    joblib.dump(clf, "models/semantic_lr.joblib", compress=3)
    net.cpu()
    torch.onnx.export(
        net,
        torch.rand(2, 3, SEMANTIC, SEMANTIC),
        "models/semantic.onnx",
        input_names=["crops"],
        output_names=["features"],
        dynamic_axes={"crops": {0: "n"}, "features": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )


if __name__ == "__main__":
    main()
