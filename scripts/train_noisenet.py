"""Train the noise-residual CNN, export it to ONNX and report held-out results.

Usage: python scripts/train_noisenet.py [steps]   (default 12000, about 1.2 h on an M3 Pro)
Same data and splits as train_spam.py. Writes models/noisenet.onnx and reports/noisenet.json.
"""

import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from train_spam import (
    CONDITIONS,
    ROOT,
    TRAIN_SOURCES,
    degrade,
    jpeg,
    operating_point,
    threshold_at_fpr,
)

from detector.features import PATCH, patches

WEIGHTS = Path("models/noisenet.pt")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def residuals(x: torch.Tensor) -> torch.Tensor:
    """x: (N, 3, H, W) in [0, 1], H and W even. Returns (N, 6, H, W) noise residuals."""
    # Local smoothness residual: what a 3x3 neighbourhood cannot explain.
    blur = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    # 2x up-sampling residual (NPR): decoders that up-sample leave correlated 2x2 blocks.
    coarse = F.interpolate(F.avg_pool2d(x, 2), scale_factor=2, mode="nearest")
    return torch.cat([x - blur, x - coarse], dim=1) * 20


def _block(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class NoiseNet(nn.Module):
    """Content is removed first (residuals), so the network only sees pixel-level traces."""

    def __init__(self) -> None:
        super().__init__()
        # No early down-sampling: the traces live in the highest frequencies.
        self.features = nn.Sequential(
            _block(6, 32, 1),
            _block(32, 64, 2),
            _block(64, 128, 2),
            _block(128, 128, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, 128, 128) in [0, 1]. Returns one logit per patch."""
        return self.head(self.features(residuals(x))).squeeze(1)


class TrainSet(Dataset):
    """Identical random degradations for both classes, so file format cannot be the shortcut."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(ROOT / row["file"]).convert("RGB")
        if random.random() < 0.3:
            scale = random.uniform(0.5, 1.5)
            size = [max(PATCH, int(side * scale)) for side in image.size]
            image = image.resize(size, random.choice([Image.BILINEAR, Image.BICUBIC]))
        if random.random() < 0.8:
            image = jpeg(image, random.randint(60, 100))
        array = np.asarray(image, dtype=np.float32) / 255
        top = random.randint(0, array.shape[0] - PATCH)
        left = random.randint(0, array.shape[1] - PATCH)
        patch = torch.from_numpy(array[top : top + PATCH, left : left + PATCH].copy())
        patch = patch.permute(2, 0, 1)
        if random.random() < 0.5:
            patch = patch.flip(2)
        return patch, float(row["label"])


class EvalSet(Dataset):
    def __init__(self, rows: list[dict], condition: tuple | None) -> None:
        self.rows = rows
        self.condition = condition

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        image = Image.open(ROOT / self.rows[index]["file"]).convert("RGB")
        return torch.from_numpy(patches(degrade(image, self.condition), limit=16))


def first(batch: list) -> torch.Tensor:
    return batch[0]


@torch.inference_mode()
def scores(model: NoiseNet, rows: list[dict], condition: tuple | None) -> np.ndarray:
    model.eval()
    loader = DataLoader(EvalSet(rows, condition), batch_size=1, num_workers=4, collate_fn=first)
    out = [model(batch.to(DEVICE)).mean().item() for batch in loader]
    model.train()
    return np.array(out)


def main(steps: int) -> None:
    split = defaultdict(list)
    for row in csv.DictReader((ROOT / "index.csv").open()):
        split[row["split"]].append(row)
        if row["split"] == "extra":
            split[row["model"]].append(row)
    train = [row for source in TRAIN_SOURCES for row in split[source]]
    val, test = split["validation"], split["test"]
    labels = np.array([int(r["label"]) for r in train])
    weights = np.where(labels == 1, 1 / labels.sum(), 1 / (len(labels) - labels.sum()))
    loader = DataLoader(
        TrainSet(train),
        batch_size=128,
        sampler=WeightedRandomSampler(weights, steps * 128),
        num_workers=10,
        drop_last=True,
        persistent_workers=True,
    )
    model = NoiseNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, 2e-3, total_steps=steps)
    y_val = np.array([int(r["label"]) for r in val])
    best, begin = -1.0, time.time()
    WEIGHTS.parent.mkdir(exist_ok=True)
    for step, (x, y) in enumerate(loader, 1):
        loss = F.binary_cross_entropy_with_logits(model(x.to(DEVICE)), y.float().to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} {time.time() - begin:.0f}s", flush=True)
        if step % 1000 == 0 or step == steps:
            auroc = roc_auc_score(y_val, scores(model, val, None))
            print(f"step {step} validation AUROC {auroc:.4f}", flush=True)
            if auroc > best:
                best = auroc
                torch.save(model.state_dict(), WEIGHTS)
    model.load_state_dict(torch.load(WEIGHTS, weights_only=True))
    val_scores = scores(model, val, None)
    thresholds = {f"fpr_{p}": threshold_at_fpr(y_val, val_scores, p) for p in (0.01, 0.05)}
    y_test = np.array([int(r["label"]) for r in test])
    report: dict = {
        "train_n": len(train),
        "steps": steps,
        "validation_auroc": float(roc_auc_score(y_val, val_scores)),
        "thresholds_from_validation": thresholds,
        "test": {},
    }
    for name, condition in CONDITIONS.items():
        values = scores(model, test, condition)
        report["test"][name] = {
            "auroc": float(roc_auc_score(y_test, values)),
            **{k: operating_point(y_test, values, t) for k, t in thresholds.items()},
        }
        print(name, report["test"][name], flush=True)
    Path("reports/noisenet.json").write_text(json.dumps(report, indent=2) + "\n")
    export(model)


def export(model: NoiseNet) -> None:
    model.cpu().eval()
    torch.onnx.export(
        model,
        torch.rand(2, 3, PATCH, PATCH),
        "models/noisenet.onnx",
        input_names=["patches"],
        output_names=["logits"],
        dynamic_axes={"patches": {0: "n"}, "logits": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12000)
