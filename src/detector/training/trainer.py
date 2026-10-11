"""Fine-tune a student with balanced batches of augmented crops, and score files with it."""

import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from detector.data.degrade import augment, degrade
from detector.preprocess import to_array
from detector.training.model import Student
from detector.training.recipe import Recipe


def _autocast(device: torch.device):
    return torch.autocast(device.type, torch.float16, enabled=device.type == "cuda")


class Trainer:
    """AdamW + one-cycle schedule, fp16 autocast on CUDA. Images are decoded and augmented in
    threads (no forked workers: Kaggle's filelocks break under fork)."""

    def __init__(
        self, student: Student, recipe: Recipe, device: str, log: Callable[[str], None] = print
    ) -> None:
        self.student, self.recipe = student.to(device), recipe
        self.device, self.log = torch.device(device), log

    def fit(self, files: Sequence[str], labels: np.ndarray, seed: int = 0) -> Student:
        r, student = self.recipe, self.student
        backbone = [p for p in student.backbone.parameters() if p.requires_grad]
        groups = [
            {"params": backbone, "lr": r.lr},
            {"params": student.head.parameters(), "lr": r.head_lr},
        ]
        opt = torch.optim.AdamW(groups, weight_decay=r.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, [r.lr, r.head_lr], total_steps=r.steps, pct_start=0.1
        )
        scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        rng = random.Random(seed)
        y = torch.from_numpy(labels.astype(np.float32))
        pos, neg = list(np.flatnonzero(labels == 1)), list(np.flatnonzero(labels == 0))

        def load(i: int) -> np.ndarray:
            return to_array(augment(Image.open(files[i]).convert("RGB"), rng), r.size)

        student.train()
        begin = time.time()
        with ThreadPoolExecutor(3) as pool:
            for step in range(r.steps):
                idx = np.r_[rng.sample(pos, r.batch // 2), rng.sample(neg, r.batch // 2)]
                x = torch.from_numpy(np.stack(list(pool.map(load, idx)))).to(self.device)
                opt.zero_grad(set_to_none=True)
                with _autocast(self.device):
                    logits = student(x).float()
                loss = F.binary_cross_entropy_with_logits(logits, y[idx].to(self.device))
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                if step % 100 == 0:
                    self.log(
                        f"{r.name} step {step} loss {loss.item():.4f} {time.time() - begin:.0f}s"
                    )
        return student.eval()


def score_files(
    student: Student, files: Sequence[str], condition: str, size: int, device: str, batch: int = 32
) -> np.ndarray:
    """Logits for files (crops) under a fixed degradation (detector.data.degrade.CONDITIONS)."""
    device_ = torch.device(device)
    out = []
    with torch.inference_mode():
        for a in range(0, len(files), batch):
            images = [
                degrade(Image.open(f).convert("RGB"), condition) for f in files[a : a + batch]
            ]
            x = torch.from_numpy(np.stack([to_array(im, size) for im in images])).to(device_)
            with _autocast(device_):
                out.append(student(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)
