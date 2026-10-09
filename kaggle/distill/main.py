"""Kaggle GPU job: distil the PE-Core-G probe (teacher, 1.9B) into a fine-tuned PE-Core-L (student,
0.3B, about 0.44 s per image on a laptop CPU).

Data: the same 15,000 training and 6,000 test crops as experiment 20 (factory.from_datasets with the
same subset). teacher.npz holds, per training crop, the teacher's logit on the CLEAN original crop
(cross-fitted: 5 folds, so no crop is scored by a probe that saw it) and the 20% held-aside reals
used only for the threshold.

Two students train at the same time, one per GPU, identical except for the target:
  distill  loss = 0.5 * BCE(student(degraded), label)
                + 0.5 * BCE(student(degraded), sigmoid(teacher(clean)))
  labels   loss = BCE(student(degraded), label)        (control: does the teacher add anything?)
The student sees a degraded copy 75% of the time (random JPEG 50-95 or SNS chain), the original 25%.
Its last 8 of 24 blocks, the attention pooling and a linear head are trained.
Outputs in /kaggle/working: scores_<student>.npz (test logits per condition, held-aside real
logits) and student_<student>.pt (trained weights).
"""

import io
import random
import threading
import time

import numpy as np

STEPS = 1500  # x 32 images = 4 epochs over the ~12,000 training crops (about 1 h on a T4)
SMOKE = False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def augment(image, rng):
    from PIL import Image

    def jpeg(im, q):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q)
        return Image.open(buf).convert("RGB")

    r = rng.random()
    if r < 0.25:
        return image
    if r < 0.6:
        return jpeg(image, rng.randint(50, 95))
    s = rng.uniform(0.5, 1.0)
    image = image.resize(
        (max(64, int(image.width * s)), max(64, int(image.height * s))), Image.BICUBIC
    )
    return jpeg(image, rng.randint(60, 90))


def to_tensor(image, size):
    import torch
    from PIL import Image

    a = np.asarray(image.resize((size, size), Image.BICUBIC), np.float32) / 127.5 - 1
    return torch.from_numpy(a.transpose(2, 0, 1))


def student(gpu):
    import timm
    import torch

    model = timm.create_model("vit_pe_core_large_patch14_336.fb", pretrained=True, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    trainable = list(model.blocks[-8:].parameters())
    for name in ("attn_pool", "norm"):
        if hasattr(model, name) and isinstance(getattr(model, name), torch.nn.Module):
            trainable += list(getattr(model, name).parameters())
    for p in trainable:
        p.requires_grad = True
    head = torch.nn.Linear(model.num_features, 1)
    return model.to(f"cuda:{gpu}"), head.to(f"cuda:{gpu}"), trainable


def train(kind, gpu, model, head, trainable, files, labels, teacher, results):
    import torch
    import torch.nn.functional as F
    from PIL import Image

    device = f"cuda:{gpu}"
    opt = torch.optim.AdamW(
        [{"params": trainable, "lr": 2e-5}, {"params": head.parameters(), "lr": 1e-3}],
        weight_decay=0.05,
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [2e-5, 1e-3], total_steps=STEPS, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    rng = random.Random(gpu)
    soft = torch.sigmoid(torch.from_numpy(teacher))
    y = torch.from_numpy(labels.astype(np.float32))
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    from concurrent.futures import ThreadPoolExecutor

    def load(i):
        return to_tensor(augment(Image.open(files[i]).convert("RGB"), rng), 336)

    model.train()
    begin = time.time()
    with ThreadPoolExecutor(3) as pool:
        for step in range(STEPS):
            idx = np.r_[rng.sample(list(pos), 16), rng.sample(list(neg), 16)]  # balanced batch
            x = torch.stack(list(pool.map(load, idx))).to(device)
            with torch.autocast("cuda", torch.float16):
                logit = head(model(x)).squeeze(1).float()
            target = y[idx].to(device)
            loss = F.binary_cross_entropy_with_logits(logit, target)
            if kind == "distill":
                loss = 0.5 * loss + 0.5 * F.binary_cross_entropy_with_logits(
                    logit, soft[idx].to(device)
                )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 200 == 0:
                log(f"{kind} step {step} loss {loss.item():.4f} {time.time() - begin:.0f}s")
    results[kind] = (model.eval(), head.eval())


def score(model, head, gpu, files, cond):
    import factory
    import torch
    from PIL import Image

    out = []
    with torch.inference_mode():
        for a in range(0, len(files), 64):
            x = torch.stack(
                [
                    to_tensor(factory.degrade(Image.open(f).convert("RGB"), cond, None), 336)
                    for f in files[a : a + 64]
                ]
            )
            with torch.autocast("cuda", torch.float16):
                out.append(head(model(x.to(f"cuda:{gpu}"))).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


def main():
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "timm"], check=False)
    import factory

    meta = factory.from_datasets(keep=factory.COMPARE_KEEP)
    train_rows = [m for m in meta if m["set"] == "train"]
    test_rows = [m for m in meta if m["set"] == "test"]
    t = np.load(JOB / "teacher.npz")  # noqa: F821  (JOB is defined by the push header)
    assert len(t["logit"]) == len(train_rows), "teacher scores do not match the training subset"
    assert (t["label"] == np.array([int(m["label"]) for m in train_rows])).all()
    fit = ~t["cal"]
    files = [m["file"] for m, f in zip(train_rows, fit) if f]
    labels, teacher = t["label"][fit], np.clip(t["logit"][fit], -15, 15).astype(np.float32)
    global STEPS
    if SMOKE:
        STEPS = 30
        test_rows = test_rows[:200]
    log(f"train {len(files)} crops, test {len(test_rows)}")
    students = {
        "distill": (0, *student(0)),
        "labels": (1, *student(1)),
    }  # load before threads start
    results = {}
    threads = [
        threading.Thread(target=train, args=(k, g, m, h, p, files, labels, teacher, results))
        for k, (g, m, h, p) in students.items()
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    cal_files = [m["file"] for m, c in zip(train_rows, t["cal"]) if c]

    def evaluate(kind):
        model, head = results[kind]
        gpu = students[kind][0]
        out = {"cal_original": score(model, head, gpu, cal_files, "original")}
        for cond in ("original", "jpeg75", "sns"):
            out[cond] = score(model, head, gpu, [m["file"] for m in test_rows], cond)
            log(f"{kind} scored test/{cond}")
        np.savez(
            f"/kaggle/working/scores_{kind}.npz",
            **out,
            label=np.array([int(m["label"]) for m in test_rows]),
            source=np.array([m["source"] for m in test_rows]),
            model=np.array([m["model"] for m in test_rows]),
        )
        torch_save(model, head, kind)

    threads = [threading.Thread(target=evaluate, args=(k,)) for k in results]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    log("done")


def torch_save(model, head, kind):
    """Trained weights of the student (blocks, pooling, head) for later export."""
    import torch

    state = {
        k: v.half().cpu()
        for k, v in model.state_dict().items()
        if k.startswith(
            ("blocks.16", "blocks.17", "blocks.18", "blocks.19", "blocks.2", "attn_pool", "norm")
        )
    }
    torch.save({"model": state, "head": head.state_dict()}, f"/kaggle/working/student_{kind}.pt")


if __name__ == "__main__":
    main()
