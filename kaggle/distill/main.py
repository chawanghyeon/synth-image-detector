"""Kaggle GPU job: distil the PE-Core-G probe (teacher, 1.9B) into a fine-tuned PE-Core-L (student,
0.3B, about 0.44 s per image on a laptop CPU).

Data: the same 15,000 training and 6,000 test crops as experiment 20 (factory.from_datasets with the
same subset). teacher.npz holds, per training crop, the teacher's logit on the CLEAN original crop
(cross-fitted: 5 folds, so no crop is scored by a probe that saw it) and the 20% held-aside reals
used only for the threshold.

RUNS trains one model per GPU at the same time. Targets:
  distill  loss = 0.5 * BCE(student(degraded), label)
                + 0.5 * BCE(student(degraded), sigmoid(teacher(clean)))
  labels   loss = BCE(student(degraded), label)
The model sees a degraded copy 75% of the time (random JPEG 50-95 or SNS chain), the original 25%.
Its last `blocks` blocks ("all" = full fine-tuning), the attention pooling, the final norm and a
linear head are trained; the frozen part runs in fp16 so that PE-Core-G fits a 16 GB T4. `accum`
micro-batches are accumulated per optimizer step (PE-Core-G at 448 px: 8 images per micro-batch).
Experiment 23: PE-Core-L (8 blocks, 336 px), distill vs labels, 1,500 steps of 32.
Experiment 25: PE-Core-G (last 6 blocks, 448 px) vs PE-Core-L fully fine-tuned (336 px), labels,
test limited to 3,000.
Outputs in /kaggle/working: scores_<student>.npz (test logits per condition, held-aside real
logits) and student_<student>.pt (trained weights).
"""

import io
import random
import threading
import time

import numpy as np

PE_L, PE_G = "vit_pe_core_large_patch14_336.fb", "vit_pe_core_gigantic_patch14_448.fb"
RUNS = [  # one per GPU
    dict(name="peg448", arch=PE_G, size=448, blocks=6, kind="labels", steps=800, batch=8, accum=3),
    dict(
        name="pel_full",
        arch=PE_L,
        size=336,
        blocks="all",
        kind="labels",
        steps=1500,
        batch=32,
        accum=1,
    ),
]
TEST_LIMIT = 3000  # None = all 6,000 test crops
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


def build(run, gpu):
    import timm
    import torch

    model = timm.create_model(run["arch"], pretrained=True, num_classes=0, img_size=run["size"])
    model = model.half()
    model.set_grad_checkpointing(True)
    for p in model.parameters():
        p.requires_grad = False
    blocks = model.blocks if run["blocks"] == "all" else model.blocks[-run["blocks"] :]
    trainable = [blocks, model.attn_pool, model.norm]
    if run["blocks"] == "all":
        trainable += [model.patch_embed]
    for module in trainable:
        module.float()
        for p in module.parameters():
            p.requires_grad = True
    head = torch.nn.Linear(model.num_features, 1)
    return model.to(f"cuda:{gpu}"), head.to(f"cuda:{gpu}")


def train(run, gpu, model, head, files, labels, teacher, results):
    import torch
    import torch.nn.functional as F
    from PIL import Image

    device, steps, half, accum = f"cuda:{gpu}", run["steps"], run["batch"] // 2, run["accum"]
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": trainable, "lr": 2e-5}, {"params": head.parameters(), "lr": 1e-3}],
        weight_decay=0.05,
    )
    lr = 1e-5 if run["blocks"] == "all" else 2e-5
    for group, value in zip(opt.param_groups, (lr, 1e-3)):
        group["lr"] = value
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [lr, 1e-3], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    rng = random.Random(gpu)
    soft = torch.sigmoid(torch.from_numpy(teacher))
    y = torch.from_numpy(labels.astype(np.float32))
    pos, neg = list(np.flatnonzero(labels == 1)), list(np.flatnonzero(labels == 0))
    from concurrent.futures import ThreadPoolExecutor

    def load(i):
        return to_tensor(augment(Image.open(files[i]).convert("RGB"), rng), run["size"])

    model.train()
    begin = time.time()
    with ThreadPoolExecutor(3) as pool:
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            for _ in range(accum):
                idx = np.r_[rng.sample(pos, half), rng.sample(neg, half)]  # balanced batch
                x = torch.stack(list(pool.map(load, idx))).to(device)
                with torch.autocast("cuda", torch.float16):
                    logit = head(model(x)).squeeze(1).float()
                loss = F.binary_cross_entropy_with_logits(logit, y[idx].to(device))
                if run["kind"] == "distill":
                    loss = 0.5 * loss + 0.5 * F.binary_cross_entropy_with_logits(
                        logit, soft[idx].to(device)
                    )
                scaler.scale(loss / accum).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 100 == 0:
                log(f"{run['name']} step {step} loss {loss.item():.4f} {time.time() - begin:.0f}s")
    results[run["name"]] = (model.eval(), head.eval())


def score(model, head, gpu, files, cond, size):
    import factory
    import torch
    from PIL import Image

    out = []
    with torch.inference_mode():
        for a in range(0, len(files), 32):
            x = torch.stack(
                [
                    to_tensor(factory.degrade(Image.open(f).convert("RGB"), cond, None), size)
                    for f in files[a : a + 32]
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
    if TEST_LIMIT:
        test_rows = test_rows[:TEST_LIMIT]
    if SMOKE:
        for run in RUNS:
            run["steps"] = 20
        test_rows = test_rows[:100]
    log(f"train {len(files)} crops, test {len(test_rows)}")
    models = {
        run["name"]: (gpu, *build(run, gpu)) for gpu, run in enumerate(RUNS)
    }  # before threads
    results = {}
    threads = [
        threading.Thread(target=train, args=(run, g, m, h, files, labels, teacher, results))
        for run, (g, m, h) in zip(RUNS, models.values())
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    cal_files = [m["file"] for m, c in zip(train_rows, t["cal"]) if c]

    def evaluate(run):
        model, head = results[run["name"]]
        gpu, size = models[run["name"]][0], run["size"]
        out = {"cal_original": score(model, head, gpu, cal_files, "original", size)}
        for cond in ("original", "jpeg75", "sns"):
            out[cond] = score(model, head, gpu, [m["file"] for m in test_rows], cond, size)
            log(f"{run['name']} scored test/{cond}")
        np.savez(
            f"/kaggle/working/scores_{run['name']}.npz",
            **out,
            label=np.array([int(m["label"]) for m in test_rows]),
            source=np.array([m["source"] for m in test_rows]),
            model=np.array([m["model"] for m in test_rows]),
        )
        save(model, head, run["name"])

    threads = [
        threading.Thread(target=evaluate, args=(run,)) for run in RUNS if run["name"] in results
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    log("done")


def save(model, head, name):
    """Trained weights only (last blocks, pooling, norm, head), fp16, for later export."""
    import torch

    names = {n for n, p in model.named_parameters() if p.requires_grad}
    state = {k: v.half().cpu() for k, v in model.state_dict().items() if k in names}
    torch.save({"model": state, "head": head.state_dict()}, f"/kaggle/working/student_{name}.pt")


if __name__ == "__main__":
    main()
