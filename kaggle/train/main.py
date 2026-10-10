"""Kaggle GPU job: train PE-Core-L students on the data-factory parts and score the sealed sets.

Inputs (kernel sources = outputs of kaggle/data_factory, dataset synth-student-v1):
  data-factory-train-a  OpenFake core train shards (12)
  data-factory-train-b  COCO, WikiArt, ImageNet-R, CC3M (reals: photos, art, web/designs)
  data-factory-train-c  OpenFake Reddit shards 0-2 (posted online), if attached
  data-factory-eval     sealed sets: OpenFake core test 1-2, Reddit 3-4, hard reals
  synth-student-v1      the served student (experiment 23, trained on 13,000 crops)

Each entry of VARIANTS trains one student on its own GPU (recipe of experiment 23 "labels": last 8
blocks, JPEG/SNS copies, balanced batches of 32, STEPS steps) and scores the sealed sets.
`reddit` repeats the Reddit training rows that many times (oversampling the in-the-wild data).
SCORE_SERVED scores the served student too (needs a free GPU).
Experiment 26: VARIANTS = [("v2", 1)] without train-c, SCORE_SERVED = True.
Experiment 27: VARIANTS = [("v3", 1), ("v3r", 3)] with train-c.
20% of the training reals (by file hash) are held aside for each student's threshold.
Outputs: scores_<model>.npz per model and student_<model>.pt.
"""

import csv
import glob
import hashlib
import tarfile
import threading
import time
from pathlib import Path

import numpy as np

STEPS = 3000
VARIANTS = [("v3", 1), ("v3r", 3)]  # (name, Reddit repeat), one per GPU
SCORE_SERVED = False
CONDS = ["original", "sns"]
DATA = Path("/tmp/data")
SMOKE = False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_parts():
    """Unpack every data-factory output found under /kaggle/input; return meta rows."""
    rows = []
    for meta in sorted(
        Path(p) for d in range(1, 5) for p in glob.glob("/kaggle/input" + "/*" * d + "/meta.csv")
    ):
        part = meta.parent
        archives = sorted(part.glob("crops_*.bin"))
        if not archives:
            continue
        dest = DATA / part.name
        dest.mkdir(parents=True, exist_ok=True)
        for archive in archives:
            with tarfile.open(archive) as tar:
                tar.extractall(dest)
        for r in csv.DictReader(meta.open()):
            rows.append({**r, "file": str(dest / r["file"])})
        log(f"{part.name}: {len(archives)} archives, {len(rows)} rows so far")
    return rows


def held_aside(row):
    return (
        row["label"] == "0"
        and int(hashlib.sha256(row["file"].encode()).hexdigest()[:8], 16) % 5 == 0
    )


def served_student(gpu):
    """The experiment-23 student: public PE-Core-L weights + its trained blocks and head."""
    import distill
    import torch

    model, head = distill.build(dict(arch=distill.PE_L, size=336, blocks=8), gpu)
    found = (glob.glob("/kaggle/input" + "/*" * d + "/student_distill.pt") for d in range(1, 5))
    saved = torch.load(next(p for paths in found for p in paths), map_location="cpu")
    result = model.load_state_dict(saved["model"], strict=False)  # copies cast to each dtype
    assert not result.unexpected_keys, result.unexpected_keys
    head.load_state_dict(saved["head"])
    return model.eval(), head.eval()


def score_sets(name, model, head, gpu, rows, extra=None):
    import distill

    files = [r["file"] for r in rows]
    out = {c: distill.score(model, head, gpu, files, c, 336) for c in CONDS}
    log(f"{name} scored {len(files)} images x {len(CONDS)}")
    np.savez(
        f"/kaggle/working/scores_{name}.npz",
        **out,
        **(extra or {}),
        label=np.array([int(r["label"]) for r in rows]),
        set=np.array([r["set"] for r in rows]),
        source=np.array([r["source"] for r in rows]),
        model=np.array([r["model"] for r in rows]),
    )


def main():
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "timm"], check=False)
    import distill

    rows = load_parts()
    train = [r for r in rows if r["set"] == "train"]
    evaluation = [r for r in rows if r["set"] != "train"]
    fit = [r for r in train if not held_aside(r)]
    cal = [r for r in train if held_aside(r)]
    steps = STEPS
    if SMOKE:
        steps, evaluation = 20, evaluation[::100]
    reddit = sum(r["source"] == "openfake_reddit" for r in fit)
    log(f"train {len(fit)} ({reddit} Reddit) +{len(cal)} held-aside reals, eval {len(evaluation)}")
    jobs = []  # (name, gpu, model, head, files, labels); all models load before threads start
    for gpu, (name, repeat) in enumerate(VARIANTS):
        rows_v = [r for r in fit for _ in range(repeat if r["source"] == "openfake_reddit" else 1)]
        run = dict(name=name, arch=distill.PE_L, size=336, blocks=8, kind="labels")
        run.update(steps=steps, batch=32, accum=1)
        labels = np.array([int(r["label"]) for r in rows_v])
        jobs.append((run, gpu, *distill.build(run, gpu), [r["file"] for r in rows_v], labels))
    served = served_student(len(VARIANTS)) if SCORE_SERVED else None
    results = {}

    def work(run, gpu, model, head, files, labels):
        distill.train(
            run, gpu, model, head, files, labels, np.zeros(len(files), np.float32), results
        )
        model, head = results[run["name"]]
        cal_scores = distill.score(model, head, gpu, [r["file"] for r in cal], "original", 336)
        score_sets(run["name"], model, head, gpu, evaluation, {"cal_original": cal_scores})
        distill.save(model, head, run["name"])

    threads = [threading.Thread(target=work, args=job) for job in jobs]
    if served:
        threads.append(
            threading.Thread(
                target=score_sets, args=("v1_served", *served, len(VARIANTS), evaluation)
            )
        )
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("done")


if __name__ == "__main__":
    main()
