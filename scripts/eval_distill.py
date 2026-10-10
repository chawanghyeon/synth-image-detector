"""Compare the distilled PE-Core-L students (kaggle/distill) with their PE-Core-G teacher.

Usage: python scripts/eval_distill.py data/<run> [data/compare_peg] [data/distill]
The third folder adds the experiment-23 students for comparison (cut to the same test prefix).
Same 6,000 test crops and conditions for all. Reported: AUROC, detection at 1% FPR on the test
reals (same operating point for everyone), and detection / FPR at a threshold fixed at 1% FPR on the
held-aside training reals.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/distill")
PEG = Path(sys.argv[2] if len(sys.argv) > 2 else "data/compare_peg")
EXTRA = Path(sys.argv[3] if len(sys.argv) > 3 else "data/distill")
CONDS = ["original", "jpeg75", "sns"]


def teacher_scores():
    meta = list(csv.DictReader((PEG / "meta.csv").open()))
    y = np.array([int(m["label"]) for m in meta if m["set"] == "train"])
    cal = (y == 0) & (np.random.default_rng(0).random(len(y)) < 0.2)

    def load(set_name, cond):
        return np.load(PEG / "feats" / f"pe_g__{set_name}__{cond}.npy").astype(np.float32)

    tr = {c: load("train", c) for c in ("original", "sns")}
    sc = StandardScaler().fit(tr["original"][~cal])
    x = np.concatenate([tr[c][~cal] for c in tr])
    yy = np.tile(y[~cal], 2)
    w = np.where(yy == 1, 0.5 / yy.sum(), 0.5 / (len(yy) - yy.sum()))
    clf = LogisticRegression(C=0.03, max_iter=5000).fit(
        sc.transform(x), yy, sample_weight=w / w.mean()
    )

    def f(a):
        return clf.decision_function(sc.transform(a))

    return {"cal_original": f(tr["original"][cal]), **{c: f(load("test", c)) for c in CONDS}}


def report(name, s, label, of):
    thr = np.quantile(s["cal_original"], 0.99)
    cells = []
    for c in CONDS:
        reals, fakes = s[c][label == 0], s[c][of & (label == 1)]
        auc = roc_auc_score(np.r_[np.zeros(len(reals)), np.ones(len(fakes))], np.r_[reals, fakes])
        same = (fakes > np.quantile(reals, 0.99)).mean()
        cells.append(
            f"{c}: auc {auc:.4f} tpr@1% {same:.3f} | "
            f"fixed {(fakes > thr).mean():.3f}/{(reals > thr).mean():.3f}"
        )
    print(f"{name:16s} " + "   ".join(cells), flush=True)


def main():
    runs = {p.stem[7:]: dict(np.load(p)) for p in sorted(ROOT.glob("scores_*.npz"))}
    if EXTRA.exists():  # earlier students scored on a longer prefix of the same test crops
        runs.update(
            {f"{p.stem[7:]} (23)": dict(np.load(p)) for p in sorted(EXTRA.glob("scores_*.npz"))}
        )
    n = min(len(r["label"]) for r in runs.values())
    first = next(iter(runs.values()))
    label, of = first["label"][:n], first["source"][:n] == "openfake_core"
    print(
        f"test AI {int((of & (label == 1)).sum())}, reals {int((label == 0).sum())}; "
        "fixed = detection/FPR at the held-aside threshold"
    )
    cut = {"cal_original"}
    teacher = {k: v if k in cut else v[:n] for k, v in teacher_scores().items()}
    report("teacher PE-G", teacher, label, of)
    for name, r in runs.items():
        report(name, {k: v if k in cut else v[:n] for k, v in r.items()}, label, of)


if __name__ == "__main__":
    main()
