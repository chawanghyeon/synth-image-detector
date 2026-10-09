"""Teacher scores for kaggle/distill from the PE-Core-G features of kaggle/siglip_features.

Usage: python scripts/make_teacher.py [data/compare_peg]
Per training crop: the logit of a PE-Core-G linear probe (original + SNS copies, C 0.03) on the
clean original crop, cross-fitted over 5 folds so that no crop is scored by a probe that saw it.
The 20% held-aside reals (seed 0, as in the evaluation scripts) get no score and are not trained on.
Writes kaggle/distill/teacher.npz (logit, cal, label) in the order of the training rows
of meta.csv.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/compare_peg")


def main() -> None:
    meta = [m for m in csv.DictReader((ROOT / "meta.csv").open()) if m["set"] == "train"]
    y = np.array([int(m["label"]) for m in meta])
    cal = (y == 0) & (np.random.default_rng(0).random(len(y)) < 0.2)
    x = {
        c: np.load(ROOT / "feats" / f"pe_g__train__{c}.npy").astype(np.float32)
        for c in ("original", "sns")
    }
    teacher = np.full(len(y), np.nan, np.float32)
    fit = np.flatnonzero(~cal)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(fit, y[fit]):
        train, held = fit[a], fit[b]
        scaler = StandardScaler().fit(x["original"][train])
        xx = np.concatenate([x[c][train] for c in x])
        yy = np.tile(y[train], len(x))
        w = np.where(yy == 1, 0.5 / yy.sum(), 0.5 / (len(yy) - yy.sum()))
        clf = LogisticRegression(C=0.03, max_iter=5000)
        clf.fit(scaler.transform(xx), yy, sample_weight=w / w.mean())
        teacher[held] = clf.decision_function(scaler.transform(x["original"][held]))
    print("cross-fitted AUROC", round(roc_auc_score(y[~cal], teacher[~cal]), 4))
    np.savez_compressed("kaggle/distill/teacher.npz", logit=teacher, cal=cal, label=y)


if __name__ == "__main__":
    main()
