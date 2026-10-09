"""Compare frozen encoders on the subset made by kaggle/siglip_features in COMPARE mode.

Usage: python scripts/eval_encoders.py data/compare [data/siglip_full]
Each encoder gets the same linear probe (original + degraded training copies, C 0.03). Reported:
AUROC and detection at 1% FPR on the test reals (same operating point for every encoder) plus
detection/FPR at the threshold fixed on 20% held-aside training reals. SigLIP2 rows are taken
from the full feature run by file path. Conditions are whatever the compare run produced.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/compare")
FULL = Path(sys.argv[2] if len(sys.argv) > 2 else "data/siglip_full")


def main():
    meta = list(csv.DictReader((ROOT / "meta.csv").open()))
    rows = {s: [m for m in meta if m["set"] == s] for s in ("train", "test")}
    full = {
        (m["set"], m["file"]): i
        for s in ("train", "test")
        for i, m in enumerate(
            r for r in csv.DictReader((FULL / "meta.csv").open()) if r["set"] == s
        )
    }
    y = np.array([int(m["label"]) for m in rows["train"]])
    cal = (y == 0) & (np.random.default_rng(0).random(len(y)) < 0.2)
    t_label = np.array([int(m["label"]) for m in rows["test"]])
    of = np.array([m["source"] == "openfake_core" for m in rows["test"]])

    def feats(enc, set_name, cond):
        if enc == "siglip2":
            idx = [full[(set_name, m["file"])] for m in rows[set_name]]
            return np.load(FULL / "feats" / f"{set_name}__{cond}.npy")[idx].astype(np.float32)
        return np.load(ROOT / "feats" / f"{enc}__{set_name}__{cond}.npy").astype(np.float32)

    encoders = ["siglip2"] + sorted(
        {p.name.split("__")[0] for p in (ROOT / "feats").glob("*__train__original.npy")}
    )

    def conds(set_name):
        found = {
            p.stem.split("__")[2]
            for p in (ROOT / "feats").glob(f"*__{set_name}__*.npy")
            if not p.stem.endswith("__ok")
        }
        return sorted(found, key=lambda c: c != "original")

    TRAIN, TEST = conds("train"), conds("test")
    print(
        f"train {TRAIN}, test {TEST}; "
        f"test AI {int((of & (t_label == 1)).sum())}, reals {int((t_label == 0).sum())}"
    )
    for enc in encoders:
        try:
            tr = {c: feats(enc, "train", c) for c in TRAIN}
            te = {c: feats(enc, "test", c) for c in TEST}
        except FileNotFoundError as error:
            print(enc, "missing", error.filename)
            continue
        fit = ~cal
        sc = StandardScaler().fit(tr["original"][fit])
        x = np.concatenate([tr[c][fit] for c in TRAIN])
        yy = np.tile(y[fit], len(TRAIN))
        w = np.where(yy == 1, 0.5 / yy.sum(), 0.5 / (len(yy) - yy.sum()))
        clf = LogisticRegression(C=0.03, max_iter=5000).fit(
            sc.transform(x), yy, sample_weight=w / w.mean()
        )
        thr = np.quantile(clf.decision_function(sc.transform(tr["original"][cal])), 0.99)
        cells = []
        for c in TEST:
            s = clf.decision_function(sc.transform(te[c]))
            reals, fakes = s[t_label == 0], s[of & (t_label == 1)]
            auc = roc_auc_score(
                np.r_[np.zeros(len(reals)), np.ones(len(fakes))], np.r_[reals, fakes]
            )
            same = (fakes > np.quantile(reals, 0.99)).mean()
            cells.append(
                f"{c}: auc {auc:.4f} tpr@1% {same:.3f} | "
                f"fixed thr tpr {(fakes > thr).mean():.3f} fpr {(reals > thr).mean():.3f}"
            )
        print(f"{enc:8s} {tr['original'].shape[1]:5d}  " + "   ".join(cells), flush=True)


if __name__ == "__main__":
    main()
