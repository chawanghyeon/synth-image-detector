"""Train and evaluate linear probes on SigLIP2 features made by kaggle/siglip_features.

Usage: python scripts/eval_siglip_probe.py [data/siglip]
Reads meta.csv and feats/<set>__<condition>.npy. 20% of the training reals (by image) are held
aside to set the threshold at 1% FPR on original images; nothing from the test set is used to fit.
Probes: "original" (original crops only) and "augmented" (original + JPEG + SNS copies of every
training image). Writes reports/siglip_probe.json.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/siglip")
TEST_CONDS = ["original", "jpeg75", "jpeg50", "half", "sns"]


def load(set_name, cond):
    return np.load(ROOT / "feats" / f"{set_name}__{cond}.npy").astype(np.float32)


def readable(set_name, conds):
    """Rows whose image was read under every condition (the job flags unreadable files)."""
    masks = [
        np.load(p) for c in conds if (p := ROOT / "feats" / f"{set_name}__{c}__ok.npy").exists()
    ]
    return np.logical_and.reduce(masks) if masks else None


def main():
    meta = list(csv.DictReader((ROOT / "meta.csv").open()))
    rows = {s: [m for m in meta if m["set"] == s] for s in ("train", "val", "test")}
    y = np.array([int(m["label"]) for m in rows["train"]])
    rng = np.random.default_rng(0)
    cal = (y == 0) & (rng.random(len(y)) < 0.2)
    fit = ~cal
    train = {c: load("train", c) for c in ("original", "jpeg", "sns")}
    good = readable("train", train)
    if good is not None:
        fit, cal = fit & good, cal & good
    test = {c: load("test", c) for c in TEST_CONDS}
    keep = readable("test", TEST_CONDS)
    keep = np.ones(len(rows["test"]), bool) if keep is None else keep
    test = {c: v[keep] for c, v in test.items()}
    t_label = np.array([int(m["label"]) for m in rows["test"]])[keep]
    t_source = np.array([m["source"] for m in rows["test"]])[keep]
    t_model = np.array([m["model"] for m in rows["test"]])[keep]
    report = {"n_train_fit": int(fit.sum()), "n_cal": int(cal.sum()), "probes": {}}
    for probe, conds in (("original", ["original"]), ("augmented", ["original", "jpeg", "sns"])):
        x = np.concatenate([train[c][fit] for c in conds])
        yy = np.tile(y[fit], len(conds))
        sc = StandardScaler().fit(train["original"][fit])
        w = np.where(yy == 1, 0.5 / yy.sum(), 0.5 / (len(yy) - yy.sum()))
        clf = LogisticRegression(C=0.1, max_iter=5000).fit(
            sc.transform(x), yy, sample_weight=w / w.mean()
        )

        def score(a, clf=clf, sc=sc):
            return clf.decision_function(sc.transform(a))

        thr = float(np.quantile(score(train["original"][cal]), 0.99))
        res = {}
        for cond in TEST_CONDS:
            s = score(test[cond])
            r = {}
            for source in sorted(set(t_source.tolist())):
                m = t_source == source
                reals, fakes = s[m & (t_label == 0)], s[m & (t_label == 1)]
                entry = {"fpr": round(float((reals > thr).mean()), 4), "n_real": int(len(reals))}
                if len(fakes):
                    entry["tpr"] = round(float((fakes > thr).mean()), 4)
                    entry["n_fake"] = int(len(fakes))
                    entry["auroc"] = round(
                        float(
                            roc_auc_score(
                                np.r_[np.zeros(len(reals)), np.ones(len(fakes))],
                                np.r_[reals, fakes],
                            )
                        ),
                        4,
                    )
                r[source] = entry
            if cond == "original":
                by = defaultdict(list)
                for i in np.flatnonzero((t_label == 1) & (t_source == "openfake_core")):
                    by[str(t_model[i])].append(s[i] > thr)
                r["openfake_tpr_by_generator"] = {
                    k: [round(float(np.mean(v)), 3), len(v)]
                    for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))
                }
            res[cond] = r
            print(
                probe,
                cond,
                {k: v for k, v in r.items() if k != "openfake_tpr_by_generator"},
                flush=True,
            )
        report["probes"][probe] = res
    Path("reports").mkdir(exist_ok=True)
    Path("reports/siglip_probe.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
