"""Score models on the sealed evaluation sets made by kaggle/data_factory (PART eval).

Usage: python scripts/eval_sealed.py data/exp26
For every scores_<model>.npz written by kaggle/train: per set (OpenFake core test 1-2 = new
generators, Reddit 3 and 4 = posted online, hard reals = CC3M / WikiArt / ImageNet-R) and condition,
AUROC, detection at 1% FPR on that set's reals, and detection / FPR at the model's own threshold
(1% FPR on its held-aside training reals; the served model's threshold is models/student.json).
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/exp26")
GROUPS = {
    "new generators (core test 1-2)": ("core_test1", "core_test2"),
    "Reddit 3": ("reddit3",),
    "Reddit 4 (sealed)": ("reddit4",),
    "hard reals": ("hard_real",),
}


def threshold(name: str, scores: dict) -> float:
    if "cal_original" in scores:
        return float(np.quantile(scores["cal_original"], 0.99))
    return json.loads(Path("models/student.json").read_text())["threshold"]


def main() -> None:
    for path in sorted(ROOT.glob("scores_*.npz")):
        s = dict(np.load(path))
        name, thr = path.stem[7:], threshold(path.stem[7:], s)
        print(f"== {name} (threshold {thr:+.2f})")
        conds = [c for c in ("original", "jpeg75", "sns") if c in s]
        for group, sets in GROUPS.items():
            m = np.isin(s["set"], sets)
            label = s["label"][m]
            cells = []
            for c in conds:
                x = s[c][m]
                reals, fakes = x[label == 0], x[label == 1]
                cell = f"{c}: FPR {(reals > thr).mean():.3f}"
                if len(fakes):
                    y = np.r_[np.zeros(len(reals)), np.ones(len(fakes))]
                    auc = roc_auc_score(y, np.r_[reals, fakes])
                    same = (fakes > np.quantile(reals, 0.99)).mean()
                    cell += f" TPR {(fakes > thr).mean():.3f} | auc {auc:.4f} tpr@1% {same:.3f}"
                cells.append(cell)
            n = f"AI {int((label == 1).sum())} / real {int((label == 0).sum())}"
            print(f"  {group:32s} {n:20s} " + "   ".join(cells))


if __name__ == "__main__":
    main()
