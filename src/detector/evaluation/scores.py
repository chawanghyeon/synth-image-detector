"""scores_<name>.npz (detector.training.job): logits on the sealed sets, thresholds, metrics.

Protocol: the threshold is the 99th percentile of real-image logits on a set never trained on,
fixed before looking at the other sets. "reddit3" (images posted online) is the served setting;
"cal" uses the held-aside training reals.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

GROUPS = {  # report rows -> evaluation sets
    "new generators (core test 1-2)": ("core_test1", "core_test2"),
    "Reddit 3 (threshold)": ("reddit3",),
    "Reddit 4 (sealed)": ("reddit4",),
    "hard reals": ("hard_real",),
}
SERVED_REALS = "reddit3"


def auroc(reals: np.ndarray, fakes: np.ndarray) -> float:
    """Probability that a random AI image outscores a random real one (ties count half)."""
    values = np.concatenate([reals, fakes])
    ranks = np.empty(len(values))
    ranks[values.argsort(kind="mergesort")] = np.arange(1, len(values) + 1)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inverse, weights=ranks) / counts)[inverse]  # average ranks over ties
    n_real, n_fake = len(reals), len(fakes)
    return float((ranks[n_real:].sum() - n_fake * (n_fake + 1) / 2) / (n_real * n_fake))


@dataclass(frozen=True)
class Rates:
    n_ai: int
    n_real: int
    tpr: float  # at the fixed threshold
    fpr: float
    auroc: float | None = None
    tpr_at_1pct: float | None = None  # at 1% FPR on this group's own reals

    def __str__(self) -> str:
        text = f"FPR {self.fpr:.3f}"
        if self.n_ai:
            text = f"TPR {self.tpr:.3f} {text} | auc {self.auroc:.4f} tpr@1% {self.tpr_at_1pct:.3f}"
        return text


@dataclass
class Scores:
    name: str
    logits: dict[str, np.ndarray]  # condition -> logit per row
    label: np.ndarray
    set: np.ndarray
    cal: np.ndarray | None = None  # held-aside training reals, original

    @classmethod
    def load(cls, path: str | Path) -> "Scores":
        path = Path(path)
        data = dict(np.load(path))
        meta = {k: data.pop(k) for k in ("label", "set", "source", "model") if k in data}
        cal = data.pop("cal_original", None)
        return cls(path.stem.removeprefix("scores_"), data, meta["label"], meta["set"], cal)

    @property
    def conditions(self) -> list[str]:
        return list(self.logits)

    def threshold(self, reals: str = SERVED_REALS) -> float:
        """1% FPR on the original reals of an evaluation set, or on the held-aside reals ("cal")."""
        if reals == "cal":
            if self.cal is None:
                raise ValueError(f"{self.name} has no held-aside reals")
            return float(np.quantile(self.cal, 0.99))
        mask = (self.set == reals) & (self.label == 0)
        if not mask.any():
            raise ValueError(f"{self.name} has no reals in set {reals!r}")
        return float(np.quantile(self.logits["original"][mask], 0.99))

    def rates(self, sets: tuple[str, ...], condition: str, threshold: float) -> Rates:
        mask = np.isin(self.set, sets)
        x, y = self.logits[condition][mask], self.label[mask]
        reals, fakes = x[y == 0], x[y == 1]
        if not len(fakes):
            return Rates(0, len(reals), float("nan"), float((reals > threshold).mean()))
        return Rates(
            len(fakes),
            len(reals),
            float((fakes > threshold).mean()),
            float((reals > threshold).mean()) if len(reals) else float("nan"),
            auroc(reals, fakes) if len(reals) else None,
            float((fakes > np.quantile(reals, 0.99)).mean()) if len(reals) else None,
        )


def report(paths: list[Path], reals: str = SERVED_REALS) -> str:
    lines = []
    for path in paths:
        scores = Scores.load(path)
        thr = scores.threshold(reals)
        lines.append(f"== {scores.name} (threshold {thr:+.2f}, 1% FPR on {reals})")
        for group, sets in GROUPS.items():
            if not np.isin(scores.set, sets).any():
                continue
            cells = [f"{c}: {scores.rates(sets, c, thr)}" for c in scores.conditions]
            r = scores.rates(sets, scores.conditions[0], thr)
            lines.append(
                f"  {group:32s} AI {r.n_ai:5d} / real {r.n_real:5d}   " + "   ".join(cells)
            )
    return "\n".join(lines)
