import itertools

import numpy as np
import pytest
from conftest import noise

from detector.evaluation.codex import CodexBench
from detector.evaluation.scores import Scores, auroc, report


def test_auroc_matches_pair_counting():
    rng = np.random.default_rng(0)
    reals, fakes = rng.integers(0, 5, 40).astype(float), rng.integers(2, 8, 30).astype(float)
    pairs = [(f > r) + 0.5 * (f == r) for r, f in itertools.product(reals, fakes)]
    assert auroc(reals, fakes) == pytest.approx(np.mean(pairs))


@pytest.fixture
def scores_file(tmp_path):
    rng = np.random.default_rng(0)
    sets = np.array(["reddit3"] * 200 + ["reddit4"] * 200 + ["hard_real"] * 100)
    label = np.array(([0] * 100 + [1] * 100) * 2 + [0] * 100)
    original = rng.normal(0, 1, len(label)) + 4 * label
    path = tmp_path / "scores_v9.npz"
    np.savez(
        path,
        original=original,
        sns=original - label,
        cal_original=rng.normal(0, 1, 300),
        label=label,
        set=sets,
        source=np.array(["x"] * len(label)),
        model=np.array(["x"] * len(label)),
    )
    return path


def test_scores_thresholds_and_rates(scores_file):
    scores = Scores.load(scores_file)
    assert scores.name == "v9" and scores.conditions == ["original", "sns"]
    thr = scores.threshold("reddit3")
    reals = scores.logits["original"][(scores.set == "reddit3") & (scores.label == 0)]
    assert (reals > thr).mean() == pytest.approx(0.01)
    assert scores.threshold("cal") == pytest.approx(np.quantile(scores.cal, 0.99))
    r = scores.rates(("reddit4",), "original", thr)
    assert (r.n_ai, r.n_real) == (100, 100) and r.tpr > 0.8 and r.auroc > 0.95
    hard = scores.rates(("hard_real",), "sns", thr)
    assert hard.n_ai == 0 and np.isnan(hard.tpr)
    with pytest.raises(ValueError):
        scores.threshold("reddit5")
    text = report([scores_file])
    assert "v9" in text and "Reddit 4" in text and "core test" not in text


def test_codex_bench_counts(tmp_path):
    for folder, n in (("ai", 3), ("real", 2)):
        (tmp_path / folder).mkdir()
        for i in range(n):
            noise((400, 400), 20 + i).save(tmp_path / folder / f"{i}.png")
    bench = CodexBench(tmp_path / "ai", tmp_path / "real")
    result = bench.run("all", lambda x: np.ones(len(x)), threshold=0.5, size=32)
    assert result.ai == (3, 3, 3) and result.real == (2, 2, 2)
    assert "Codex 3/3 original" in str(result)
