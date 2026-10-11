"""Kaggle GPU job: fine-tune one student per T4 and score the sealed sets
(detector.training.job). Attached: data-factory outputs train-a..d and eval, and the Codex
GPT-Image dataset synth-codex-train.

RECIPES, one per GPU (at most 2). Served model (experiment 28):
  Recipe("v4", repeats={"openfake_reddit": 3, "codex": 5}) with synth-codex-train version 1
  (23 images); the current version has 300 (experiment 29: no better).
Experiments 26-30 are recorded in docs/experiments.md.
"""

from detector.training.recipe import Recipe

SMOKE = False  # 20 steps and 1% of the evaluation rows
RECIPES = [
    Recipe("v4", repeats={"openfake_reddit": 3, "codex": 5}),
]


def main():
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "timm"], check=False)
    from detector.training import job

    job.run(RECIPES, smoke=SMOKE)


if __name__ == "__main__":
    main()
