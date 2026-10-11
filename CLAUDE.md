# synth-image-detector: working notes

- Commit directly on `main`. Pre-commit runs ruff (check + format), file checks and pytest; run `uv run pre-commit run --all-files` before committing.
- Environment: `export PATH="$HOME/.local/bin:$PATH" UV_PYTHON_INSTALL_DIR=.python`, then `uv run ...`. The `kaggle` CLI is in `~/.local/bin`.
- Layout and import rules: see the docstring of `src/detector/__init__.py`. `serving` must never import torch; `training` may.
- Experiments: change `RECIPES` in `kaggle/train/main.py` (one `Recipe` per GPU, max 2), `detector push kaggle/train`, then `detector wait chawanghyeon/train-v2 data/expNN` (run it in the background; it polls, downloads and prints the report), `detector bench data/expNN <names>`.
- Evaluation protocol (do not change silently): threshold = 1% FPR on Reddit 3 reals (`--reals reddit3`, the served setting); report Reddit 4, new generators (core test 1-2), hard reals, Codex 50 and real photos 100, each original and SNS (0.75x + JPEG 80). `ai_score` is not a calibrated probability.
- Record every experiment in `docs/experiments.md` (Korean, numbered) with the comparison table and the decision; update README numbers only when the served model changes.
- Data under `data/` is not in Git: `codex_test` (50 eval images), `real_photos` (100), `codex_train` (300), `exp28` (served v4 checkpoint and scores). Kaggle holds the crop datasets (data-factory-* notebook outputs, synth-codex-train).
