"""Kaggle GPU job body: train one student per GPU on the attached crops, score the sealed sets.

Inputs: data-factory notebook outputs and datasets (crops_*.bin + meta.csv) under /kaggle/input.
20% of the training reals (detector.data.records.held_aside) are scored, never trained on, for the
"cal" threshold. Outputs per recipe in /kaggle/working:
  student_<name>.pt   trained weights (detector.training.model.Student checkpoint)
  scores_<name>.npz   logits per condition for every evaluation row, plus label/set/source/model
                      and cal_original (detector.evaluation.scores.Scores reads it)
"""

import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from detector.data.records import Row, held_aside, unpack_parts
from detector.training.model import Student
from detector.training.recipe import Recipe
from detector.training.trainer import Trainer, score_files

CONDITIONS = ["original", "sns"]
OUT = Path("/kaggle/working")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_scores(path: Path, rows: list[Row], logits: dict[str, np.ndarray]) -> None:
    np.savez(
        path,
        **logits,
        label=np.array([r.label for r in rows]),
        set=np.array([r.set for r in rows]),
        source=np.array([r.source for r in rows]),
        model=np.array([r.model for r in rows]),
    )


def run(recipes: list[Recipe], smoke: bool = False, out: Path = OUT) -> None:
    rows = unpack_parts()
    train = [r for r in rows if r.set == "train"]
    evaluation = [r for r in rows if r.set != "train"]
    fit = [r for r in train if not held_aside(r)]
    cal = [r for r in train if held_aside(r)]
    if smoke:
        recipes, evaluation = [replace(r, steps=20) for r in recipes], evaluation[::100]
    reddit = sum(r.source == "openfake_reddit" for r in fit)
    log(f"train {len(fit)} ({reddit} Reddit) +{len(cal)} held-aside reals, eval {len(evaluation)}")

    jobs = []  # build every model before threads start
    for gpu, recipe in enumerate(recipes):
        rows_r = [r for r in fit for _ in range(recipe.repeats.get(r.source, 1))]
        log(f"{recipe.name}: {len(rows_r)} training rows after repeats {recipe.repeats}")
        trainer = Trainer(Student.for_training(recipe), recipe, f"cuda:{gpu}", log)
        jobs.append((gpu, recipe, trainer, rows_r))

    def work(gpu: int, recipe: Recipe, trainer: Trainer, rows_r: list[Row]) -> None:
        labels = np.array([r.label for r in rows_r])
        student = trainer.fit([r.file for r in rows_r], labels, seed=gpu)
        device, size = f"cuda:{gpu}", recipe.size
        logits = {
            c: score_files(student, [r.file for r in evaluation], c, size, device)
            for c in CONDITIONS
        }
        logits["cal_original"] = score_files(
            student, [r.file for r in cal], "original", size, device
        )
        log(f"{recipe.name} scored {len(evaluation)} images x {len(CONDITIONS)}")
        save_scores(out / f"scores_{recipe.name}.npz", evaluation, logits)
        student.save(out / f"student_{recipe.name}.pt")

    threads = [threading.Thread(target=work, args=job) for job in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("done")
