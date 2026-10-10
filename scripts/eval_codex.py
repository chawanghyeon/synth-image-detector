"""Score students on the Codex (GPT-Image) evaluation images and real test photos, on the laptop.

Usage: python scripts/eval_codex.py data/exp28 v4 v4d [...]
For each name: data/<run>/student_<name>.pt with the threshold at 1% FPR on Reddit 3 reals from
data/<run>/scores_<name>.npz (the served setting). Images: data/codex_test/*.png (50, never trained
on) and 100 real OpenFake test photos (short side >= 400 px), each as the original and as an
SNS-style copy (0.75x + JPEG 80). Same crop as the server (detector.features.student_crop).
"""

import csv
import io
import sys
from pathlib import Path

import numpy as np
import torch
from export_student import Student, threshold
from PIL import Image

from detector.features import student_crop

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def sns(image: Image.Image) -> Image.Image:
    image = image.resize((int(image.width * 0.75), int(image.height * 0.75)), Image.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=80)
    return Image.open(buffer).convert("RGB")


def main() -> None:
    run = Path(sys.argv[1])
    ai = [Image.open(p).convert("RGB") for p in sorted(Path("data/codex_test").glob("*.png"))]
    rows = [
        r
        for r in csv.DictReader(open("data/crops/index.csv"))
        if r["split"] == "test"
        and r["label"] == "0"
        and min(int(r["width"]), int(r["height"])) >= 400
    ]
    pick = np.random.default_rng(2).choice(len(rows), 100, replace=False)
    reals = [Image.open("data/crops/" + rows[i]["file"]).convert("RGB") for i in pick]
    for name in sys.argv[2:]:
        model = Student(run / f"student_{name}.pt").eval().to(DEVICE)
        thr = threshold(run / f"scores_{name}.npz", "reddit3")
        with torch.inference_mode():

            def logits(images):
                return np.array(
                    [
                        model(torch.from_numpy(student_crop(im, 336)).to(DEVICE)).item()
                        for im in images
                    ]
                )

            counts = [
                int((logits(ims) > thr).sum())
                for ims in (ai, [sns(i) for i in ai], reals, [sns(i) for i in reals])
            ]
        print(
            f"{name}: Codex {counts[0]}/{len(ai)} original, {counts[1]}/{len(ai)} SNS; "
            f"real photos flagged {counts[2]}/100, {counts[3]}/100",
            flush=True,
        )


if __name__ == "__main__":
    main()
