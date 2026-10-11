"""`detector` command: run Kaggle jobs, evaluate their outputs, export the served model.

detector push kaggle/train [--smoke] [--set NAME=VALUE ...] [--suffix S]
detector wait chawanghyeon/train-v2 data/exp31      wait, download, print the sealed report
detector report data/exp31 [--reals reddit3|cal]    sealed-set metrics at a fixed threshold
detector bench data/exp31 v7 [v8 ...]               Codex / real-photo benchmark (laptop)
detector bench --served                             same, through the served ONNX model
detector export data/exp28/student_v4.pt data/exp28/scores_v4.npz [--reals reddit3]
detector pack-codex data/codex_train data/codex_train_ds
"""

import argparse
from pathlib import Path

from detector.evaluation.scores import SERVED_REALS


def cmd_push(args) -> None:
    from detector.kaggle_jobs import KaggleJob

    settings = dict(s.split("=", 1) for s in args.set)
    print(KaggleJob(args.job).push(settings, args.suffix, args.smoke))


def cmd_wait(args) -> None:
    from detector import kaggle_jobs
    from detector.evaluation.scores import report

    state = kaggle_jobs.wait(args.kernel, args.poll)
    print(f"{args.kernel}: {state}")
    files = kaggle_jobs.fetch(args.kernel, Path(args.out))
    print("\n".join(str(f) for f in files))
    scores = [f for f in files if f.name.startswith("scores_")]
    if scores:
        print(report(scores, args.reals))


def cmd_report(args) -> None:
    from detector.evaluation.scores import report

    run = Path(args.run)
    paths = [run / f"scores_{n}.npz" for n in args.names] or sorted(run.glob("scores_*.npz"))
    print(report(paths, args.reals))


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def cmd_bench(args) -> None:
    import numpy as np

    from detector.evaluation.codex import CodexBench

    bench = CodexBench()
    if args.served:
        from detector.serving.detector import Detector

        detector = Detector(args.models)
        session = detector.session

        def logits(x):
            return session.run(None, {"images": x})[0]

        print(bench.run("served", logits, detector.card.threshold, detector.card.size))
        return
    import torch

    from detector.evaluation.scores import Scores
    from detector.training.model import Student

    run, device = Path(args.run), _device()
    for name in args.names:
        student = Student.from_checkpoint(run / f"student_{name}.pt").to(device)
        threshold = Scores.load(run / f"scores_{name}.npz").threshold(args.reals)

        def logits(x, student=student):
            with torch.inference_mode():  # one image at a time: MPS fails on some batch shapes
                return np.array(
                    [
                        student(torch.from_numpy(x[i : i + 1]).to(device)).item()
                        for i in range(len(x))
                    ]
                )

        print(bench.run(name, logits, threshold, 336))


def cmd_export(args) -> None:
    from detector.evaluation.scores import Scores
    from detector.model_card import ModelCard
    from detector.training.export import export_onnx
    from detector.training.model import Student
    from detector.training.recipe import PE_L

    weights = Path(args.weights)
    threshold = Scores.load(args.scores).threshold(args.reals)
    card = ModelCard(336, threshold, PE_L, weights.name, args.reals)
    error = export_onnx(Student.from_checkpoint(weights), card, Path(args.out))
    print(f"max |onnx - torch| {error:.2e}; threshold {threshold:.4f} (1% FPR on {args.reals})")


def cmd_pack_codex(args) -> None:
    from detector.data.factory import pack_codex

    print(f"{pack_codex(Path(args.src), Path(args.out))} crops packed into {args.out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="detector", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("push", help="push and start a Kaggle job")
    p.add_argument("job")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--suffix")
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("wait", help="wait for a Kaggle job, download its outputs, report")
    p.add_argument("kernel")
    p.add_argument("out")
    p.add_argument("--poll", type=int, default=120)
    p.add_argument("--reals", default=SERVED_REALS)
    p.set_defaults(func=cmd_wait)

    p = sub.add_parser("report", help="sealed-set metrics of scores_*.npz in a run folder")
    p.add_argument("run")
    p.add_argument("names", nargs="*")
    p.add_argument("--reals", default=SERVED_REALS, help="threshold set, or 'cal'")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("bench", help="Codex / real-photo benchmark")
    p.add_argument("run", nargs="?")
    p.add_argument("names", nargs="*")
    p.add_argument("--served", action="store_true")
    p.add_argument("--models", default="models")
    p.add_argument("--reals", default=SERVED_REALS)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("export", help="checkpoint -> models/student.onnx + student.json")
    p.add_argument("weights")
    p.add_argument("scores")
    p.add_argument("--reals", default=SERVED_REALS)
    p.add_argument("--out", default="models")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("pack-codex", help="Codex training images -> Kaggle dataset folder")
    p.add_argument("src")
    p.add_argument("out")
    p.set_defaults(func=cmd_pack_codex)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
