"""Kaggle CPU job: download one data part (detector.data.sources.PARTS), cut the native crops and
pack them as crops_<part>_NN.bin + meta.csv in /kaggle/working. No GPU quota.

One notebook per part; training jobs attach them as kernel sources:
  detector push kaggle/data_factory --set 'PART="train_a"' --suffix train-a
"""

from pathlib import Path

PART = "eval"
SMOKE = False  # a few hundred images per source, to check the pipeline


def main():
    from detector.data.factory import CropFactory
    from detector.data.sources import PARTS

    factory = CropFactory()
    rows = factory.build(PARTS[PART](), limit=200 if SMOKE else None)
    archives = factory.pack(rows, Path("/kaggle/working"), PART)
    print(f"{PART}: {len(rows)} crops in {archives} archives", flush=True)


if __name__ == "__main__":
    main()
