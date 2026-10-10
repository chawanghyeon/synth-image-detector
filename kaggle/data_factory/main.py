"""Kaggle CPU job: download one part of the data (see factory.sources), cut the native-resolution
crops, and pack them into a few tar archives named *.bin in /kaggle/working (Kaggle mounts a few
large files reliably, tens of thousands of small ones not, and unpacks *.tar itself).

Push each part as its own notebook (CPU only, no GPU quota):
  python kaggle/push.py kaggle/data_factory --set PART=eval --suffix eval
Output: meta.csv (set, source, label, model, file relative to the unpacked root) and crops_*.bin.
Training jobs attach these notebooks as kernel sources.
"""

import csv
import tarfile
import time
from pathlib import Path

PART = "eval"
OUT = Path("/kaggle/working")
CHUNK = 3 * 2**30  # bytes of PNG per archive
SMOKE = False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    import factory

    factory.SMOKE = SMOKE
    meta = factory.phase_a(factory.sources(PART))
    root = factory.TMP / "crops"
    archive, size, index = None, 0, 0
    for m in meta:
        path = Path(m["file"])
        if archive is None or size > CHUNK:
            if archive:
                archive.close()
            archive, size = tarfile.open(OUT / f"crops_{PART}_{index:02d}.bin", "w"), 0
            index += 1
        archive.add(path, arcname=str(path.relative_to(root)))
        size += path.stat().st_size
        m["file"] = str(path.relative_to(root))
    archive.close()
    with (OUT / "meta.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, ["set", "source", "label", "model", "file"])
        writer.writeheader()
        writer.writerows(meta)
    log(f"{PART}: {len(meta)} crops in {index} archives")


if __name__ == "__main__":
    main()
