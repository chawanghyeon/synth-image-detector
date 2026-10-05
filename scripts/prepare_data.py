"""Turn images into one native-resolution crop each, plus a metadata CSV (data/crops/index.csv).

Crops are never resized and are saved losslessly, so pixel-level traces survive.

Usage:
  python scripts/prepare_data.py data/raw/core/*.parquet        # OpenFake shards
  python scripts/prepare_data.py --real coco data/raw/val2017   # folder of real photos
"""

import csv
import hashlib
import io
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageOps

CROP = 512
OUT = Path("data/crops")
FIELDS = ["file", "split", "label", "model", "type", "src_format", "width", "height", "src_bytes"]


def crop_one(job: tuple[bytes, str, str, str, str, str]) -> dict | None:
    data, row_id, split, label, model, kind = job
    try:
        with Image.open(io.BytesIO(data)) as raw:
            src_format = raw.format
            image = ImageOps.exif_transpose(raw).convert("RGB")
    except Exception:
        return None
    width, height = image.size
    size = min(CROP, width, height) // 2 * 2
    if size < 256:
        return None
    seed = int(hashlib.sha256(row_id.encode()).hexdigest()[:8], 16)
    left = seed % (width - size + 1)
    top = (seed // 7919) % (height - size + 1)
    name = f"{split}/{hashlib.sha256(row_id.encode()).hexdigest()[:16]}.png"
    image.crop((left, top, left + size, top + size)).save(OUT / name)
    return dict(
        file=name,
        split=split,
        label=int(label == "fake"),
        model=model,
        type=kind,
        src_format=src_format,
        width=width,
        height=height,
        src_bytes=len(data),
    )


def _rows(source: str):
    """Yield (split, jobs) batches for one parquet shard."""
    split = Path(source).name.split("-")[0]
    parquet = pq.ParquetFile(source)
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=["image", "row_id", "label", "model", "type"])
        yield (
            split,
            [
                (r["image"]["bytes"], r["row_id"], split, r["label"], r["model"], r["type"])
                for r in table.to_pylist()
            ],
        )


def _real_folder(name: str, folder: str, limit: int = 5000):
    """Real photos from a folder go to the "extra" split, tagged with the source name."""
    files = sorted(f for f in Path(folder).rglob("*.jpg") if not f.name.startswith("."))[:limit]
    for start in range(0, len(files), 256):
        yield (
            "extra",
            [
                (f.read_bytes(), f"{name}/{f.name}", "extra", "real", name, "real")
                for f in files[start : start + 256]
            ],
        )


def main(args: list[str]) -> None:
    batches = [_real_folder(args[1], args[2])] if args[0] == "--real" else map(_rows, args)
    OUT.mkdir(parents=True, exist_ok=True)
    index = OUT / "index.csv"
    new = not index.exists()
    with index.open("a", newline="") as stream, ProcessPoolExecutor(8) as pool:
        writer = csv.DictWriter(stream, FIELDS)
        if new:
            writer.writeheader()
        for source in batches:
            for split, jobs in source:
                (OUT / split).mkdir(parents=True, exist_ok=True)
                writer.writerows(row for row in pool.map(crop_one, jobs) if row)
                stream.flush()
            print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
