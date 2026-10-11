"""meta.csv rows and the crop archives (crops_*.bin) that carry them between Kaggle jobs.

Kaggle mounts a few large files reliably but not tens of thousands of small ones, and it unpacks
*.tar itself, so crops travel as tar archives named *.bin and are extracted to local disk.
"""

import csv
import glob
import hashlib
import tarfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

FIELDS = ["set", "source", "label", "model", "file"]


@dataclass(frozen=True)
class Row:
    set: str  # "train", or an evaluation set: core_test1/2, reddit3/4, hard_real
    source: str  # openfake_core, openfake_reddit, coco, wikiart, imagenet_r, cc3m, codex
    label: int  # 1 = AI-generated
    model: str  # generator name, or the real source
    file: str  # path relative to the archive root, or absolute once unpacked


def read_meta(path: Path) -> list[Row]:
    with path.open() as f:
        return [Row(**{**r, "label": int(r["label"])}) for r in csv.DictReader(f)]


def write_meta(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)


def held_aside(row: Row) -> bool:
    """20% of the training reals (by path hash), kept out of training for the threshold."""
    return row.label == 0 and int(hashlib.sha256(row.file.encode()).hexdigest()[:8], 16) % 5 == 0


def unpack_parts(inputs: Path = Path("/kaggle/input"), dest: Path = Path("/tmp/data")) -> list[Row]:
    """Extract every crops_*.bin next to a meta.csv under `inputs` (attached notebook outputs and
    datasets, up to 4 levels deep; never walked recursively, the mount is slow). Rows get absolute
    paths under dest/<folder name>."""
    rows = []
    metas = sorted(Path(p) for d in range(1, 5) for p in glob.glob(f"{inputs}{'/*' * d}/meta.csv"))
    for meta in metas:
        archives = sorted(meta.parent.glob("crops_*.bin"))
        if not archives:
            continue
        root = dest / meta.parent.name
        root.mkdir(parents=True, exist_ok=True)
        for archive in archives:
            with tarfile.open(archive) as tar:
                tar.extractall(root, filter="data")
        rows += [replace(r, file=str(root / r.file)) for r in read_meta(meta)]
    return rows
