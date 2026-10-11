"""Cut one deterministic native-resolution crop per image and pack the crops for Kaggle.

The crop is a square of min(512, width, height) px (even, at least 256) at a position seeded by
the item key, rebuilt from pixels (generator PNGs carry huge text chunks in their metadata).
"""

import hashlib
import io
import json
import shutil
import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin

from detector.data.records import Row, write_meta
from detector.data.sources import TMP, Source

PngImagePlugin.MAX_TEXT_CHUNK = 64 * 2**20  # tolerate large metadata when decoding sources
CROP, MIN_CROP = 512, 256
CHUNK = 3 * 2**30  # bytes of PNG per archive


def crop_name(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16] + ".png"


def native_crop(data: bytes, key: str) -> Image.Image | None:
    """The crop for this image, or None if it cannot be decoded or is too small."""
    try:
        with Image.open(io.BytesIO(data)) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
    except Exception:
        return None
    width, height = image.size
    size = min(CROP, width, height) // 2 * 2
    if size < MIN_CROP:
        return None
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    left, top = seed % (width - size + 1), (seed // 7919) % (height - size + 1)
    return Image.fromarray(np.asarray(image.crop((left, top, left + size, top + size))))


def _save_crop(job: tuple[bytes, str, str]) -> bool:
    data, key, out = job
    crop = native_crop(data, key)
    if crop is not None:
        crop.save(out)
    return crop is not None


class CropFactory:
    """Sources -> crops under root/<set>/<hash>.png -> crops_<name>_NN.bin + meta.csv."""

    def __init__(self, root: Path = TMP / "crops", workers: int = 4) -> None:
        self.root, self.workers = root, workers

    def build(self, sources: list[Source], limit: int | None = None, log=print) -> list[Row]:
        rows = []
        with Pool(self.workers) as pool:
            for source in sources:
                folder = self.root / source.set_name
                folder.mkdir(parents=True, exist_ok=True)
                for batch in source.batches(limit):
                    jobs = [(it.data, it.key, str(folder / crop_name(it.key))) for it in batch]
                    for ok, item, job in zip(
                        pool.imap(_save_crop, jobs, 8), batch, jobs, strict=False
                    ):
                        if ok:
                            file = str(Path(job[2]).relative_to(self.root))
                            rows.append(
                                Row(source.set_name, source.source, item.label, item.model, file)
                            )
                log(f"{source.set_name}/{source.source}: {len(rows)} crops so far")
        return rows

    def pack(self, rows: list[Row], out: Path, name: str, chunk: int = CHUNK) -> int:
        """Tar archives of at most ~chunk bytes; returns how many were written."""
        chunks: list[list[Row]] = []
        size = chunk + 1
        for row in rows:
            if size > chunk:
                chunks.append([])
                size = 0
            chunks[-1].append(row)
            size += (self.root / row.file).stat().st_size
        for i, rows_i in enumerate(chunks):
            with tarfile.open(out / f"crops_{name}_{i:02d}.bin", "w") as archive:
                for row in rows_i:
                    archive.add(self.root / row.file, arcname=row.file)
        write_meta(out / "meta.csv", rows)
        return len(chunks)


def pack_codex(src: Path, out: Path, dataset: str = "chawanghyeon/synth-codex-train") -> int:
    """Codex (GPT-Image) training images src/t*.png -> a Kaggle dataset folder (label 1, source
    "codex"), ready for `kaggle datasets version -p out`."""
    factory = CropFactory(root=out / "crops")
    rows = []
    for path in sorted(src.glob("t*.png")):
        key = f"codex/{path.name}"
        crop = native_crop(path.read_bytes(), key)
        if crop is not None:
            (factory.root / "codex").mkdir(parents=True, exist_ok=True)
            crop.save(factory.root / "codex" / crop_name(key))
            rows.append(Row("train", "codex", 1, "gpt-image-codex", f"codex/{crop_name(key)}"))
    factory.pack(rows, out, "codex")
    title = dataset.split("/")[1]
    meta = {"title": title, "id": dataset, "licenses": [{"name": "other"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta))
    shutil.rmtree(factory.root)
    return len(rows)
