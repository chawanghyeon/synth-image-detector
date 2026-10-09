"""Kaggle GPU job: SigLIP2 feature factory.

Downloads every source itself, cuts the same deterministic native-resolution crop (<= 512 px) as
scripts/prepare_data.py, then extracts frozen SigLIP2 so400m/14@384 features (mean and std of the
patch tokens at the middle and last layer, plus the pooled embedding: 5,760 values, float16) for
the original crop and for degraded copies. Probes are trained on a laptop from these files.

Sets (split, sources):
  train  OpenFake core train shards (12), COCO val2017 (5,000), WikiArt (5 shards), ImageNet-R
         (2 shards), CC3M train (2 shards)       -> original, jpeg (q 50-95), sns (0.5-1x + q 60-90)
  val    OpenFake validation shard 0          -> original
  test   OpenFake test shard 0, Reddit shard 3, CC3M validation (3 shards), WikiArt (shard 70),
         ImageNet-R (shard 4)                 -> original, jpeg75, jpeg50, half, sns
Outputs in /kaggle/working: meta.csv and feats/<set>__<condition>.npy (rows follow meta order).
"""

import csv
import hashlib
import io
import random
import tarfile
import time
import urllib.request
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 64 * 2**20  # tolerate large metadata when decoding sources

TMP = Path("/tmp/factory")
OUT = Path("/kaggle/working")
OPENFAKE_TRAIN = [
    "train-00000-of-00032-00000", "train-00005-of-00032-00003", "train-00011-of-00032-00005",
    "train-00016-of-00032-00007", "train-00022-of-00032-00010", "train-00027-of-00032-00012",
    "train-00000-of-00032-00010", "train-00005-of-00032-00018", "train-00011-of-00032-00008",
    "train-00016-of-00032-00017", "train-00022-of-00032-00006", "train-00027-of-00032-00016",
]
TRAIN_CONDS = ["original", "jpeg", "sns"]
TEST_CONDS = ["original", "jpeg75", "jpeg50", "half", "sns"]
SMOKE = False  # True: a few minutes end to end, to check the pipeline before a full run
# attached: read crops from the private datasets synth-crops (data/crops) and synth-extra
#           (data/extra_crops), made on a laptop, so the GPU session only extracts features.
# download: fetch and crop every source inside the job (slow: about 1 h of CPU work).
MODE = "attached"
ONLY = [("train", "original"), ("train", "sns"), ("test", "original"), ("test", "jpeg50"), ("test", "sns")]  # rerun just these jobs; None = all
COMPARE = True  # run COMPARE_PLAN (other encoders) on a subset instead of the SigLIP2 job
COMPARE_KEEP = {"train": 15000, "test": 6000}
# GPU -> [(encoder, [(set, condition), ...])]; experiment 19 used train 2,500 / test 1,500 with
# PE-G, PE-L and EmbeddingGemma 2 (train: original + jpeg; test: original, jpeg75, sns).
COMPARE_PLAN = {
    0: [("pe_g", [("train", "original"), ("test", "original"), ("test", "sns")])],
    1: [("pe_g", [("train", "sns"), ("test", "jpeg75")])],
}
QUICK = False  # attached mode: subsample (6k train, 1k val, 4k test) for a ~20 min check


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------- phase A: download and crop (CPU) ----------------


def crop(job):
    """(bytes, key, out_path) -> same deterministic crop as scripts/prepare_data.py."""
    data, key, out = job
    from PIL import Image, ImageOps

    try:
        with Image.open(io.BytesIO(data)) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
    except Exception:
        return False
    w, h = image.size
    size = min(512, w, h) // 2 * 2
    if size < 256:
        return False
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    left, top = seed % (w - size + 1), (seed // 7919) % (h - size + 1)
    # Rebuild from pixels: generator PNGs carry huge text chunks (prompts, workflows) in .info.
    Image.fromarray(np.asarray(image.crop((left, top, left + size, top + size)))).save(out)
    return True


def hf_parquet(repo, name, columns):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, name, repo_type="dataset", local_dir=TMP / "raw")
    parquet = pq.ParquetFile(path)
    for g in range(parquet.num_row_groups):
        yield parquet.read_row_group(g, columns=columns).to_pylist()
    Path(path).unlink()


def sources():
    """Yields (set, source, label, model, batches of (bytes, key))."""
    limit = 200 if SMOKE else None

    def openfake(config, shard, set_name):
        def gen():
            n = 0
            for rows in hf_parquet("ComplexDataLab/OpenFake", f"{config}/{shard}.parquet", ["image", "row_id", "label", "model"]):
                yield [((r["image"]["bytes"], r["row_id"]), int(r["label"] == "fake"), r["model"]) for r in rows]
                n += len(rows)
                if limit and n >= limit:
                    return
        return (set_name, f"openfake_{config}", gen)

    def hf_images(repo, shard, set_name, source, take=None):
        def gen():
            n = 0
            for rows in hf_parquet(repo, shard, ["image"]):
                yield [((r["image"]["bytes"], f"{source}/{shard}/{n + i}"), 0, source) for i, r in enumerate(rows)]
                n += len(rows)
                if (take and n >= take) or (limit and n >= limit):
                    return
        return (set_name, source, gen)

    def coco():
        path = TMP / "val2017.zip"
        urllib.request.urlretrieve("http://images.cocodataset.org/zips/val2017.zip", path)
        z = zipfile.ZipFile(path)
        names = sorted(n for n in z.namelist() if n.endswith(".jpg"))[: limit or 5000]
        for i in range(0, len(names), 256):
            yield [((z.read(n), f"coco/{Path(n).name}"), 0, "coco") for n in names[i : i + 256]]

    def cc3m(shard, set_name):
        def gen():
            from huggingface_hub import hf_hub_download

            path = hf_hub_download("pixparse/cc3m-wds", shard, repo_type="dataset", local_dir=TMP / "raw")
            batch = []
            with tarfile.open(path) as tar:
                for m in tar:
                    if m.name.endswith(".jpg"):
                        batch.append(((tar.extractfile(m).read(), f"cc3m/{m.name}"), 0, "cc3m"))
                    if len(batch) == 256:
                        yield batch
                        batch = []
                        if limit:
                            break
            if batch:
                yield batch
            Path(path).unlink()
        return (set_name, "cc3m", gen)

    out = [openfake("core", s, "train") for s in (OPENFAKE_TRAIN[:1] if SMOKE else OPENFAKE_TRAIN)]
    out += [("train", "coco", coco)]
    out += [hf_images("huggan/wikiart", f"data/train-{i:05d}-of-00072.parquet", "train", "wikiart") for i in (0, 10, 20, 30, 40)]
    out += [hf_images("axiong/imagenet-r", f"test/test-{i:05d}-of-00005.parquet", "train", "imagenet_r") for i in (0, 1)]
    out += [cc3m(f"cc3m-train-{i:04d}.tar", "train") for i in (0, 1)]
    out += [openfake("core", "validation-00000-of-00015", "val")]
    out += [openfake("core", "test-00000-of-00013", "test"), openfake("reddit", "test-00003-of-00005", "test")]
    out += [cc3m(f"cc3m-validation-{i:04d}.tar", "test") for i in (0, 1, 2)]
    out += [hf_images("huggan/wikiart", "data/train-00070-of-00072.parquet", "test", "wikiart"),
            hf_images("axiong/imagenet-r", "test/test-00004-of-00005.parquet", "test", "imagenet_r", take=1500)]
    return out


def crop_roots():
    """Where the crops are. They are uploaded as tar archives named *.bin so Kaggle keeps them packed
    (it unpacks *.tar itself, into tens of thousands of files on a slow network mount); they are
    extracted to local disk here. An unpacked layout (crops_train/train/...) also works.
    The dataset is a network mount where every file lookup is slow, so never walk it recursively:
    look only at the dataset folder and its direct subfolders."""
    import glob

    inputs = Path("/kaggle/input")
    index = Path(next(p for d in range(1, 5) for p in glob.glob(str(inputs) + "/*" * d + "/index.csv")))
    dataset = index.parent
    local = TMP / "unpacked"
    for archive in sorted([*dataset.glob("*.tar"), *dataset.glob("*.bin")]):  # .bin: Kaggle leaves it packed
        dest = local / ("crops" if archive.name.startswith("crops_") else "extra")
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(dest)
        log(f"unpacked {archive.name}")
    candidates = [local / "crops", local / "extra", dataset] + sorted(p for p in dataset.iterdir() if p.is_dir())
    log(f"crop folders: {[c.name for c in candidates]}")

    def root_of(rel):  # first folder that actually holds this file (folder names can repeat)
        return next(c for c in candidates if (c / rel).exists())

    return root_of, index, dataset / "extra_index.csv"


def from_datasets(keep=None):
    """Meta for crops attached as Kaggle datasets; human art and CC3M are split 50/50 train/test."""
    root_of, index, extra_index = crop_roots()
    roots = {}

    def where(rel):
        top = rel.split("/")[0]
        if top not in roots:
            roots[top] = root_of(rel)
        return str(roots[top] / rel)

    meta = []
    for r in csv.DictReader(index.open()):
        real_extra = r["model"] in ("coco", "flickr30k")
        set_name = "train" if r["split"] == "train" or real_extra else {"validation": "val", "test": "test"}.get(r["split"])
        if set_name:
            source = r["model"] if real_extra else "openfake_core"
            meta.append({"set": set_name, "source": source, "label": r["label"], "model": r["model"], "file": where(r["file"])})
    for r in csv.DictReader(extra_index.open()):
        half = int(hashlib.sha256(r["file"].encode()).hexdigest()[:8], 16) % 2
        meta.append({"set": "train" if half == 0 else "test", "source": r["source"], "label": "0", "model": r["source"], "file": where(r["file"])})
    missing = sum(not Path(m["file"]).exists() for m in meta[:: max(1, len(meta) // 500)])
    if missing:
        raise RuntimeError(f"{missing} of ~500 sampled crop files are missing")
    keep = keep or ({"train": 6000, "val": 1000, "test": 4000} if QUICK else None)
    if keep:
        rng = random.Random(0)
        meta = [m for s in keep for m in rng.sample([m for m in meta if m["set"] == s], min(keep[s], sum(x["set"] == s for x in meta)))]
    with (OUT / "meta.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, ["set", "source", "label", "model", "file"])
        w.writeheader()
        w.writerows(meta)
    log(f"attached crops: {len(meta)} rows")
    return meta


def phase_a():
    TMP.mkdir(parents=True, exist_ok=True)
    meta = []
    with Pool(4) as pool:
        for set_name, source, gen in sources():
            folder = TMP / "crops" / set_name
            folder.mkdir(parents=True, exist_ok=True)
            for batch in gen():
                jobs = []
                for (data, key), label, model in batch:
                    out = folder / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.png"
                    jobs.append((data, key, str(out)))
                for ok, ((_, key), label, model), job in zip(pool.imap(crop, jobs, 8), batch, jobs):
                    if ok:
                        meta.append({"set": set_name, "source": source, "label": label, "model": model, "file": job[2]})
            log(f"{set_name}/{source}: {len(meta)} crops so far")
    keys = ["set", "source", "label", "model", "file"]
    with (OUT / "meta.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(meta)
    return meta


# ---------------- phase B: features (one process per GPU) ----------------


def degrade(image, cond, seed):
    from PIL import Image

    rng = random.Random(seed)

    def jpeg(im, q):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q)
        return Image.open(buf).convert("RGB")

    if cond == "original":
        return image
    if cond == "jpeg":
        return jpeg(image, rng.randint(50, 95))
    if cond == "sns":
        s = rng.uniform(0.5, 1.0) if seed is not None else 0.75
        image = image.resize((max(64, int(image.width * s)), max(64, int(image.height * s))), Image.BICUBIC)
        return jpeg(image, rng.randint(60, 90) if seed is not None else 80)
    if cond.startswith("jpeg"):
        return jpeg(image, int(cond[4:]))
    if cond == "half":
        return image.resize((image.width // 2, image.height // 2), Image.BICUBIC)
    raise ValueError(cond)


class Images:
    """Degraded crops resized to the encoder's input size and normalised; size None gives PIL."""

    def __init__(self, files, cond, fixed, size=384, mean=(0.5,) * 3, std=(0.5,) * 3):
        self.files, self.cond, self.fixed = files, cond, fixed
        self.size, self.mean, self.std = size, np.array(mean, np.float32), np.array(std, np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import torch
        from PIL import Image

        f = self.files[i]
        seed = None if self.fixed else int(hashlib.sha256(f"{f}{self.cond}".encode()).hexdigest()[:8], 16)
        size = self.size or 384
        try:
            image = degrade(Image.open(f).convert("RGB"), self.cond, seed)
        except Exception as error:  # one unreadable file must not stop the run; it is flagged
            print(f"skip {f}: {error!r}", flush=True)
            return (Image.new("RGB", (size, size)) if self.size is None else torch.zeros(3, size, size)), False
        if self.size is None:
            return image, True
        a = (np.asarray(image.resize((size, size), Image.BICUBIC), np.float32) / 255 - self.mean) / self.std
        return torch.from_numpy(a.transpose(2, 0, 1)), True


class Encoder:
    """Frozen image encoder -> mean and std of the tokens at a middle and the last block, plus the
    pooled embedding (when the model has one)."""

    def __init__(self, name, gpu):
        import torch

        self.name, self.device, self.captured = name, f"cuda:{gpu}", {}
        self.size, self.mean, self.std = 384, (0.5,) * 3, (0.5,) * 3
        if name == "siglip2":
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384").vision_model
            blocks = self.model.encoder.layers
        elif name.startswith("pe_"):
            import timm

            arch = {"pe_l": "vit_pe_core_large_patch14_336.fb", "pe_g": "vit_pe_core_gigantic_patch14_448.fb"}[name]
            self.model = timm.create_model(arch, pretrained=True, num_classes=0)
            cfg = timm.data.resolve_model_data_config(self.model)
            self.size, self.mean, self.std = cfg["input_size"][-1], cfg["mean"], cfg["std"]
            blocks = self.model.blocks
        elif name == "embgemma2":  # retrieval embedding only, through sentence-transformers
            from sentence_transformers import SentenceTransformer

            self.model, self.size = SentenceTransformer("google/embeddinggemma-2", device=self.device), None
            return
        else:
            raise ValueError(name)
        for i in (len(blocks) // 2 - 1, len(blocks) - 1):
            blocks[i].register_forward_hook(lambda m, a, o, i=i: self.captured.__setitem__(i, o[0] if isinstance(o, tuple) else o))
        self.model = self.model.half().eval().to(self.device)
        self.torch = torch

    def __call__(self, x):
        if self.size is None:
            return np.asarray(self.model.encode(x, batch_size=len(x), convert_to_numpy=True), np.float32)
        torch = self.torch
        out = self.model(pixel_values=x.to(self.device).half()) if self.name == "siglip2" else self.model(x.to(self.device).half())
        pooled = (out.pooler_output if self.name == "siglip2" else out).float()
        mid, last = (self.captured[k].float() for k in sorted(self.captured))
        return torch.cat([mid.mean(1), mid.std(1), last.mean(1), last.std(1), pooled], 1).cpu().numpy()


def load_vision(gpu):
    return Encoder("siglip2", gpu)


def batches(dataset, size=32, threads=3):
    """Yield (images, ok) batches, decoding the next batch in threads while the GPU works.
    Threads, not DataLoader worker processes: forking next to a second GPU thread was refused
    ("os.fork is unsafe while filelock is changing descriptor ownership") and killed a worker."""
    from concurrent.futures import ThreadPoolExecutor

    import torch

    starts = range(0, len(dataset), size)
    with ThreadPoolExecutor(threads) as pool:
        submit = lambda a: [pool.submit(dataset.__getitem__, i) for i in range(a, min(a + size, len(dataset)))]
        pending = submit(0)
        for a in starts:
            ready = [f.result() for f in pending]
            pending = submit(a + size) if a + size < len(dataset) else []
            images = [x for x, _ in ready]
            yield (torch.stack(images) if torch.is_tensor(images[0]) else images), torch.tensor([ok for _, ok in ready])


def phase_b(encoder, jobs):
    """jobs: (set, condition); writes feats/<set>__<condition>.npy (prefixed by the encoder name
    unless it is SigLIP2, for compatibility with earlier runs)."""
    import torch

    meta = list(csv.DictReader((OUT / "meta.csv").open()))
    (OUT / "feats").mkdir(exist_ok=True)
    prefix = "" if encoder.name == "siglip2" else f"{encoder.name}__"
    for set_name, cond in jobs:
        files = [m["file"] for m in meta if m["set"] == set_name]
        out, ok, begin = [], [], time.time()
        with torch.inference_mode():
            for x, good in batches(Images(files, cond, set_name != "train", encoder.size, encoder.mean, encoder.std)):
                ok.append(good.numpy())
                out.append(encoder(x).astype(np.float16))
        np.save(OUT / "feats" / f"{prefix}{set_name}__{cond}.npy", np.concatenate(out))
        np.save(OUT / "feats" / f"{prefix}{set_name}__{cond}__ok.npy", np.concatenate(ok))
        log(f"{encoder.device} {encoder.name} {set_name}/{cond}: {len(files)} images {time.time() - begin:.0f}s")


def compare():
    """Run COMPARE_PLAN on a COMPARE_KEEP subset. Encoders load before any thread starts; one that
    fails to load or run is logged and skipped, the others still run."""
    import subprocess
    import sys
    import threading

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "timm", "sentence-transformers"], check=False)
    from_datasets(keep=COMPARE_KEEP)
    encoders = {}
    for g, items in COMPARE_PLAN.items():
        for name, _ in items:
            try:
                encoders[(name, g)] = Encoder(name, g)
            except Exception as error:
                log(f"skip encoder {name}: {error!r}")

    def run(g):
        for name, jobs in COMPARE_PLAN[g]:
            if (name, g) in encoders:
                try:
                    phase_b(encoders[(name, g)], jobs)
                except Exception as error:
                    log(f"{name} failed: {error!r}")

    threads = [threading.Thread(target=run, args=(g,)) for g in COMPARE_PLAN]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("done")


def main():
    import threading

    if COMPARE:
        compare()
        return

    from_datasets() if MODE == "attached" else phase_a()
    jobs = [("train", c) for c in TRAIN_CONDS] + [("val", "original")] + [("test", c) for c in TEST_CONDS]
    jobs = [j for j in jobs if ONLY is None or j in ONLY]
    # One thread per GPU (Kaggle runs scripts as notebooks, so no re-executable __file__).
    # Load both models before any worker forks: a fork during a Hugging Face download (file lock
    # in flight in the other thread) is refused by filelock and killed one GPU worker.
    visions = [load_vision(g) for g in (0, 1)]
    errors = []

    def run(g):
        try:
            phase_b(visions[g], jobs[g::2])
        except Exception as error:
            errors.append(error)
            raise

    threads = [threading.Thread(target=run, args=(g,)) for g in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError(f"feature workers failed: {errors!r}")
    log("done")


if __name__ == "__main__":
    main()
