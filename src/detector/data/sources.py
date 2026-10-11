"""Where the images come from. Each Source yields batches of encoded images with labels; PARTS
groups them into the data-factory jobs (one Kaggle CPU notebook per part).

  train_a  OpenFake core train shards (12)
  train_b  real images: COCO val2017, WikiArt (5 shards), ImageNet-R (2 shards), CC3M train (2)
  train_c  OpenFake Reddit shards 0-2 (images posted online; shards 3-4 stay sealed)
  train_d  closed/commercial generators only (GPT-Image, DALL-E, Imagen, Midjourney, ...) from
           40 OpenFake core train shards used nowhere else
  eval     sealed sets, never trained on: OpenFake core test 1-2 (new generators), Reddit 3
           (threshold) and 4, hard reals (CC3M validation 0-2, WikiArt 70, ImageNet-R 4)
"""

import tarfile
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

TMP = Path("/tmp/factory")


@dataclass(frozen=True)
class Item:
    data: bytes  # encoded image file
    key: str  # stable id: names the crop and seeds its position
    label: int
    model: str


Batch = list[Item]


class Source(ABC):
    def __init__(self, set_name: str, source: str) -> None:
        self.set_name, self.source = set_name, source

    @abstractmethod
    def batches(self, limit: int | None = None) -> Iterator[Batch]:
        """Batches of items; `limit` stops after about that many (smoke runs)."""


def hf_parquet(repo: str, name: str, columns: list[str]) -> Iterator[list[dict]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, name, repo_type="dataset", local_dir=TMP / "raw")
    parquet = pq.ParquetFile(path)
    for group in range(parquet.num_row_groups):
        yield parquet.read_row_group(group, columns=columns).to_pylist()
    Path(path).unlink()


class OpenFake(Source):
    """ComplexDataLab/OpenFake (CC BY-NC 4.0): config "core" or "reddit"."""

    def __init__(self, config: str, shard: str, set_name: str, keep=None) -> None:
        super().__init__(set_name, f"openfake_{config}")
        self.config, self.shard, self.keep = config, shard, keep

    def batches(self, limit=None):
        n = 0
        columns = ["image", "row_id", "label", "model"]
        for rows in hf_parquet(
            "ComplexDataLab/OpenFake", f"{self.config}/{self.shard}.parquet", columns
        ):
            yield [
                Item(r["image"]["bytes"], r["row_id"], int(r["label"] == "fake"), r["model"])
                for r in rows
                if self.keep is None or self.keep(r)
            ]
            n += len(rows)
            if limit and n >= limit:
                return


class HFImages(Source):
    """Real images from a Hugging Face parquet dataset with an "image" column."""

    def __init__(self, repo: str, shard: str, set_name: str, source: str) -> None:
        super().__init__(set_name, source)
        self.repo, self.shard = repo, shard

    def batches(self, limit=None):
        n = 0
        for rows in hf_parquet(self.repo, self.shard, ["image"]):
            key = f"{self.source}/{self.shard}"
            yield [
                Item(r["image"]["bytes"], f"{key}/{n + i}", 0, self.source)
                for i, r in enumerate(rows)
            ]
            n += len(rows)
            if limit and n >= limit:
                return


class Coco(Source):
    def __init__(self, set_name: str) -> None:
        super().__init__(set_name, "coco")

    def batches(self, limit=None):
        path = TMP / "val2017.zip"
        urllib.request.urlretrieve("http://images.cocodataset.org/zips/val2017.zip", path)
        archive = zipfile.ZipFile(path)
        names = sorted(n for n in archive.namelist() if n.endswith(".jpg"))[: limit or 5000]
        for i in range(0, len(names), 256):
            yield [
                Item(archive.read(n), f"coco/{Path(n).name}", 0, "coco") for n in names[i : i + 256]
            ]


class CC3M(Source):
    """pixparse/cc3m-wds web images (photos, designs, illustrations)."""

    def __init__(self, shard: str, set_name: str) -> None:
        super().__init__(set_name, "cc3m")
        self.shard = shard

    def batches(self, limit=None):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "pixparse/cc3m-wds", self.shard, repo_type="dataset", local_dir=TMP / "raw"
        )
        batch = []
        with tarfile.open(path) as tar:
            for member in tar:
                if member.name.endswith(".jpg"):
                    data = tar.extractfile(member).read()
                    batch.append(Item(data, f"cc3m/{member.name}", 0, "cc3m"))
                if len(batch) == 256:
                    yield batch
                    batch = []
                    if limit:
                        break
        if batch:
            yield batch
        Path(path).unlink()


OPENFAKE_TRAIN = [
    "train-00000-of-00032-00000",
    "train-00005-of-00032-00003",
    "train-00011-of-00032-00005",
    "train-00016-of-00032-00007",
    "train-00022-of-00032-00010",
    "train-00027-of-00032-00012",
    "train-00000-of-00032-00010",
    "train-00005-of-00032-00018",
    "train-00011-of-00032-00008",
    "train-00016-of-00032-00017",
    "train-00022-of-00032-00006",
    "train-00027-of-00032-00016",
]
CLOSED_GENERATORS = (
    "gpt-image", "dall", "imagen", "ideogram", "midjourney", "flux-1.1-pro", "flux.2",
    "seedream", "nano-banana", "grok", "recraft", "mystic", "kling", "firefly", "sora",
)  # fmt: skip


def closed_generator(row: dict) -> bool:
    return row["label"] == "fake" and row["model"].startswith(CLOSED_GENERATORS)


def _train_b() -> list[Source]:
    wikiart = [f"data/train-{i:05d}-of-00072.parquet" for i in (0, 10, 20, 30, 40)]
    imagenet_r = [f"test/test-{i:05d}-of-00005.parquet" for i in (0, 1)]
    return [
        Coco("train"),
        *(HFImages("huggan/wikiart", s, "train", "wikiart") for s in wikiart),
        *(HFImages("axiong/imagenet-r", s, "train", "imagenet_r") for s in imagenet_r),
        *(CC3M(f"cc3m-train-{i:04d}.tar", "train") for i in (0, 1)),
    ]


def _eval() -> list[Source]:
    return [
        OpenFake("core", "test-00001-of-00013", "core_test1"),
        OpenFake("core", "test-00002-of-00013", "core_test2"),
        OpenFake("reddit", "test-00003-of-00005", "reddit3"),
        OpenFake("reddit", "test-00004-of-00005", "reddit4"),
        *(CC3M(f"cc3m-validation-{i:04d}.tar", "hard_real") for i in (0, 1, 2)),
        HFImages("huggan/wikiart", "data/train-00070-of-00072.parquet", "hard_real", "wikiart"),
        HFImages(
            "axiong/imagenet-r", "test/test-00004-of-00005.parquet", "hard_real", "imagenet_r"
        ),
    ]


PARTS: dict[str, Callable[[], list[Source]]] = {
    "train_a": lambda: [OpenFake("core", s, "train") for s in OPENFAKE_TRAIN],
    "train_b": _train_b,
    "train_c": lambda: [OpenFake("reddit", f"test-{i:05d}-of-00005", "train") for i in (0, 1, 2)],
    "train_d": lambda: [
        OpenFake("core", f"train-{i:05d}-of-00032-{j:05d}", "train", keep=closed_generator)
        for i in range(0, 32, 4)
        for j in (1, 2, 4, 6, 9)
    ],
    "eval": _eval,
}
