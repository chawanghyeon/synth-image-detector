import random
import tarfile

from conftest import noise, png_bytes

from detector.data import degrade as dg
from detector.data.factory import CropFactory, crop_name, native_crop, pack_codex
from detector.data.records import Row, held_aside, read_meta, unpack_parts, write_meta
from detector.data.sources import PARTS, Item, Source, closed_generator


def test_conditions_keep_rgb_and_shrink_where_expected():
    image = noise((400, 300))
    for name in dg.CONDITIONS:
        out = dg.degrade(image, name)
        assert out.mode == "RGB"
    assert dg.degrade(image, "half").size == (200, 150)
    assert dg.degrade(image, "sns").size == (300, 225)
    assert dg.degrade(image, "original") is image


def test_augment_always_degrades():
    image = noise((128, 128))
    rng = random.Random(0)
    assert all(dg.augment(image, rng, always=True) is not image for _ in range(50))
    assert any(dg.augment(image, rng) is image for _ in range(50))


def test_meta_round_trip_and_held_aside(tmp_path):
    rows = [Row("train", "coco", 0, "coco", f"train/{i}.png") for i in range(200)]
    write_meta(tmp_path / "meta.csv", rows)
    assert read_meta(tmp_path / "meta.csv") == rows
    aside = [held_aside(r) for r in rows]
    assert 20 < sum(aside) < 60  # about 20%
    assert not held_aside(Row("train", "x", 1, "gen", "train/0.png"))  # only reals


def test_native_crop_is_deterministic_and_bounded():
    data = png_bytes((900, 600))
    a, b = native_crop(data, "key"), native_crop(data, "key")
    assert a.size == (512, 512) and a.tobytes() == b.tobytes()
    assert native_crop(png_bytes((300, 1000)), "k").size == (300, 300)
    assert native_crop(png_bytes((200, 1000)), "k") is None
    assert native_crop(b"not an image", "k") is None


class FakeSource(Source):
    def __init__(self):
        super().__init__("train", "fake")

    def batches(self, limit=None):
        yield [Item(png_bytes((300, 300)), f"k{i}", i % 2, "gen") for i in range(3)]
        yield [Item(b"junk", "bad", 1, "gen")]


def test_factory_builds_packs_and_unpacks(tmp_path):
    factory = CropFactory(root=tmp_path / "crops", workers=1)
    rows = factory.build([FakeSource()], log=lambda _: None)
    assert [r.file for r in rows] == [f"train/{crop_name(f'k{i}')}" for i in range(3)]
    part = tmp_path / "input" / "data-factory-x"
    part.mkdir(parents=True)
    assert factory.pack(rows, part, "x") == 1
    unpacked = unpack_parts(tmp_path / "input", tmp_path / "data")
    assert [r.label for r in unpacked] == [0, 1, 0]
    assert all(r.file.startswith(str(tmp_path / "data" / "data-factory-x")) for r in unpacked)


def test_pack_codex(tmp_path):
    src = tmp_path / "codex"
    src.mkdir()
    for i in range(2):
        noise((300, 400)).save(src / f"t{i:03d}.png")
    noise((300, 400)).save(src / "other.png")  # not a training image
    out = tmp_path / "ds"
    assert pack_codex(src, out) == 2
    with tarfile.open(out / "crops_codex_00.bin") as tar:
        assert len(tar.getnames()) == 2
    assert {r.source for r in read_meta(out / "meta.csv")} == {"codex"}
    assert not (out / "crops").exists()


def test_parts_and_closed_generator_filter():
    assert set(PARTS) == {"train_a", "train_b", "train_c", "train_d", "eval"}
    sets = {s.set_name for s in PARTS["eval"]()}
    assert sets == {"core_test1", "core_test2", "reddit3", "reddit4", "hard_real"}
    assert len(PARTS["train_d"]()) == 40
    assert closed_generator({"label": "fake", "model": "gpt-image-1"})
    assert not closed_generator({"label": "real", "model": "gpt-image-1"})
    assert not closed_generator({"label": "fake", "model": "sdxl"})
