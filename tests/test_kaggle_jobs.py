import base64
import re
from pathlib import Path

import pytest

from detector.kaggle_jobs import KaggleJob

JOBS = Path(__file__).resolve().parents[1] / "kaggle"


@pytest.mark.parametrize("name", ["data_factory", "train"])
def test_job_script_embeds_package_and_compiles(name):
    script = KaggleJob(JOBS / name).script()
    compile(script, f"{name}.py", "exec")
    embedded = re.search(r"in (\{.*?\})\.items\(\)", script).group(1)
    assert "'detector/cli.py'" in embedded and "'detector/training/job.py'" in embedded
    assert "def main()" in script


def test_settings_replace_constants_and_suffix_names_kernel():
    job = KaggleJob(JOBS / "data_factory")
    script = job.script({"PART": '"train_a"', "SMOKE": "True"})
    assert '\nPART = "train_a"\n' in script and "\nSMOKE = True\n" in script
    with pytest.raises(ValueError):
        job.script({"MISSING": "1"})
    assert job.kernel("train-a", smoke=True)["id"] == job.meta["id"] + "-train-a-smoke"
    assert job.meta["enable_gpu"] is False


def test_embedded_files_are_the_package_sources():
    script = KaggleJob(JOBS / "train").script()
    files = eval(re.search(r"in (\{.*?\})\.items\(\)", script).group(1))
    source = Path(__file__).resolve().parents[1] / "src" / "detector" / "preprocess.py"
    assert base64.b64decode(files["detector/preprocess.py"]) == source.read_bytes()
