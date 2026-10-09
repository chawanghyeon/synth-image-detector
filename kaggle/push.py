"""Bundle a Kaggle job with the repo code it imports, then push and run it on a T4.

Kaggle runs one script file, so the repo modules (and any other file in the job folder, e.g. teacher
scores) are embedded in it and unpacked at start-up. The feature job kaggle/siglip_features/main.py is
importable as `factory`.
Usage: python kaggle/push.py kaggle/noisenet_scaling [--smoke]
  --smoke  sets SMOKE = True and pushes to a separate "<id>-smoke" notebook.
"""

import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODULES = {  # unpacked name -> repo path
    "src/detector/__init__.py": "src/detector/__init__.py",
    "src/detector/features.py": "src/detector/features.py",
    "scripts/train_spam.py": "scripts/train_spam.py",
    "scripts/train_noisenet.py": "scripts/train_noisenet.py",
    "scripts/factory.py": "kaggle/siglip_features/main.py",
}
HEADER = '''import base64, pathlib, sys
for _name, _data in {files!r}.items():
    _path = pathlib.Path("/tmp/code") / _name
    _path.parent.mkdir(parents=True, exist_ok=True)
    _path.write_bytes(base64.b64decode(_data))
sys.path[:0] = ["/tmp/code/src", "/tmp/code/scripts"]
JOB = pathlib.Path("/tmp/code/job")
'''


def main(job: Path, smoke: bool) -> None:
    files = {name: base64.b64encode((REPO / path).read_bytes()).decode() for name, path in MODULES.items()}
    for extra in sorted(job.iterdir()):
        if extra.is_file() and extra.name not in ("main.py", "kernel-metadata.json"):
            files[f"job/{extra.name}"] = base64.b64encode(extra.read_bytes()).decode()
    code = (job / "main.py").read_text()
    if smoke:
        code = code.replace("\nSMOKE = False", "\nSMOKE = True", 1)
    meta = json.loads((job / "kernel-metadata.json").read_text())
    if smoke:
        meta["id"] += "-smoke"
        meta["title"] += "-smoke"
    with tempfile.TemporaryDirectory() as folder:
        Path(folder, meta["code_file"]).write_text(HEADER.format(files=files) + code)
        Path(folder, "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        subprocess.run(["kaggle", "kernels", "push", "-p", folder, "--accelerator", "NvidiaTeslaT4"], check=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), "--smoke" in sys.argv[2:])
