"""Bundle a Kaggle job with the repo code it imports, then push and run it on a T4.

Kaggle runs one script file, so the repo modules (and any other file in the job folder, e.g.
teacher scores) are embedded in it and unpacked at start-up. The feature job
kaggle/siglip_features/main.py is importable as `factory`.
Usage: python kaggle/push.py kaggle/distill [--smoke] [--set NAME=VALUE ...] [--suffix S]
  --smoke   sets SMOKE = True and pushes to a separate "<id>-smoke" notebook.
  --set     replaces the value of a top-level constant (NAME = ...) in main.py, e.g. PART="eval".
  --suffix  pushes to "<id>-<S>" (one notebook per variant).
Jobs whose kernel-metadata.json has "enable_gpu": false run on CPU (no GPU quota).
"""

import base64
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODULES = {  # unpacked name -> repo path
    "src/detector/__init__.py": "src/detector/__init__.py",
    "src/detector/features.py": "src/detector/features.py",
    "scripts/factory.py": "kaggle/siglip_features/main.py",
    "scripts/distill.py": "kaggle/distill/main.py",
}
HEADER = """import base64, pathlib, sys
for _name, _data in {files!r}.items():
    _path = pathlib.Path("/tmp/code") / _name
    _path.parent.mkdir(parents=True, exist_ok=True)
    _path.write_bytes(base64.b64decode(_data))
sys.path[:0] = ["/tmp/code/src", "/tmp/code/scripts"]
JOB = pathlib.Path("/tmp/code/job")
"""


def main(job: Path, smoke: bool, settings: dict[str, str], suffix: str | None) -> None:
    files = {
        name: base64.b64encode((REPO / path).read_bytes()).decode()
        for name, path in MODULES.items()
    }
    for extra in sorted(job.iterdir()):
        if extra.is_file() and extra.name not in ("main.py", "kernel-metadata.json"):
            files[f"job/{extra.name}"] = base64.b64encode(extra.read_bytes()).decode()
    code = (job / "main.py").read_text()
    if smoke:
        settings = {**settings, "SMOKE": "True"}
    for name, value in settings.items():
        code, n = re.subn(rf"^{name} = .*$", f"{name} = {value}", code, count=1, flags=re.M)
        if not n:
            raise SystemExit(f"{name} is not a top-level constant in {job / 'main.py'}")
    meta = json.loads((job / "kernel-metadata.json").read_text())
    for tag in ([suffix] if suffix else []) + (["smoke"] if smoke else []):
        meta["id"] += f"-{tag}"
        meta["title"] += f"-{tag}"
    command = ["kaggle", "kernels", "push", "-p"]
    accelerator = ["--accelerator", "NvidiaTeslaT4"] if meta.get("enable_gpu", True) else []
    with tempfile.TemporaryDirectory() as folder:
        Path(folder, meta["code_file"]).write_text(HEADER.format(files=files) + code)
        Path(folder, "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        subprocess.run([*command, folder, *accelerator], check=True)


if __name__ == "__main__":
    args = sys.argv[2:]
    pairs = [args[i + 1] for i, a in enumerate(args) if a == "--set"]
    suffix = next((args[i + 1] for i, a in enumerate(args) if a == "--suffix"), None)
    main(Path(sys.argv[1]), "--smoke" in args, dict(p.split("=", 1) for p in pairs), suffix)
