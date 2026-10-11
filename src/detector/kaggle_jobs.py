"""Push Kaggle jobs (kaggle/<job>/main.py + kernel-metadata.json) and collect their outputs.

Kaggle runs a single script, so the whole detector package is embedded in it (base64) and
unpacked to /tmp/code/src at start-up; the job's main.py then imports detector.* as usual.
Needs the kaggle command line tool, logged in (`kaggle auth login`).
"""

import base64
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
HEADER = """import base64, pathlib, sys
for _name, _data in {files!r}.items():
    _path = pathlib.Path("/tmp/code/src") / _name
    _path.parent.mkdir(parents=True, exist_ok=True)
    _path.write_bytes(base64.b64decode(_data))
sys.path.insert(0, "/tmp/code/src")
"""
FOOTER = '\n\nif __name__ == "__main__":\n    main()\n'
DONE = ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED")


class KaggleJob:
    """kaggle/<job>: main.py defines top-level constants (overridable by `settings`) and main()."""

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "kernel-metadata.json").read_text())

    def script(self, settings: dict[str, str] | None = None) -> str:
        code = (self.folder / "main.py").read_text()
        for name, value in (settings or {}).items():
            code, n = re.subn(rf"^{name} = .*$", f"{name} = {value}", code, count=1, flags=re.M)
            if not n:
                raise ValueError(f"{name} is not a one-line top-level constant in {self.folder}")
        files = {
            str(p.relative_to(PACKAGE.parent)): base64.b64encode(p.read_bytes()).decode()
            for p in sorted(PACKAGE.rglob("*.py"))
        }
        if "__main__" not in code:
            code += FOOTER
        return HEADER.format(files=files) + code

    def kernel(self, suffix: str | None = None, smoke: bool = False) -> dict:
        meta = dict(self.meta)
        for tag in ([suffix] if suffix else []) + (["smoke"] if smoke else []):
            meta["id"] += f"-{tag}"
            meta["title"] += f"-{tag}"
        return meta

    def push(
        self, settings: dict[str, str] | None = None, suffix: str | None = None, smoke: bool = False
    ) -> str:
        """Push and start the job ("-smoke" notebook with SMOKE = True); returns the kernel id."""
        settings = {**(settings or {}), **({"SMOKE": "True"} if smoke else {})}
        meta = self.kernel(suffix, smoke)
        gpu = ["--accelerator", "NvidiaTeslaT4"] if meta.get("enable_gpu", True) else []
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, meta["code_file"]).write_text(self.script(settings))
            Path(folder, "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
            subprocess.run(["kaggle", "kernels", "push", "-p", folder, *gpu], check=True)
        return meta["id"]


def status(kernel: str) -> str:
    """RUNNING, QUEUED, COMPLETE, ERROR, ... or UNKNOWN when the API call fails."""
    result = subprocess.run(["kaggle", "kernels", "status", kernel], capture_output=True, text=True)
    found = re.search(r"KernelWorkerStatus\.(\w+)", result.stdout)
    return found.group(1) if found else "UNKNOWN"


def wait(kernel: str, poll: int = 120, log=print) -> str:
    last = None
    while (state := status(kernel)) not in DONE:
        if state != last:
            log(f"{time.strftime('%H:%M:%S')} {kernel}: {state}")
            last = state
        time.sleep(poll)
    return state


def fetch(kernel: str, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "kernels", "output", kernel, "-p", str(dest), "-o"],
        check=True,
        capture_output=True,
    )
    return sorted(dest.iterdir())
