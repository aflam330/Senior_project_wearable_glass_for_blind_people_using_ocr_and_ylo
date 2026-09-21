"""Wait for CAMVA checkpoints, then run the full evaluation chain."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "results" / "camva" / "checkpoints"
PY = sys.executable
SCRIPTS = ROOT / "scripts"


def wait_file(path: Path, label: str) -> None:
    print("waiting for", label, path, flush=True)
    while not path.is_file():
        time.sleep(30)
    print("found", path, "size", path.stat().st_size, flush=True)


def run(script: str) -> None:
    cmd = [PY, "-u", str(SCRIPTS / script)]
    print("RUN", script, flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    wait_file(CKPT / "baseline_cnnvit_seed42.pt", "baseline")
    wait_file(CKPT / "camva_quality_attention_seed42.pt", "camva")
    for name in (
        "evaluate_camva.py",
        "calibrate_camva.py",
        "evaluate_ablation.py",
        "evaluate_adaptive.py",
        "evaluate_camva_robustness.py",
        "write_camva_report.py",
    ):
        run(name)
    print("CAMVA eval chain finished", flush=True)


if __name__ == "__main__":
    main()
