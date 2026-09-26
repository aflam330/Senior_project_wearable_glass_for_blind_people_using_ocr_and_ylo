"""Train then evaluate all ten algorithms, seed 42, one after another."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
ALGOS = ["ogpd", "vcie", "apc", "sfaq", "igcr", "ugf", "ndal", "sfpl", "cvs", "mtpt"]


def main() -> None:
    for algo in ALGOS:
        cfg = ROOT / "configs" / f"{algo}.yaml"
        out = ROOT / "results" / "novel" / algo / "seed42"
        subprocess.run([
            PY, str(ROOT / "scripts" / "train" / "train_novel.py"),
            "--algo", algo, "--seed", "42", "--config", str(cfg), "--output-dir", str(out),
        ], cwd=ROOT, check=True)
        subprocess.run([
            PY, str(ROOT / "scripts" / "eval" / "eval_novel.py"),
            "--algo", algo, "--seed", "42", "--config", str(cfg),
            "--checkpoint", str(out / "checkpoint.pt"),
            "--split", "test", "--output-dir", str(out / "test"),
        ], cwd=ROOT, check=True)
    print("all ten novel algorithms finished", flush=True)


if __name__ == "__main__":
    main()
