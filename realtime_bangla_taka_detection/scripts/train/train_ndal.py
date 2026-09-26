"""Train NDAL."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "train" / "train_novel.py"), "--algo", "ndal", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
