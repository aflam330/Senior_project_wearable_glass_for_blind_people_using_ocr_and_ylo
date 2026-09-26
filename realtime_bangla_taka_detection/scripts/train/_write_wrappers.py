"""Generate thin per-algorithm train and eval entry points."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALGOS = ["ogpd", "vcie", "apc", "sfaq", "igcr", "ugf", "ndal", "sfpl", "cvs", "mtpt"]
TRAIN = '''"""Train {name}."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "train" / "train_novel.py"), "--algo", "{algo}", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
'''
EVAL = '''"""Evaluate {name}."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "eval" / "eval_novel.py"), "--algo", "{algo}", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
'''


def main() -> None:
    for algo in ALGOS:
        (ROOT / "scripts" / "train" / f"train_{algo}.py").write_text(TRAIN.format(name=algo.upper(), algo=algo), encoding="utf-8")
        (ROOT / "scripts" / "eval" / f"eval_{algo}.py").write_text(EVAL.format(name=algo.upper(), algo=algo), encoding="utf-8")
    print("wrappers", len(ALGOS))


if __name__ == "__main__":
    main()
