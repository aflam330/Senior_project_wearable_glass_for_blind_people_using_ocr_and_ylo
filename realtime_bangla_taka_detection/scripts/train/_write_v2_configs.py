"""Write v2 configs and thin entry points. Safe to run more than once."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPECS = {
    "vcie": {"epochs": 6, "patience": 2, "warmup_steps": 100},
    "sfpl": {"epochs": 4, "patience": 2, "warmup_steps": 0},
    "mtpt": {"epochs": 6, "patience": 2, "warmup_steps": 100},
    "apc": {"epochs": 8, "patience": 3, "warmup_steps": 0},
    "ogpd": {"epochs": 6, "patience": 2, "warmup_steps": 0},
    "ndal": {"epochs": 6, "patience": 2, "warmup_steps": 0},
    "cvs": {"epochs": 4, "patience": 2, "warmup_steps": 0},
    "pravt": {"epochs": 4, "patience": 2, "warmup_steps": 0},
    "cris": {"epochs": 4, "patience": 2, "warmup_steps": 0},
    "savs": {"epochs": 3, "patience": 2, "warmup_steps": 0},
    "mavt": {"epochs": 4, "patience": 2, "warmup_steps": 0},
    "vat": {"epochs": 4, "patience": 2, "warmup_steps": 0},
}
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
    for algo, extra in SPECS.items():
        lines = [
            f"name: {algo}",
            f"algo: {algo}",
            f"epochs: {extra['epochs']}",
            "batch: 4",
            "lr: 0.001",
            "seed: 42",
            "views: 6",
            f"patience: {extra['patience']}",
            f"warmup_steps: {extra['warmup_steps']}",
            "note: v2 run. Train and validation only.",
        ]
        dest = ROOT / "configs" / "v2"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{algo}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (ROOT / "scripts" / "train" / f"train_{algo}.py").write_text(TRAIN.format(name=algo.upper(), algo=algo), encoding="utf-8")
        (ROOT / "scripts" / "eval" / f"eval_{algo}.py").write_text(EVAL.format(name=algo.upper(), algo=algo), encoding="utf-8")
    print("configs", len(SPECS))


if __name__ == "__main__":
    main()
