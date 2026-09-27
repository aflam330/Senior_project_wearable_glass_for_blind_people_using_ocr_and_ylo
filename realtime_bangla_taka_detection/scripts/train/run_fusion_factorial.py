"""Two new prefix 6+3 cells. Does not overwrite full PRMVT or her_base."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
TRAIN = ROOT / "scripts" / "train_qduig.py"
EVAL = ROOT / "scripts" / "eval_prefix_views.py"

JOBS = [
    ("fusion_meanpool_aux", "fusion_meanpool_aux_ft", None),
    ("fusion_rsqa_noaux", "fusion_rsqa_noaux_ft", None),
]


def run(cmd):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NOVEL_WORKERS"] = "0"
    env["NOVEL_COMPILE"] = "0"
    print("CMD", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def main():
    for stage, ft, _ in JOBS:
        stage_dir = ROOT / "results" / "qduig" / stage / "seed42"
        ft_dir = ROOT / "results" / "qduig" / ft / "seed42"
        views = ft_dir / "test_views" / "views_1_to_6.json"
        if views.is_file():
            print("skip", ft, flush=True)
            continue
        if not (stage_dir / "checkpoint.pt").is_file():
            run([PY, str(TRAIN), "--config", str(ROOT / "configs" / f"{stage}.yaml"), "--seed", "42", "--output-dir", str(stage_dir)])
        if not (ft_dir / "checkpoint.pt").is_file():
            run([
                PY, str(TRAIN), "--config", str(ROOT / "configs" / f"{ft}.yaml"), "--seed", "42",
                "--output-dir", str(ft_dir), "--resume", str(stage_dir / "checkpoint.pt"),
            ])
        run([
            PY, str(EVAL), "--checkpoint", str(ft_dir / "checkpoint.pt"),
            "--config", str(ROOT / "configs" / f"{ft}.yaml"), "--split", "test", "--seed", "42",
            "--output-dir", str(ft_dir / "test_views"),
        ])
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
