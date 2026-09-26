"""Phase 1: prefix-robust seeds 43 and 44, same protocol as seed 42.

Stage 1: configs/proposed_prefix.yaml, 6 epochs, mixed view-dropout.
Stage 2: configs/proposed_prefix_ft.yaml, 3 epochs, resume from stage-1 checkpoint.
Then one test evaluation of views 1-6. Test labels are not used for training,
checkpoint selection, or calibration.

The earlier seed-43 process stopped after epoch 5. Its last-epoch weights were
not saved (only the best checkpoint). That partial run is kept at
results/qduig/prefix/seed43_stopped_epoch5. This script trains a full 6-epoch
seed 43 into results/qduig/prefix/seed43.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TRAIN = ROOT / "scripts" / "train_qduig.py"
EVAL = ROOT / "scripts" / "eval_prefix_views.py"
PREFIX = ROOT / "configs" / "proposed_prefix.yaml"
FT = ROOT / "configs" / "proposed_prefix_ft.yaml"


def run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    for seed in (43, 44):
        stage1 = ROOT / "results" / "qduig" / "prefix" / f"seed{seed}"
        stage2 = ROOT / "results" / "qduig" / "prefix_ft" / f"seed{seed}"
        run([
            PY, str(TRAIN),
            "--config", str(PREFIX),
            "--seed", str(seed),
            "--output-dir", str(stage1),
        ])
        ckpt = stage1 / "checkpoint.pt"
        if not ckpt.is_file():
            raise SystemExit(f"missing stage-1 checkpoint {ckpt}")
        run([
            PY, str(TRAIN),
            "--config", str(FT),
            "--seed", str(seed),
            "--output-dir", str(stage2),
            "--resume", str(ckpt),
        ])
        ft_ckpt = stage2 / "checkpoint.pt"
        run([
            PY, str(EVAL),
            "--checkpoint", str(ft_ckpt),
            "--config", str(FT),
            "--split", "test",
            "--seed", str(seed),
            "--output-dir", str(stage2 / "test_views"),
        ])
    print("phase1 seeds 43 and 44 finished", flush=True)


if __name__ == "__main__":
    main()
