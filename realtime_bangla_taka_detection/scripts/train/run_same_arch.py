"""Same-architecture fixed-view vs prefix runs on JaalTaka (PRMVT network, seeds 42-44).

Arms (configs/same_arch/):
  fixed        6 views only, single-view loss off, per-count BN on (BN for k<6 never trained)
  fixed_shbn   6 views only, single-view loss off, one shared BN
  prefix_shbn  mixed prefixes, single-view loss 0.5, one shared BN
The existing results/qduig/prefix_ft/seed* runs are the prefix arm with per-count BN.

Each run: stage 1 (6 epochs) -> stage 2 (3 epochs, resumed) -> forced first-k test eval.
Finished steps are skipped, so the script can be restarted. A stage counts as finished only
when train_summary.json exists: checkpoint.pt is written every epoch, so a killed run leaves one.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
ARMS = ["fixed", "fixed_shbn", "prefix_shbn"]
SEEDS = [42, 43, 44]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"   done in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    arms = sys.argv[1].split(",") if len(sys.argv) > 1 else ARMS
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else SEEDS
    for seed in seeds:
        for arm in arms:
            base = ROOT / "results" / "same_arch" / arm
            s1, s2 = base / "s1" / f"seed{seed}", base / "s2" / f"seed{seed}"
            if not (s1 / "train_summary.json").is_file():
                run([PY, "scripts/train_qduig.py", "--config", f"configs/same_arch/{arm}_s1.yaml",
                     "--seed", str(seed), "--output-dir", str(s1)])
            if not (s2 / "train_summary.json").is_file():
                run([PY, "scripts/train_qduig.py", "--config", f"configs/same_arch/{arm}_s2.yaml",
                     "--seed", str(seed), "--output-dir", str(s2), "--resume", str(s1 / "checkpoint.pt")])
            if not (s2 / "test_views" / "6view" / "test_metrics.json").is_file():
                run([PY, "scripts/eval_prefix_views.py", "--checkpoint", str(s2 / "checkpoint.pt"),
                     "--config", f"configs/same_arch/{arm}_s2.yaml", "--split", "test", "--seed", str(seed),
                     "--output-dir", str(s2 / "test_views")])


if __name__ == "__main__":
    main()
