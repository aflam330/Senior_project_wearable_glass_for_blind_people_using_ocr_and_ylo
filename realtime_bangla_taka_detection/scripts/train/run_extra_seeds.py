"""Seeds 43 and 44 for the main seed-42-only methods, with the exact seed-42 recipes.

Chains (configs/v2/):
  mtpt_prefix (6 ep, from scratch) -> mtpt_prefix_ft (3 ep, resumed)
  vcie (6 ep) -> vcie_k1 (4 ep, resumed; keeps an epoch only if VAL 1-view accuracy rises)
  apc, cris, mavt, savs (single stage)
Checkpoints are chosen on validation inside train_novel.py. Each final model is scored once on
test with eval_novel.py into results/novel_v2/<name>/seed<s>/test/. A step whose output exists is
skipped, so the script can be restarted.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
R = ROOT / "results" / "novel_v2"
CHAINS = [
    [("mtpt_prefix", "mtpt", None), ("mtpt_prefix_ft", "mtpt", "mtpt_prefix")],
    [("vcie", "vcie", None), ("vcie_k1", "vcie", "vcie")],
    [("apc", "apc", None)], [("cris", "cris", None)], [("mavt", "mavt", None)], [("savs", "savs", None)],
]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"   done in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [43, 44]
    for seed in seeds:
        for chain in CHAINS:
            for name, algo, parent in chain:
                out = R / name / f"seed{seed}"
                cfg = f"configs/v2/{name}.yaml"
                if not (out / "test" / "test_metrics.json").is_file():
                    if not (out / "checkpoint.pt").is_file() or not (out / "val_metrics.json").is_file():
                        cmd = [PY, "scripts/train/train_novel.py", "--algo", algo, "--config", cfg,
                               "--seed", str(seed), "--output-dir", str(out)]
                        if parent:
                            cmd += ["--resume", str(R / parent / f"seed{seed}" / "checkpoint.pt")]
                        run(cmd)
                    run([PY, "scripts/eval/eval_novel.py", "--algo", algo, "--config", cfg, "--seed", str(seed),
                         "--checkpoint", str(out / "checkpoint.pt"), "--split", "test", "--output-dir", str(out / "test")])


if __name__ == "__main__":
    main()
