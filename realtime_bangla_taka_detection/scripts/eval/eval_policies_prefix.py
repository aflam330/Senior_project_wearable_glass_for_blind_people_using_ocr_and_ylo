"""Re-run the 9 view-selection policies on the deployed Q-DUIG prefix_ft checkpoints.

The original policy results (results/qduig/eval/seed42/policies) were measured on
results/qduig/proposed, an early model whose training was unstable (1-view accuracy
~0.59-0.65). This runs the same eval_policies code on the prefix_ft checkpoints that
the glass app uses. Output: results/qduig/eval_prefix_ft/seed<N>/policies. The old
results are left in place.

Usage:
  python scripts/eval/eval_policies_prefix.py            # seeds 42 43 44
  python scripts/eval/eval_policies_prefix.py --seeds 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.camva.notes import load_splits
from roboeye.qduig.artifacts import init_run
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig
from scripts.evaluate_all import eval_policies

QDUIG = ROOT / "results" / "qduig"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--run", default="prefix_ft", help="folder under results/qduig holding seed<N>/checkpoint.pt")
    args = p.parse_args()
    splits, records = load_splits()
    for seed in args.seeds:
        run_dir = QDUIG / args.run / f"seed{seed}"
        ckpt = run_dir / "checkpoint.pt"
        if not ckpt.is_file():
            print(f"missing {ckpt}, skipped", flush=True)
            continue
        cfg = load_config(run_dir / "config.yaml")
        out = QDUIG / f"eval_{args.run}" / f"seed{seed}"
        init_run(out, cfg, seed, f"policies on {args.run} seed {seed}",
                 f"Same eval_policies code as scripts/evaluate_all.py, checkpoint {ckpt.relative_to(ROOT)}. Test split, threshold 0.5.")
        model = load_qduig(ckpt, cfg)
        eval_policies(model, cfg, splits, records, out / "policies", seed)


if __name__ == "__main__":
    main()
