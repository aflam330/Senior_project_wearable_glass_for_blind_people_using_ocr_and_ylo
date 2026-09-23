"""Evaluate one Q-DUIG checkpoint. Supports --checkpoint --split --seed --output-dir."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboeye.camva.engine import make_loader
from roboeye.camva.notes import load_splits
from roboeye.qduig.artifacts import init_run, save_json
from roboeye.qduig.config_io import load_config
from roboeye.config import DEVICE
from roboeye.qduig.engine import load_qduig, pack_and_save, predict_qduig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=str(ROOT / "configs" / "proposed.yaml"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--views", type=int, default=6)
    args = p.parse_args()

    cfg = load_config(args.config)
    splits, records = load_splits()
    out = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / f"eval_{args.split}_{args.views}view"
    init_run(out, cfg, args.seed, f"eval {args.split}", f"views={args.views}")
    model = load_qduig(Path(args.checkpoint), cfg)
    loader = make_loader(splits[args.split], records, n_views=args.views, train=False, batch=8)
    pred = predict_qduig(model, loader, DEVICE)
    m = pack_and_save(pred, f"qduig_{args.split}_{args.views}view", out)
    save_json(out / "eval_metrics.json", m)
    print(m)


if __name__ == "__main__":
    main()
