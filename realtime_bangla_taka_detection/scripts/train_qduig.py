"""Train Q-DUIG-CAMVA. Supports --seed --config --output-dir --resume."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboeye.camva.notes import load_splits
from roboeye.qduig.artifacts import init_run, run_dir, save_json
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import build_model, set_seed, train_qduig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "proposed.yaml"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    splits, records = load_splits()
    out = Path(args.output_dir) if args.output_dir else run_dir(cfg.get("name", "qduig"), args.seed)
    init_run(
        out,
        cfg,
        args.seed,
        title=f"Q-DUIG train {cfg.get('name')} seed {args.seed}",
        notes="Note-disjoint JaalTaka split. Test labels unused during training.",
    )
    shutil.copy2(ROOT / "results" / "camva" / "splits" / "split_metadata.json", out / "dataset_manifest.json")
    set_seed(args.seed)
    model = build_model(cfg)
    summary = train_qduig(
        model,
        splits["train"],
        splits["val"],
        records,
        config=cfg,
        seed=args.seed,
        output_dir=out,
        resume=Path(args.resume) if args.resume else None,
    )
    save_json(out / "train_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
