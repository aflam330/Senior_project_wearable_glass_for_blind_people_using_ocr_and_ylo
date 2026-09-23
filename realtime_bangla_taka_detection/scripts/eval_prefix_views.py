"""Evaluate 1-6 views with the same first-k protocol as the CNN+ViT baseline."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from roboeye.camva.engine import load_baseline, make_loader
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.artifacts import init_run, save_json
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig, pack_and_save, predict_qduig
from roboeye.qduig.metrics_ext import full_binary_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=str(ROOT / "configs" / "proposed_prefix.yaml"))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--baseline-checkpoint", default=str(ROOT / "results/camva/checkpoints/baseline_cnnvit_seed42.pt"))
    args = p.parse_args()

    cfg = load_config(args.config)
    splits, records = load_splits()
    out = Path(args.output_dir)
    init_run(out, cfg, args.seed, f"prefix views {args.split}", "First-k views, same protocol as baseline. No test tuning.")
    t0 = time.perf_counter()
    model = load_qduig(Path(args.checkpoint), cfg)
    rows = []
    for k in range(1, 7):
        loader = make_loader(splits[args.split], records, n_views=k, train=False, batch=8)
        pred = predict_qduig(model, loader, DEVICE)
        sub = out / f"{k}view"
        m = pack_and_save(pred, f"prefix_{k}view", sub)
        m["n_views_requested"] = k
        rows.append({"k": k, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "ece": m["ece"]})
        print(f"proposed {k}view acc={m['accuracy']:.4f}", flush=True)

    base_rows = []
    bpath = Path(args.baseline_checkpoint)
    if bpath.is_file() and args.split == "test":
        from roboeye.camva.engine import predict_baseline

        baseline = load_baseline(bpath)
        for k in range(1, 7):
            loader = make_loader(splits["test"], records, n_views=k, train=False, batch=8)
            pred = predict_baseline(baseline, loader, DEVICE)
            acc = full_binary_report(np.array(pred["y_true"]), np.array(pred["genuine_score"]))["accuracy"]
            base_rows.append({"k": k, "accuracy": acc})
            print(f"baseline {k}view acc={acc:.4f}", flush=True)

    payload = {
        "split": args.split,
        "proposed": rows,
        "baseline": base_rows,
        "elapsed_s": time.perf_counter() - t0,
        "protocol": "first_k_views_same_as_baseline",
    }
    save_json(out / "views_1_to_6.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
