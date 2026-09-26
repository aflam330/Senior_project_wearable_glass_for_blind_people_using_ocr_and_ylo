"""Evaluate a trained novel algorithm on one split. Default split is test, used once."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.camva.engine import make_loader
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.artifacts import init_run, save_json
from roboeye.qduig.config_io import load_config
from roboeye.qduig.metrics_ext import full_binary_report
from scripts.train.train_novel import BUILDERS, _module


@torch.inference_mode()
def predict(model, loader):
    ys, ps, ids = [], [], []
    nviews = []
    for batch in loader:
        views = batch["views"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        out = model(views, mask)
        ys.append(batch["label"].numpy())
        ps.append(out["prob"].detach().cpu().numpy())
        ids.extend(batch["note_id"])
        nviews.extend(mask.sum(1).cpu().tolist())
    return np.concatenate(ys), np.concatenate(ps), ids, nviews


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--algo", required=True, choices=sorted(BUILDERS))
    args = p.parse_args()
    cfg = load_config(args.config)
    splits, records = load_splits()
    out = Path(args.output_dir)
    init_run(out, cfg, args.seed, f"{args.algo} {args.split}", "Evaluation only. Threshold 0.5 was not tuned on test.")
    model = BUILDERS[args.algo]().to(DEVICE)
    try:
        blob = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    except TypeError:
        blob = torch.load(args.checkpoint, map_location=DEVICE)
    _module(model).load_state_dict(blob["model"])
    model.eval()
    rows = []
    preds = {}
    for k in range(1, 7):
        cuda = DEVICE.type == "cuda"
        workers = int(os.environ.get("NOVEL_WORKERS", "4" if cuda else "0"))
        loader = make_loader(
            splits[args.split], records, n_views=k, train=False, batch=4,
            workers=max(0, workers), pin_memory=cuda,
        )
        y, prob, ids, nviews = predict(model, loader)
        report = full_binary_report(y, prob)
        rows.append({"k": k, "accuracy": report["accuracy"], "macro_f1": report["macro_f1"], "ece": report["ece"], "n": int(len(y))})
        preds[str(k)] = {"note_id": ids, "y_true": y.astype(int).tolist(), "genuine_score": prob.astype(float).tolist(), "n_views": nviews}
        print(f"{args.algo} {args.split} {k}view acc={report['accuracy']:.4f}", flush=True)
    save_json(out / "test_metrics.json" if args.split == "test" else "val_metrics.json", {"split": args.split, "views": rows, "seed": args.seed, "algo": args.algo})
    save_json(out / "test_predictions.json" if args.split == "test" else "val_predictions.json", preds)
    # confusion at 6 views
    y = np.array(preds["6"]["y_true"])
    hat = (np.array(preds["6"]["genuine_score"]) >= 0.5).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for t, h in zip(y, hat):
        cm[int(t), int(h)] += 1
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{args.algo} 6-view")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    scores = np.array(preds["6"]["genuine_score"])
    fig, ax = plt.subplots(figsize=(4, 3))
    bins = np.linspace(0, 1, 11)
    centers = 0.5 * (bins[:-1] + bins[1:])
    accs = []
    for i in range(10):
        m = (scores >= bins[i]) & (scores < bins[i + 1] if i < 9 else scores <= bins[i + 1])
        accs.append(float(y[m].mean()) if m.any() else np.nan)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(centers, accs, marker="o")
    ax.set_title(f"{args.algo} calibration 6-view")
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=120)
    plt.close(fig)
    (out / "README.md").write_text(
        f"# {args.algo} {args.split} seed {args.seed}\n\nNumbers in test_metrics.json. Threshold 0.5 fixed, not tuned on this split.\n",
        encoding="utf-8",
    )
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
