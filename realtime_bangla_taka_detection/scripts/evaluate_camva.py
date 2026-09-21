"""Evaluate baseline CNN+ViT and CAMVA for 1..6 views on the saved note split."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboeye.camva.engine import (
    CKPT_DIR,
    CM_DIR,
    METRIC_DIR,
    PLOT_DIR,
    PRED_DIR,
    load_baseline,
    load_camva,
    make_loader,
    pack_eval,
    predict_baseline,
    predict_camva,
    save_json,
    save_plots_views,
)
from roboeye.camva.metrics import paired_bootstrap
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
import numpy as np


def _latest(pattern: str) -> Path:
    hits = sorted(CKPT_DIR.glob(pattern))
    if not hits:
        raise SystemExit(f"missing checkpoint {CKPT_DIR}/{pattern}")
    return hits[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--baseline-ckpt", default="")
    p.add_argument("--camva-ckpt", default="")
    p.add_argument("--fusion", default="quality_attention")
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()

    splits, records = load_splits()
    test_ids = splits["test"]
    base_ckpt = Path(args.baseline_ckpt) if args.baseline_ckpt else _latest("baseline_cnnvit_seed*.pt")
    camva_ckpt = Path(args.camva_ckpt) if args.camva_ckpt else _latest(f"camva_{args.fusion}_seed*.pt")
    baseline = load_baseline(base_ckpt)
    camva = load_camva(camva_ckpt, args.fusion)

    rows = []
    last_base_s = last_camva_s = last_y = None
    for n_views in range(1, 7):
        loader = make_loader(test_ids, records, n_views=n_views, train=False, batch=args.batch)
        bp = predict_baseline(baseline, loader, DEVICE)
        bm, bp2 = pack_eval(bp, f"baseline_{n_views}view", temperature=args.temperature if args.temperature != 1.0 else None)
        save_json(PRED_DIR / f"baseline_{n_views}view.json", bp2)
        save_json(METRIC_DIR / f"baseline_{n_views}view.json", bm)
        save_json(CM_DIR / f"baseline_{n_views}view.json", {"cm": bm["confusion_matrix"]})

        loader = make_loader(test_ids, records, n_views=n_views, train=False, batch=args.batch)
        cp = predict_camva(camva, loader, DEVICE)
        cm, cp2 = pack_eval(cp, f"camva_{n_views}view", temperature=args.temperature if args.temperature != 1.0 else None)
        save_json(PRED_DIR / f"camva_{n_views}view.json", cp2)
        save_json(METRIC_DIR / f"camva_{n_views}view.json", cm)
        save_json(CM_DIR / f"camva_{n_views}view.json", {"cm": cm["confusion_matrix"]})

        y = np.array(bp2["y_true"])
        sb = np.array(bp2["genuine_score"])
        sc = np.array(cp2["genuine_score"])
        paired = paired_bootstrap(y, sb, sc, metric="accuracy")
        abs_imp = cm["accuracy"] - bm["accuracy"]
        rel_imp = abs_imp / max(bm["accuracy"], 1e-12)
        print("BASELINE RESULT", n_views, "view acc", bm["accuracy"], "f1", bm["f1"])
        print("CAMVA RESULT", n_views, "view acc", cm["accuracy"], "f1", cm["f1"])
        print("ABSOLUTE IMPROVEMENT", abs_imp)
        print("RELATIVE IMPROVEMENT", rel_imp)
        print("95% CI baseline acc", bm["bootstrap_accuracy"])
        print("95% CI camva acc", cm["bootstrap_accuracy"])
        print("STATISTICAL SIGNIFICANCE paired_bootstrap", paired)
        row = {
            "n_views": n_views,
            "baseline_accuracy": bm["accuracy"],
            "baseline_f1": bm["f1"],
            "baseline_sensitivity": bm["sensitivity"],
            "baseline_specificity": bm["specificity"],
            "baseline_roc_auc": bm["roc_auc"],
            "baseline_pr_auc": bm["pr_auc"],
            "baseline_ece": bm["ece"],
            "baseline_latency_median_ms": bm["latency"]["median_ms"],
            "camva_accuracy": cm["accuracy"],
            "camva_f1": cm["f1"],
            "camva_sensitivity": cm["sensitivity"],
            "camva_specificity": cm["specificity"],
            "camva_roc_auc": cm["roc_auc"],
            "camva_pr_auc": cm["pr_auc"],
            "camva_ece": cm["ece"],
            "camva_latency_median_ms": cm["latency"]["median_ms"],
            "absolute_improvement_acc": abs_imp,
            "relative_improvement_acc": rel_imp,
            "paired_p_accuracy": paired["p_value"],
            "paired_diff_acc": paired["mean_difference"],
        }
        rows.append(row)
        last_base_s, last_camva_s, last_y = sb, sc, y

    save_json(METRIC_DIR / "views_1_to_6.json", rows)
    csv_path = METRIC_DIR / "views_1_to_6.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_plots_views(
        [{"n_views": r["n_views"], "accuracy": r["camva_accuracy"]} for r in rows],
        PLOT_DIR / "camva_accuracy_vs_views.png",
        "CAMVA accuracy vs number of views (JaalTaka note-disjoint test)",
        "accuracy",
        "Accuracy",
    )
    save_plots_views(
        [{"n_views": r["n_views"], "accuracy": r["baseline_accuracy"]} for r in rows],
        PLOT_DIR / "baseline_accuracy_vs_views.png",
        "Baseline CNN+ViT accuracy vs number of views",
        "accuracy",
        "Accuracy",
    )
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
