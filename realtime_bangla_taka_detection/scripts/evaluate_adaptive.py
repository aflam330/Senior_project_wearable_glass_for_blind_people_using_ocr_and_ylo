"""Adaptive stopping + view-order experiments. No future views in the stop decision."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboeye.camva.engine import (
    CAL_DIR,
    CKPT_DIR,
    METRIC_DIR,
    PLOT_DIR,
    PRED_DIR,
    adaptive_predict,
    load_camva,
    pack_eval,
    save_adaptive_plot,
    save_json,
)
from roboeye.camva.notes import load_splits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fusion", default="quality_attention")
    args = p.parse_args()
    splits, records = load_splits()
    hits = sorted(CKPT_DIR.glob(f"camva_{args.fusion}_seed*.pt"))
    if not hits:
        raise SystemExit("train CAMVA first")
    model = load_camva(hits[-1], args.fusion)
    t_path = CAL_DIR / "temperature.json"
    temperature = 1.0
    if t_path.is_file():
        import json
        temperature = float(json.loads(t_path.read_text(encoding="utf-8"))["temperature"])

    thresholds = [0.80, 0.85, 0.90, 0.95, 0.98]
    orders = ["fixed", "random", "confidence"]
    rows = []
    for order in orders:
        order_rows = []
        for thr in thresholds:
            pred = adaptive_predict(
                model, splits["test"], records,
                threshold=thr, order=order, temperature=temperature,
            )
            m, pred2 = pack_eval(pred, f"adaptive_{order}_{thr}")
            save_json(PRED_DIR / f"adaptive_{order}_{thr}.json", pred2)
            row = {
                "order": order,
                "threshold": thr,
                "accuracy": m["accuracy"],
                "f1": m["f1"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "ece": m["ece"],
                "false_acceptance_rate": m["false_acceptance_rate"],
                "false_rejection_rate": m["false_rejection_rate"],
                "average_views": pred["average_views"],
                "pct_requiring_six_views": pred["pct_used_all_six"],
                "median_latency_ms": m["latency"]["median_ms"],
                "p95_latency_ms": m["latency"]["p95_ms"],
            }
            print(order, thr, "acc", row["accuracy"], "f1", row["f1"], "avg_views", row["average_views"])
            rows.append(row)
            order_rows.append(row)
        if order == "fixed":
            save_adaptive_plot(
                order_rows, PLOT_DIR / "adaptive_acc_vs_avg_views.png",
                "accuracy", "Authentication accuracy",
                "CAMVA accuracy vs average views (fixed order, calibrated)",
            )
            save_adaptive_plot(
                order_rows, PLOT_DIR / "adaptive_f1_vs_avg_views.png",
                "f1", "F1",
                "CAMVA F1 vs average views (fixed order, calibrated)",
            )
    save_json(METRIC_DIR / "adaptive.json", rows)
    with (METRIC_DIR / "adaptive.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
