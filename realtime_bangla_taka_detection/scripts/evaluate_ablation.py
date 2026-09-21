"""Ablations A–E on the saved note-disjoint test split."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboeye.camva.engine import (
    CKPT_DIR,
    METRIC_DIR,
    adaptive_predict,
    load_baseline,
    load_camva,
    make_loader,
    pack_eval,
    predict_baseline,
    predict_camva,
    save_json,
)
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE


def _ckpt(pat: str) -> Path:
    hits = sorted(CKPT_DIR.glob(pat))
    if not hits:
        raise SystemExit(f"missing {pat}")
    return hits[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.90)
    args = p.parse_args()
    splits, records = load_splits()
    test = splits["test"]
    base = load_baseline(_ckpt("baseline_cnnvit_seed*.pt"))
    camva = load_camva(_ckpt("camva_quality_attention_seed*.pt"), "quality_attention")
    loader6 = make_loader(test, records, n_views=6, train=False, batch=args.batch)

    rows = []

    bp = predict_baseline(base, loader6, DEVICE)
    bm, _ = pack_eval(bp, "A_baseline_mean_cnnvit")
    rows.append({"ablation": "A_cnnvit_simple_averaging", **{k: bm[k] for k in ("accuracy", "f1", "roc_auc", "ece")}})

    for fusion, name in (("mean", "B_attention_off_mean"), ("attention", "C_attention_fusion"), ("quality_attention", "D_quality_aware_no_adaptive")):
        camva.fusion = fusion
        loader6 = make_loader(test, records, n_views=6, train=False, batch=args.batch)
        cp = predict_camva(camva, loader6, DEVICE)
        cm, _ = pack_eval(cp, name)
        rows.append({"ablation": name, **{k: cm[k] for k in ("accuracy", "f1", "roc_auc", "ece")}, "average_views": 6.0})
        print(name, cm["accuracy"], cm["f1"])

    camva.fusion = "quality_attention"
    ap = adaptive_predict(camva, test, records, threshold=args.threshold, order="fixed")
    em, _ = pack_eval(ap, "E_full_camva_adaptive")
    rows.append(
        {
            "ablation": "E_full_camva_adaptive",
            "accuracy": em["accuracy"],
            "f1": em["f1"],
            "roc_auc": em["roc_auc"],
            "ece": em["ece"],
            "average_views": ap["average_views"],
            "pct_all_six": ap["pct_used_all_six"],
        }
    )
    print("E adaptive", em["accuracy"], "avg views", ap["average_views"])
    save_json(METRIC_DIR / "ablations.json", rows)
    with (METRIC_DIR / "ablations.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
