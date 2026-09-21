"""Controlled corruption eval. Same corruptions for baseline and CAMVA."""
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

CORRUPTIONS = [
    ("gaussian_blur", 1.0),
    ("gaussian_blur", 3.0),
    ("gaussian_blur", 6.0),
    ("motion_blur", 5.0),
    ("brightness", 0.6),
    ("low_light", 0.35),
    ("occlusion", 0.2),
    ("occlusion", 0.4),
    ("occlusion", 0.6),
    ("jpeg", 30.0),
    ("rotation", 15.0),
    ("rotation", 30.0),
    ("rotation", 45.0),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    splits, records = load_splits()
    test = splits["test"]
    base = load_baseline(sorted(CKPT_DIR.glob("baseline_cnnvit_seed*.pt"))[-1])
    camva = load_camva(sorted(CKPT_DIR.glob("camva_quality_attention_seed*.pt"))[-1], "quality_attention")

    clean_loader = make_loader(test, records, n_views=6, train=False, batch=args.batch)
    clean_b, _ = pack_eval(predict_baseline(base, clean_loader, DEVICE), "clean_base")
    clean_loader = make_loader(test, records, n_views=6, train=False, batch=args.batch)
    clean_c, _ = pack_eval(predict_camva(camva, clean_loader, DEVICE), "clean_camva")

    rows = [{
        "corruption": "clean",
        "severity": 0,
        "baseline_acc": clean_b["accuracy"],
        "camva_acc": clean_c["accuracy"],
        "baseline_drop": 0.0,
        "camva_drop": 0.0,
    }]
    for name, sev in CORRUPTIONS:
        lb = make_loader(test, records, n_views=6, train=False, batch=args.batch, corruption=name, severity=sev)
        mb, _ = pack_eval(predict_baseline(base, lb, DEVICE), f"b_{name}_{sev}")
        lc = make_loader(test, records, n_views=6, train=False, batch=args.batch, corruption=name, severity=sev)
        mc, _ = pack_eval(predict_camva(camva, lc, DEVICE), f"c_{name}_{sev}")
        row = {
            "corruption": name,
            "severity": sev,
            "baseline_acc": mb["accuracy"],
            "camva_acc": mc["accuracy"],
            "baseline_drop": clean_b["accuracy"] - mb["accuracy"],
            "camva_drop": clean_c["accuracy"] - mc["accuracy"],
        }
        print(name, sev, "base", mb["accuracy"], "camva", mc["accuracy"])
        rows.append(row)
    save_json(METRIC_DIR / "robustness.json", rows)
    with (METRIC_DIR / "robustness.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
