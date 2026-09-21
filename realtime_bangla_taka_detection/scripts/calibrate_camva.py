"""Fit temperature scaling on VAL notes; never on test."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from roboeye.camva.engine import (
    CAL_DIR,
    CKPT_DIR,
    load_camva,
    make_loader,
    pack_eval,
    predict_camva,
    save_json,
    temperature_scale,
)
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE


def _ckpt(fusion: str) -> Path:
    hits = sorted(CKPT_DIR.glob(f"camva_{fusion}_seed*.pt"))
    if not hits:
        raise SystemExit("no CAMVA checkpoint")
    return hits[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fusion", default="quality_attention")
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    splits, records = load_splits()
    model = load_camva(_ckpt(args.fusion), args.fusion)
    val_loader = make_loader(splits["val"], records, n_views=6, train=False, batch=args.batch)
    val_pred = predict_camva(model, val_loader, DEVICE)
    t = temperature_scale(np.array(val_pred["logits"]), np.array(val_pred["y_true"]))
    save_json(CAL_DIR / "temperature.json", {"temperature": t, "fitted_on": "val_notes", "n_val": len(val_pred["y_true"])})

    test_loader = make_loader(splits["test"], records, n_views=6, train=False, batch=args.batch)
    test_pred = predict_camva(model, test_loader, DEVICE)
    raw, _ = pack_eval(test_pred, "uncalibrated")
    cal, _ = pack_eval(test_pred, "temperature", temperature=t)
    save_json(CAL_DIR / "test_uncalibrated.json", raw)
    save_json(CAL_DIR / "test_temperature.json", cal)
    print("temperature T=", t)
    print("UNCALIBRATED ECE", raw["ece"], "Brier", raw["brier"], "acc", raw["accuracy"])
    print("TEMPERATURE  ECE", cal["ece"], "Brier", cal["brier"], "acc", cal["accuracy"])
    print("reliability uncal", raw["reliability"])
    print("reliability cal", cal["reliability"])


if __name__ == "__main__":
    main()
