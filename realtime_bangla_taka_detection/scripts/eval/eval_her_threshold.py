"""Choose an HER scale and threshold on validation. Score test once.

ECE is computed from the probabilities. Accuracy uses the validation threshold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.camva.engine import make_loader
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.calibration import HERCalibrator, binary_entropy, expected_calibration_error
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig
from roboeye.qduig.quality import image_quality_factors

OUT = ROOT / "results" / "calibration" / "her_threshold_seed42.json"
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _collect(model, ids, records) -> dict[str, np.ndarray]:
    loader = make_loader(ids, records, n_views=6, train=False, batch=8, workers=0)
    ys, logits, qual = [], [], []
    for batch in loader:
        views = batch["views"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        with torch.inference_mode():
            logit = model(views, mask)["logits"]
            factors = image_quality_factors(views)
        ys.append(batch["label"].numpy())
        logits.append(logit.detach().cpu().numpy())
        qual.append(factors.mean(dim=(1, 2)).detach().cpu().numpy())
    return {"y": np.concatenate(ys), "logits": np.concatenate(logits), "quality": np.concatenate(qual)}


def _scaled(her: HERCalibrator, pack: dict[str, np.ndarray], scale: float) -> np.ndarray:
    saved = dict(her.params)
    her.params["alpha"] = saved["alpha"] * scale
    her.params["beta"] = saved["beta"] * scale
    her.params["gamma"] = saved["gamma"] * scale
    prob = her.transform(pack["logits"], pack["quality"])
    her.params = saved
    return prob


def _best_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    grid = np.unique(np.concatenate([np.linspace(0.05, 0.95, 19), p]))
    best_t, best_a = 0.5, -1.0
    for t in grid:
        acc = float(np.mean((p >= t).astype(np.int64) == y))
        if acc > best_a:
            best_t, best_a = float(t), acc
    return best_t, best_a


def main() -> None:
    splits, records = load_splits()
    model = load_qduig(
        ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt",
        load_config(ROOT / "configs" / "proposed_prefix_ft.yaml"),
    )
    model.eval()
    val = _collect(model, splits["val"], records)
    test = _collect(model, splits["test"], records)
    her = HERCalibrator()
    her.fit(val["logits"], val["y"], val["quality"])
    raw_val = 1.0 / (1.0 + np.exp(-(val["logits"][:, 1] - val["logits"][:, 0])))
    raw_val_acc = float(np.mean((raw_val >= 0.5).astype(np.int64) == val["y"]))
    rows = []
    for scale in SCALES:
        pv = _scaled(her, val, scale)
        ece = float(expected_calibration_error(val["y"], pv, n_bins=10))
        threshold, val_acc = _best_threshold(val["y"], pv)
        rows.append({
            "scale": scale,
            "val_ece_10": ece,
            "val_threshold": threshold,
            "val_accuracy": val_acc,
            "val_accuracy_at_0.5": float(np.mean((pv >= 0.5).astype(np.int64) == val["y"])),
        })
    feasible = [row for row in rows if row["val_ece_10"] < 0.04 and row["val_accuracy"] + 1e-12 >= raw_val_acc]
    if feasible:
        chosen = max(feasible, key=lambda row: (row["val_accuracy"], -row["val_ece_10"]))
        reason = "validation ECE under 0.04 and validation accuracy at least the raw 0.5-threshold accuracy"
    else:
        chosen = max(rows, key=lambda row: (row["val_accuracy"], -row["val_ece_10"]))
        reason = "no scale met both validation constraints; highest validation accuracy was kept"
    pt = _scaled(her, test, chosen["scale"])
    payload = {
        "fit_split": "val",
        "eval_split": "test",
        "n_test": int(len(test["y"])),
        "views": 6,
        "raw_val_accuracy_at_0.5": raw_val_acc,
        "selection": reason,
        "candidates": rows,
        "chosen": chosen,
        "test": {
            "accuracy": float(np.mean((pt >= chosen["val_threshold"]).astype(np.int64) == test["y"])),
            "accuracy_at_0.5": float(np.mean((pt >= 0.5).astype(np.int64) == test["y"])),
            "ece_10": float(expected_calibration_error(test["y"], pt, n_bins=10)),
            "threshold": chosen["val_threshold"],
            "scale": chosen["scale"],
        },
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["test"], indent=2))


if __name__ == "__main__":
    main()
