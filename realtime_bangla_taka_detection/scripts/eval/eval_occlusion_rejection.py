"""Entropy rejection for occlusion 0.55.

The threshold is chosen on the validation split only: among the 50, 60, 70,
80, and 90 percent quantiles of occluded validation entropy, keep the rule
that accepts the most notes while accepted accuracy is at least 0.90.
If none reach 0.90, keep the quantile with the highest accepted accuracy.
That one threshold is then applied to the test split.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.authenticity import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from roboeye.camva.data import apply_corruption
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig

_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
CKPT = ROOT / "results" / "qduig" / "occlusion_ft" / "seed42" / "checkpoint.pt"
CFG = ROOT / "configs" / "occlusion_ft.yaml"
OUT = ROOT / "results" / "robustness" / "occlusion_rejection_seed42.json"


def _median_fill(bgr: np.ndarray) -> np.ndarray:
    hole = bgr.sum(axis=2) == 0
    if not np.any(hole):
        return bgr
    out = bgr.copy()
    out[hole] = np.median(bgr[~hole], axis=0)
    return out


def _tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _TENSOR(transforms.Resize((IMG_SIZE, IMG_SIZE))(Image.fromarray(rgb)))


def _scores(model, records, ids, occluded: bool) -> tuple[np.ndarray, np.ndarray]:
    probs, ents = [], []
    for start in range(0, len(ids), 4):
        chunk = ids[start:start + 4]
        batch = []
        for i, nid in enumerate(chunk):
            if occluded:
                np.random.seed(1000 + start + i)
            frames = []
            for path in records[nid]["view_paths"]:
                bgr = cv2.imread(str(path))
                if occluded:
                    bgr = _median_fill(apply_corruption(bgr, "occlusion", 0.55))
                frames.append(_tensor(bgr))
            batch.append(torch.stack(frames, dim=0))
        views = torch.stack(batch, dim=0).to(DEVICE)
        mask = torch.ones(views.size(0), views.size(1), dtype=torch.long, device=DEVICE)
        with torch.inference_mode():
            out = model(views, mask)
            probs.append(torch.softmax(out["logits"], dim=1)[:, 1].detach().cpu().numpy())
            ents.append(out["entropy"].detach().cpu().numpy())
        print(f"{'occ' if occluded else 'clean'} {min(start + 4, len(ids))}/{len(ids)}", flush=True)
    return np.concatenate(probs), np.concatenate(ents)


def _accepted(prob, ent, y, threshold: float) -> tuple[float, float]:
    keep = ent <= threshold
    coverage = float(keep.mean())
    if int(keep.sum()) == 0:
        return coverage, float("nan")
    pred = prob[keep] >= 0.5
    return coverage, float(np.mean(pred == y[keep]))


def main() -> None:
    splits, records = load_splits()
    model = load_qduig(CKPT, load_config(CFG))
    model.eval()
    val_ids = list(splits["val"])
    val_y = np.array([int(records[nid]["label"]) for nid in val_ids])
    val_prob, val_ent = _scores(model, records, val_ids, True)
    quantiles = [0.50, 0.60, 0.70, 0.80, 0.90]
    rows = []
    for q in quantiles:
        thr = float(np.quantile(val_ent, q))
        coverage, acc = _accepted(val_prob, val_ent, val_y, thr)
        rows.append({"quantile": q, "threshold": thr, "val_coverage": coverage, "val_accepted_accuracy": acc})
        print(json.dumps(rows[-1]), flush=True)
    eligible = [row for row in rows if row["val_accepted_accuracy"] >= 0.90]
    if eligible:
        chosen = max(eligible, key=lambda row: row["val_coverage"])
        rule = "highest coverage among validation quantiles with accepted accuracy at least 0.90"
    else:
        chosen = max(rows, key=lambda row: row["val_accepted_accuracy"])
        rule = "no quantile reached 0.90 accepted validation accuracy; highest accepted accuracy kept"
    test_ids = list(splits["test"])
    test_y = np.array([int(records[nid]["label"]) for nid in test_ids])
    occ_prob, occ_ent = _scores(model, records, test_ids, True)
    clean_prob, clean_ent = _scores(model, records, test_ids, False)
    occ_cov, occ_acc = _accepted(occ_prob, occ_ent, test_y, chosen["threshold"])
    clean_cov, clean_acc = _accepted(clean_prob, clean_ent, test_y, chosen["threshold"])
    full_occ = float(np.mean((occ_prob >= 0.5) == test_y))
    payload = {
        "split_threshold": "val",
        "split_report": "test",
        "n": len(test_ids),
        "occlusion": 0.55,
        "repair": "median fill",
        "rule": rule,
        "validation_grid": rows,
        "chosen": chosen,
        "test_occluded_accuracy_no_rejection": full_occ,
        "test_occluded_coverage": occ_cov,
        "test_occluded_rejection_rate": 1.0 - occ_cov,
        "test_occluded_accepted_accuracy": occ_acc,
        "test_clean_coverage": clean_cov,
        "test_clean_rejection_rate": 1.0 - clean_cov,
        "test_clean_accepted_accuracy": clean_acc,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "validation_grid"}), flush=True)


if __name__ == "__main__":
    main()
