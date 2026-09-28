"""Diagnose PRMVT under 55% black-box occlusion on the VALIDATION split (test is not read).

Reports accuracy, confusion (what the model predicts when it fails), 1-view vs 6-view,
black box vs median fill, the baseline, and how much of each note stays visible.
Writes results/robustness/occlusion_diagnosis_val.json.
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
from roboeye.camva.engine import load_baseline
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig

_TF = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                          transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def median_fill(bgr):
    hole = bgr.sum(axis=2) == 0
    if not hole.any():
        return bgr
    out = bgr.copy()
    out[hole] = np.median(bgr[~hole], axis=0)
    return out


def views_for(records, nid, idx, occ, fill):
    frames, visible = [], []
    np.random.seed(1000 + idx)  # same per-note seeding rule as the test protocol, applied to val ids
    for p in records[nid]["view_paths"]:
        bgr = cv2.imread(str(p))
        if occ:
            bgr = apply_corruption(bgr, "occlusion", 0.55)
            visible.append(float((bgr.sum(axis=2) > 0).mean()))
            if fill:
                bgr = median_fill(bgr)
        frames.append(_TF(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
    return torch.stack(frames), visible


@torch.inference_mode()
def run(model, kind, ids, records, occ, fill, k):
    probs, vis = [], []
    for idx, nid in enumerate(ids):
        v, visible = views_for(records, nid, idx, occ, fill)
        v = v[:k].unsqueeze(0).to(DEVICE)
        if kind == "baseline":
            logit = model(v)
        else:
            logit = model(v, torch.ones(1, k, dtype=torch.long, device=DEVICE))["logits"]
        probs.append(float(torch.softmax(logit, 1)[0, 1]))
        vis.append(visible)
    return np.array(probs), vis


def summary(p, y):
    pred = (p >= 0.5).astype(int)
    cm = [[int(((y == t) & (pred == h)).sum()) for h in (0, 1)] for t in (0, 1)]
    return {"accuracy": float((pred == y).mean()), "confusion[true][pred]": cm,
            "genuine_recall": cm[1][1] / max(sum(cm[1]), 1), "counterfeit_recall": cm[0][0] / max(sum(cm[0]), 1),
            "mean_prob_genuine": float(p.mean())}


def main():
    splits, records = load_splits()
    ids = list(splits["val"])
    y = np.array([int(records[n]["label"]) for n in ids])
    prm = load_qduig(ROOT / "results/qduig/prefix_ft/seed42/checkpoint.pt", load_config(ROOT / "configs/proposed_prefix_ft.yaml")).to(DEVICE).eval()
    base = load_baseline(ROOT / "results/camva/checkpoints/baseline_cnnvit_seed42.pt").to(DEVICE).eval()
    out = {"split": "val", "n": len(ids), "occlusion": 0.55}
    for name, model, kind in (("prmvt", prm, "q"), ("baseline", base, "baseline")):
        for label, occ, fill, k in (("clean_6v", False, False, 6), ("occ_black_6v", True, False, 6),
                                    ("occ_black_1v", True, False, 1), ("occ_median_6v", True, True, 6)):
            p, vis = run(model, kind, ids, records, occ, fill, k)
            out[f"{name}_{label}"] = summary(p, y)
            if name == "prmvt" and label == "occ_black_6v":
                v = np.array([np.mean(x) for x in vis])
                wrong = (p >= 0.5).astype(int) != y
                out["visible_fraction_per_view_mean"] = float(np.mean([np.mean(x) for x in vis]))
                out["visible_fraction_wrong_vs_right"] = [float(v[wrong].mean()) if wrong.any() else None, float(v[~wrong].mean())]
            print(name, label, json.dumps(out[f"{name}_{label}"]), flush=True)
    dst = ROOT / "results/robustness/occlusion_diagnosis_val.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(dst)


if __name__ == "__main__":
    main()
