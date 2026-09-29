"""Confidence-threshold rejection for PRMVT under clean, low-light, occlusion and over-exposure.

Rule (same as results/robustness/occlusion_decision.json, fixed before test scoring):
  confidence = max(p, 1 - p); answer iff confidence >= t;
  t = largest value on the grid 0.50..0.99 (step 0.01) that still answers >= 95 % of CLEAN
  validation notes. The threshold is chosen once, on clean validation, and applied unchanged
  to every corrupted test condition (the device cannot know the corruption).
Corruptions follow the robustness protocol: per note index i, np.random.seed(1000 + i), then
apply_corruption on every view. Views: first 1 and all 6.

Output: results/safety/rejection_seed42.json (per condition: accuracy, coverage, wrong-verdict
rate, selective accuracy, AURC, and the risk-coverage curve on test).
"""
from __future__ import annotations

import json
import math
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

RUN = ROOT / "results" / "qduig" / "prefix_ft" / "seed42"
CONDITIONS = [("clean", None, None), ("low_light_0.35", "low_light", 0.35), ("low_light_0.2", "low_light", 0.2),
              ("occlusion_0.55", "occlusion", 0.55), ("brightness_2.2", "brightness", 2.2)]
TF = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                         transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [math.nan, math.nan]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


@torch.inference_mode()
def probs(model, records, ids, corr, sev, k):
    out = []
    for i, nid in enumerate(ids):
        if corr:
            np.random.seed(1000 + i)
        views = []
        for p in records[nid]["view_paths"][:k]:
            bgr = cv2.imread(str(p))
            if corr:
                bgr = apply_corruption(bgr, corr, sev)
            views.append(TF(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
        v = torch.stack(views).unsqueeze(0).to(DEVICE)
        m = torch.ones(1, v.size(1), dtype=torch.long, device=DEVICE)
        out.append(float(torch.softmax(model(v, m)["logits"], 1)[0, 1]))
    return np.array(out)


def decide(p, y, t):
    conf = np.maximum(p, 1 - p)
    ans = conf >= t
    right = ((p >= 0.5) == (y == 1))
    n, na = len(y), int(ans.sum())
    wrong = int((ans & ~right).sum())
    order = np.argsort(-conf)  # risk-coverage curve: answer the most confident first
    risks = np.cumsum(~right[order]) / np.arange(1, n + 1)
    curve = [{"coverage": round((j + 1) / n, 4), "selective_risk": round(float(risks[j]), 4)}
             for j in range(0, n, max(1, n // 20))]
    return {
        "accuracy_no_rejection": float(right.mean()),
        "threshold": t, "coverage": na / n, "answered": na,
        "wrong_verdicts": wrong, "wrong_verdict_rate": wrong / n,
        "wrong_verdict_wilson95": wilson(wrong, n),
        "selective_accuracy": (na - wrong) / na if na else None,
        "aurc": float(risks.mean()),
        "risk_coverage_curve": curve,
    }


def main() -> None:
    splits, records = load_splits()
    model = load_qduig(RUN / "checkpoint.pt", load_config(RUN / "config.yaml")).to(DEVICE).eval()
    val, test = list(splits["val"]), list(splits["test"])
    yv = np.array([int(records[n]["label"]) for n in val])
    yt = np.array([int(records[n]["label"]) for n in test])
    out = {"model": str(RUN.relative_to(ROOT)), "rule": __doc__.split("Corruptions")[0].strip(), "views": {}}
    for k in (1, 6):
        pv = probs(model, records, val, None, None, k)
        conf_v = np.maximum(pv, 1 - pv)
        grid = [round(0.50 + 0.01 * i, 2) for i in range(50)]
        ok = [t for t in grid if (conf_v >= t).mean() >= 0.95]
        t = max(ok) if ok else 0.5
        res = {"threshold_from_clean_val": t, "clean_val_coverage": float((conf_v >= t).mean())}
        for name, corr, sev in CONDITIONS:
            pt = probs(model, records, test, corr, sev, k)
            res[name] = decide(pt, yt, t)
            r = res[name]
            print(f"k={k} {name}: acc {r['accuracy_no_rejection']:.4f} coverage {r['coverage']:.3f} "
                  f"wrong {r['wrong_verdicts']}/208 selective {r['selective_accuracy']}", flush=True)
        out["views"][str(k)] = res
    dst = ROOT / "results" / "safety"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "rejection_seed42.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
