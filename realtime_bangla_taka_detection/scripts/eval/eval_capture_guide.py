"""Thresholds for the glass capture guide (savior_glass/modes/capture_guide.py), set on VALIDATION only.

Data: JaalTaka back-lit view 6, serial-disjoint split. Quality is measured exactly as on the glass
(frame_quality: grey mean and Laplacian variance at 400 px wide).
Thresholds: 2nd percentile of VALIDATION photos (so about 2 % of good validation photos are asked again).
Then, on TEST photos: share of clean photos accepted, and share rejected after fixed degradations
(dark: x0.25 brightness; defocus: Gaussian sigma 4; motion: 21 px horizontal line kernel).
Output: results/capture_guide/thresholds.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "savior_glass"))
from roboeye.camva.notes import load_splits  # noqa: E402
from modes.capture_guide import assess, frame_quality  # noqa: E402

OUT = ROOT / "results" / "capture_guide"
MOTION = np.zeros((21, 21), np.float32)
MOTION[10, :] = 1 / 21
DEGRADE = {"dark_x0.25": lambda im: (im.astype(np.float32) * 0.25).astype(np.uint8),
           "defocus_sigma4": lambda im: cv2.GaussianBlur(im, (0, 0), 4),
           "motion_21px": lambda im: cv2.filter2D(im, -1, MOTION)}


def load(path):
    im = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    return cv2.resize(im, (700, round(im.shape[0] * 700 / im.shape[1])), interpolation=cv2.INTER_AREA)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    splits, records = load_splits(ROOT / "results" / "serial_split")
    val = [frame_quality(load(records[n]["view_paths"][5])) for n in splits["val"]]
    th = {"min_mean": float(np.percentile([q["mean"] for q in val], 2)),
          "min_lapvar": float(np.percentile([q["lap_var"] for q in val], 2))}
    res = {"split": "serial_split", "source": "JaalTaka back-lit view 6", "rule": "2nd percentile of VAL", "n_val": len(val),
           "thresholds": th, "test": {}}
    test = [load(records[n]["view_paths"][5]) for n in splits["test"]]
    for name, fn in [("clean", lambda im: im)] + list(DEGRADE.items()):
        st = [assess(fn(im), th["min_mean"], th["min_lapvar"])[0] for im in test]
        res["test"][name] = {"n": len(st), "accepted": st.count("ok"), "rejected_dark": st.count("dark"),
                             "rejected_blurry": st.count("blurry"), "accepted_rate": st.count("ok") / len(st)}
    (OUT / "thresholds.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
