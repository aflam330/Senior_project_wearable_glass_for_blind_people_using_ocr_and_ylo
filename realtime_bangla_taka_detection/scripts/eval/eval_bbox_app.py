"""Boxes as the glass app produces them (CurrencyMode.detect_live), checked against labels.

1. Synthetic test (exact labels): IoU of the app's top box, split by whether detect_live's
   dark-frame brightening (frame mean < 80) was applied, and compared with the raw detector.
2. Real close-up photos whose heuristic label is the full frame (the note fills the photo):
   share detected and share of the photo the app's box covers.
3. The crop handed to the jaal check equals the box clipped to the frame.
Writes results/bbox/bbox_app_eval.json.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WS / "savior_glass"))
from scripts.eval.eval_bbox import iou, read_labels  # noqa: E402


def main() -> None:
    from ultralytics import YOLO
    from modes.currency_mode import CurrencyMode

    cm = CurrencyMode()
    cm._load_yolo()
    raw = YOLO(str(ROOT / "models/best.pt"))
    crops = []
    orig_auth = cm._authenticity
    cm._authenticity = lambda crop: (crops.append(crop.shape[:2]) or orig_auth(crop))
    cm._load_auth()

    syn = WS / "data set/currency_yolo_data/test"
    res = {"dark": {"app": [], "raw": []}, "normal": {"app": [], "raw": []}}
    crop_ok = crop_n = 0
    for p in sorted((syn / "images").glob("*")):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gts = read_labels(syn / "labels" / (p.stem + ".txt"), w, h)
        if not gts:
            continue
        group = "dark" if float(img.mean()) < 80 else "normal"
        crops.clear()
        hits = cm.detect_live(img)
        if hits:
            x, y, bw, bh = hits[0]["bbox"]
            ab = [x, y, x + bw, y + bh]
            res[group]["app"].append(max(iou(ab, g[1:]) for g in gts))
            if crops:  # crop given to the jaal check = box clipped to the frame
                exp = (min(h, y + bh) - max(0, y), min(w, x + bw) - max(0, x))
                crop_n += 1
                crop_ok += crops[0] == exp
        else:
            res[group]["app"].append(0.0)
        r = raw.predict(img, conf=0.25, imgsz=640, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            rb = r.boxes.xyxy[int(r.boxes.conf.argmax())].cpu().numpy().tolist()
            res[group]["raw"].append(max(iou(rb, g[1:]) for g in gts))
        else:
            res[group]["raw"].append(0.0)
    out = {"synthetic": {}}
    for g, d in res.items():
        if d["app"]:
            out["synthetic"][g] = {"n": len(d["app"]),
                                   "app_iou_mean": round(statistics.fmean(d["app"]), 4),
                                   "app_iou>=0.5": round(sum(i >= 0.5 for i in d["app"]) / len(d["app"]), 4),
                                   "raw_iou_mean": round(statistics.fmean(d["raw"]), 4),
                                   "raw_iou>=0.5": round(sum(i >= 0.5 for i in d["raw"]) / len(d["raw"]), 4)}
    out["crop_matches_box"] = {"checked": crop_n, "matching": crop_ok}

    real = WS / "data set/yolo_bbox_annotated/test"
    covered, n, det = [], 0, 0
    for p in sorted((real / "images").glob("*")):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gts = read_labels(real / "labels" / (p.stem + ".txt"), w, h)
        if not gts or (gts[0][3] - gts[0][1]) * (gts[0][4] - gts[0][2]) < 0.97 * w * h:
            continue
        n += 1
        hits = cm.detect_live(img)
        if hits:
            det += 1
            x, y, bw, bh = hits[0]["bbox"]
            covered.append(iou([x, y, x + bw, y + bh], [0, 0, w, h]))
    out["real_fullframe_closeups"] = {"n": n, "detected": round(det / max(n, 1), 4),
                                      "box_iou_with_whole_photo_median": round(statistics.median(covered), 4) if covered else None,
                                      "box_iou_with_whole_photo>=0.8": round(sum(c >= 0.8 for c in covered) / max(len(covered), 1), 4)}
    dst = ROOT / "results/bbox/bbox_app_eval.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
