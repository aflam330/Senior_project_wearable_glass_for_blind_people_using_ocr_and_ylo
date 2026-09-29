"""Cross-dataset test of the Taka detector on Bangladeshi datasets it was never trained on.

Nothing here trains, tunes or selects anything; models and thresholds (conf 0.25) are the
deployed ones. Sets (in "data set for comparison"):

  NSTU-BDTAKA Detection/test   186 images, one class "Taka" (no denomination): box finding,
                               class-agnostic (precision, recall, IoU of matches at IoU >= 0.5, AP50)
  NSTU-BDTAKA Recognition/test 9 denomination folders (2..1000): denomination = class of the
                               top-confidence box ("none" if no box)
  Bangla Money Training         denomination folders; "1" (1 taka) is outside the detector's
                               9 classes and is reported separately

"A Diverse Image Dataset for Bangladeshi Currency Recognition" is byte-identical to the
BanglaTaka source used to build the detector's training data, so it is not an external test.
NSTU filenames "<id>_jpg.rf.<hash>" are Roboflow copies; results are also given with one image
per original id. Writes results/external/external_taka.json.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.eval_bbox import iou, read_labels  # noqa: E402

EXT = ROOT.parent / "data set for comparison"
NAMES = {0: 2, 1: 5, 2: 10, 3: 20, 4: 50, 5: 100, 6: 200, 7: 500, 8: 1000}


def original_id(name: str) -> str:
    return re.split(r"_(?:jpg|jpeg|png)\.rf\.", name)[0]


def detection_set(model):
    d = EXT / "NSTU-BDTAKA dataset Mendeley/data/Detection/test"
    preds, n_gt, ious = [], 0, []
    for p in sorted((d / "images").glob("*")):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gts = [g[1:] for g in read_labels(d / "labels" / (p.stem + ".txt"), w, h)]
        n_gt += len(gts)
        r = model.predict(img, conf=0.001, imgsz=640, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        used = set()
        for i in np.argsort(-confs):
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if j not in used:
                    v = iou(boxes[i].tolist(), g)
                    if v > best:
                        best, bj = v, j
            tp = best >= 0.5
            if tp:
                used.add(bj)
            preds.append((float(confs[i]), tp, best))
    preds.sort(key=lambda x: -x[0])
    tps = np.cumsum([t for _, t, _ in preds])
    fps = np.cumsum([not t for _, t, _ in preds])
    rec = tps / max(n_gt, 1)
    prec = tps / np.maximum(tps + fps, 1)
    ap = 0.0  # all-point interpolated AP50
    for r0 in np.linspace(0, 1, 101):
        ap += (prec[rec >= r0].max() if (rec >= r0).any() else 0.0) / 101
    at = [(c, t, v) for c, t, v in preds if c >= 0.25]
    tp25 = sum(t for _, t, _ in at)
    return {"images": len(list((d / "images").glob("*"))), "gt_boxes": n_gt,
            "AP50_class_agnostic": round(float(ap), 4),
            "at_conf_0.25": {"precision": round(tp25 / max(len(at), 1), 4), "recall": round(tp25 / max(n_gt, 1), 4),
                             "mean_iou_of_matches": round(float(np.mean([v for _, t, v in at if t])), 4) if tp25 else None}}


def denomination_set(model, root: Path, folder_to_value):
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        value = folder_to_value(folder.name)
        for p in sorted(folder.glob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            r = model.predict(img, conf=0.25, imgsz=640, verbose=False)[0]
            pred = None
            if r.boxes is not None and len(r.boxes):
                pred = NAMES[int(r.boxes.cls[int(r.boxes.conf.argmax())])]
            rows.append({"true": value, "pred": pred, "orig": f"{folder.name}/{original_id(p.name)}"})
    return rows


def summarize(rows, known):
    def acc(rs):
        return round(sum(r["pred"] == r["true"] for r in rs) / max(len(rs), 1), 4)
    inside = [r for r in rows if r["true"] in known]
    first = {}
    for r in inside:
        first.setdefault(r["orig"], r)
    per = defaultdict(list)
    for r in inside:
        per[r["true"]].append(r)
    out = {"images": len(inside), "top1_accuracy": acc(inside),
           "no_detection_rate": round(sum(r["pred"] is None for r in inside) / max(len(inside), 1), 4),
           "accuracy_when_detected": acc([r for r in inside if r["pred"] is not None]),
           "unique_originals": len(first), "top1_accuracy_one_image_per_original": acc(list(first.values())),
           "per_class_accuracy": {str(k): acc(v) for k, v in sorted(per.items())},
           "confusions_top": Counter(f"{r['true']}->{r['pred']}" for r in inside if r["pred"] != r["true"]).most_common(8)}
    outside = [r for r in rows if r["true"] not in known]
    if outside:
        out["outside_classes"] = {"images": len(outside), "predictions": Counter(str(r["pred"]) for r in outside).most_common()}
    return out


def main() -> None:
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "models/best.pt"))
    known = set(NAMES.values())
    res = {"model": "models/best.pt", "conf": 0.25, "note": "no training, tuning or selection on these sets"}
    res["nstu_detection_test"] = detection_set(model)
    print("NSTU detection", json.dumps(res["nstu_detection_test"]), flush=True)
    rows = denomination_set(model, EXT / "NSTU-BDTAKA dataset Mendeley/data/Recognition/test",
                            lambda n: int(n.split("_")[0]))
    res["nstu_recognition_test"] = summarize(rows, known)
    print("NSTU recognition", json.dumps(res["nstu_recognition_test"]), flush=True)
    rows = denomination_set(model, EXT / "Bangla Money dataset Kaggle/bangla_banknote_v2/Training", lambda n: int(n))
    res["bangla_money_training_folders"] = summarize(rows, known)
    print("Bangla Money", json.dumps(res["bangla_money_training_folders"]), flush=True)
    dst = ROOT / "results/external/external_taka.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
