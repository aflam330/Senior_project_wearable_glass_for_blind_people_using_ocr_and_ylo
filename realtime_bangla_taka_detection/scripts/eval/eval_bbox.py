"""Bounding-box quality of the Taka detector, and of the box the glass app actually uses.

Sets:
  synthetic  data set/currency_yolo_data/test: exact boxes from compositing; 230 images have
             no note (false-positive check)
  real       data set/yolo_bbox_annotated/test: raw phone photos, boxes from a heuristic
             (background-difference contour); labels that fell back to the full frame are skipped

For each image with a note: IoU of the top-confidence box with the label, class match,
predicted/true area ratio, and mean edge error as a share of the label's width/height.
For note-free images: share with any box at conf >= 0.25 / 0.35 / 0.5.
App path: CurrencyMode.detect_live() returns (x, y, w, h); its IoU must equal the raw one.
Writes results/bbox/bbox_eval.json and a grid of the worst real-photo cases.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "bbox"


def read_labels(path: Path, w: int, h: int):
    boxes = []
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                c, xc, yc, bw, bh = map(float, line.split())
                boxes.append((int(c), (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
    return boxes


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def summarize(rows):
    ious = [r["iou"] for r in rows]
    if not ious:
        return None
    return {
        "n": len(ious),
        "detected": sum(r["detected"] for r in rows) / len(rows),
        "iou_mean": round(statistics.fmean(ious), 4), "iou_median": round(statistics.median(ious), 4),
        "iou>=0.5": round(sum(i >= 0.5 for i in ious) / len(ious), 4),
        "iou>=0.75": round(sum(i >= 0.75 for i in ious) / len(ious), 4),
        "iou>=0.9": round(sum(i >= 0.9 for i in ious) / len(ious), 4),
        "class_correct_when_detected": round(statistics.fmean([r["class_ok"] for r in rows if r["detected"]]), 4),
        "area_ratio_median(pred/true)": round(statistics.median([r["area_ratio"] for r in rows if r["detected"]]), 3),
        "edge_error_median(share of box size)": round(statistics.median([r["edge_err"] for r in rows if r["detected"]]), 4),
    }


def evaluate(model, img_dir: Path, lab_dir: Path, limit: int | None, skip_fullframe: bool):
    rows, negatives = [], []
    paths = sorted(img_dir.glob("*"))
    random.Random(0).shuffle(paths)
    for p in paths[:limit]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = read_labels(lab_dir / (p.stem + ".txt"), w, h)
        if skip_fullframe:
            gts = [g for g in gts if not ((g[3] - g[1]) * (g[4] - g[2]) > 0.97 * w * h)]
            if not gts and (lab_dir / (p.stem + ".txt")).is_file():
                continue
        r = model.predict(img, conf=0.05, imgsz=640, verbose=False)[0]
        b = r.boxes
        confs = b.conf.cpu().numpy() if b is not None and len(b) else np.zeros(0)
        if not gts:
            negatives.append({f"fp@{t}": bool((confs >= t).any()) for t in (0.25, 0.35, 0.5)})
            continue
        keep = confs >= 0.25
        if not keep.any():
            rows.append({"img": str(p), "detected": 0, "iou": 0.0, "class_ok": 0, "area_ratio": 0, "edge_err": 1})
            continue
        i = int(np.argmax(np.where(keep, confs, -1)))
        pb = b.xyxy[i].cpu().numpy().tolist()
        pc = int(b.cls[i])
        g = max(gts, key=lambda g: iou(pb, g[1:]))
        gb = g[1:]
        gw, gh = gb[2] - gb[0], gb[3] - gb[1]
        edge = statistics.fmean([abs(pb[0] - gb[0]) / gw, abs(pb[2] - gb[2]) / gw, abs(pb[1] - gb[1]) / gh, abs(pb[3] - gb[3]) / gh])
        rows.append({"img": str(p), "detected": 1, "iou": iou(pb, gb), "class_ok": int(pc == g[0]),
                     "area_ratio": (pb[2] - pb[0]) * (pb[3] - pb[1]) / (gw * gh), "edge_err": edge,
                     "pred": pb, "gt": list(gb), "conf": float(confs[i])})
    neg = None
    if negatives:
        neg = {k: round(sum(n[k] for n in negatives) / len(negatives), 4) for k in negatives[0]}
        neg["n"] = len(negatives)
    return rows, neg


def main() -> None:
    from ultralytics import YOLO
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    syn = WS / "data set/currency_yolo_data/test"
    real = WS / "data set/yolo_bbox_annotated/test"
    worst_real = None
    for wname in ("best.pt", "best_int8.onnx"):
        m = YOLO(str(ROOT / "models" / wname), task="detect")
        rows, neg = evaluate(m, syn / "images", syn / "labels", None if wname == "best.pt" else 600, False)
        report[f"synthetic_{wname}"] = {"with_note": summarize(rows), "no_note_false_positive_rate": neg}
        rows_r, _ = evaluate(m, real / "images", real / "labels", None, True)
        report[f"real_{wname}"] = {"with_note": summarize(rows_r),
                                   "note": "labels are heuristic; full-frame fallback labels skipped"}
        if wname == "best.pt":
            worst_real = sorted([r for r in rows_r], key=lambda r: r["iou"])[:8]
        print(wname, json.dumps({k: v for k, v in report.items() if wname in k}, indent=1), flush=True)

    # app path: CurrencyMode.detect_live bbox (x, y, w, h) must match the raw detector
    sys.path.insert(0, str(WS / "savior_glass"))
    from modes.currency_mode import CurrencyMode
    cm = CurrencyMode()
    cm._load_yolo()
    raw = YOLO(str(ROOT / "models/best.pt"))
    diffs, checked = [], 0
    for p in sorted((syn / "images").glob("*"))[:150]:
        img = cv2.imread(str(p))
        hits = cm.detect_live(img)
        r = raw.predict(img, conf=0.25, imgsz=640, verbose=False)[0]
        if not hits or r.boxes is None or not len(r.boxes):
            continue
        x, y, w, h = hits[0]["bbox"]
        pb = r.boxes.xyxy[int(r.boxes.conf.argmax())].cpu().numpy()
        diffs.append(iou([x, y, x + w, y + h], pb.tolist()))
        checked += 1
    report["app_detect_live_vs_raw"] = {"images": checked, "min_iou": round(min(diffs), 4) if diffs else None,
                                        "mean_iou": round(statistics.fmean(diffs), 4) if diffs else None,
                                        "yolo_path_loaded": str(getattr(cm._yolo, "ckpt_path", "") or "")}

    # grid of the worst real-photo boxes (green = model, red = heuristic label)
    tiles = []
    for r in worst_real or []:
        img = cv2.imread(r["img"])
        if r.get("gt"):
            g = [int(v) for v in r["gt"]]
            cv2.rectangle(img, (g[0], g[1]), (g[2], g[3]), (0, 0, 255), max(2, img.shape[1] // 200))
        if r.get("pred"):
            pb = [int(v) for v in r["pred"]]
            cv2.rectangle(img, (pb[0], pb[1]), (pb[2], pb[3]), (0, 255, 0), max(2, img.shape[1] // 200))
        img = cv2.resize(img, (400, int(400 * img.shape[0] / img.shape[1])))
        img = cv2.copyMakeBorder(img, 0, max(0, 400 - img.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:400]
        cv2.putText(img, f"IoU {r['iou']:.2f}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(img)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(OUT / "worst_real_boxes.jpg"), grid)
        report["worst_real_grid"] = str(OUT / "worst_real_boxes.jpg")
    (OUT / "bbox_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["app_detect_live_vs_raw"]))


if __name__ == "__main__":
    main()
