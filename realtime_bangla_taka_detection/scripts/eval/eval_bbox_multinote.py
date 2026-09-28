"""Detector quality on images that contain 2, 3, or 5 notes.

Each note is a crop from a synthetic test image with an exact box. Crops are
pasted onto a blank canvas without overlap. The test labels of JaalTaka
authenticity are not used. Live camera frames are not in this script.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT.parent
sys.path.insert(0, str(ROOT))
IMG = WS / "data set" / "currency_yolo_data" / "test" / "images"
LAB = WS / "data set" / "currency_yolo_data" / "test" / "labels"
OUT = ROOT / "results" / "bbox" / "bbox_multinote.json"
WEIGHTS = ROOT / "models" / "best.pt"


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _one_note_crops(rng: random.Random, n: int) -> list[np.ndarray]:
    paths = sorted(IMG.glob("*"))
    rng.shuffle(paths)
    crops = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label = LAB / (path.stem + ".txt")
        if not label.is_file():
            continue
        lines = [ln for ln in label.read_text().splitlines() if ln.strip()]
        if len(lines) != 1:
            continue
        _, xc, yc, bw, bh = map(float, lines[0].split())
        x1 = max(0, int((xc - bw / 2) * w))
        y1 = max(0, int((yc - bh / 2) * h))
        x2 = min(w, int((xc + bw / 2) * w))
        y2 = min(h, int((yc + bh / 2) * h))
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        crops.append(img[y1:y2, x1:x2])
        if len(crops) >= n:
            break
    return crops


def _canvas(crops: list[np.ndarray]) -> tuple[np.ndarray, list[list[int]]]:
    canvas = np.full((960, 1280, 3), 40, np.uint8)
    boxes = []
    cols = 3 if len(crops) > 3 else len(crops)
    cell_w, cell_h = 1280 // cols, 960 // ((len(crops) + cols - 1) // cols)
    for i, crop in enumerate(crops):
        row, col = divmod(i, cols)
        target_w = max(40, cell_w - 30)
        target_h = max(40, cell_h - 30)
        scale = min(target_w / crop.shape[1], target_h / crop.shape[0])
        nw, nh = max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        x1 = col * cell_w + 15
        y1 = row * cell_h + 15
        canvas[y1:y1 + nh, x1:x1 + nw] = resized
        boxes.append([x1, y1, x1 + nw, y1 + nh])
    return canvas, boxes


def main() -> None:
    from ultralytics import YOLO
    rng = random.Random(42)
    pool = _one_note_crops(rng, 40)
    if len(pool) < 5:
        raise SystemExit(f"only {len(pool)} note crops found")
    model = YOLO(str(WEIGHTS))
    report = {"weights": str(WEIGHTS), "conf": 0.25, "counts": {}}
    for count in (2, 3, 5):
        ious, found, n_gt = [], 0, 0
        for image_i in range(12):
            crops = [pool[(image_i * count + j) % len(pool)] for j in range(count)]
            canvas, gts = _canvas(crops)
            pred = model.predict(canvas, conf=0.25, imgsz=640, verbose=False)[0]
            boxes = pred.boxes.xyxy.cpu().numpy().tolist() if pred.boxes is not None and len(pred.boxes) else []
            used = set()
            for gt in gts:
                n_gt += 1
                best_i, best_v = None, 0.0
                for pi, pb in enumerate(boxes):
                    if pi in used:
                        continue
                    value = _iou(pb, gt)
                    if value > best_v:
                        best_i, best_v = pi, value
                ious.append(best_v)
                if best_v >= 0.5 and best_i is not None:
                    found += 1
                    used.add(best_i)
        report["counts"][str(count)] = {
            "images": 12,
            "notes": n_gt,
            "detected_iou_at_least_0.5": found / n_gt,
            "iou_mean": float(np.mean(ious)),
        }
        print(json.dumps({count: report["counts"][str(count)]}), flush=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("MULTI_DONE", flush=True)


if __name__ == "__main__":
    main()
