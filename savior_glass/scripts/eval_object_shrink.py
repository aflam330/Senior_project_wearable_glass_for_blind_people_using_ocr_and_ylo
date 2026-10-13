"""Object mode: does shrinking the camera picture to 320 x 240 before detection help or hurt?

modes/object_mode.py resized every frame to 320 x 240 "for fast inference" and then called the detector with its
default input size (640), which enlarges the small picture again. So the detector did the same amount of work on
half the detail. This script measures both paths on the whole COCO val2017 set with the glass's model (YOLOv8s),
each picture first brought to the camera's size (640 x 480), and times both on the CPU with four threads.

    python scripts/eval_object_shrink.py
Output: results/object_shrink_vs_full.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from publish_boost import iou_xyxy, voc_ap  # noqa: E402
import config  # noqa: E402

COCO = HERE.parents[1] / "data set" / "coco2017"


def ap_all(gt, pred, n_cls):
    aps = []
    for c in range(n_cls):
        g = gt.get(c, [])
        if not g:
            continue
        by = defaultdict(list)
        for j, (ii, b) in enumerate(g):
            by[ii].append((j, b))
        p = sorted(pred.get(c, []), key=lambda z: -z[1])
        used, tp = set(), np.zeros(len(p))
        for k, (ii, s, box) in enumerate(p):
            best, bj = 0.0, -1
            for j, b in by.get(ii, []):
                if j not in used:
                    v = iou_xyxy(box, b)
                    if v > best:
                        best, bj = v, j
            if best >= 0.5:
                tp[k] = 1
                used.add(bj)
        tpc, fpc = np.cumsum(tp), np.cumsum(1 - tp)
        aps.append(float(voc_ap(tpc / len(g), tpc / np.maximum(tpc + fpc, 1e-9))) if len(p) else 0.0)
    return float(np.mean(aps))


def main():
    import torch
    from ultralytics import YOLO
    coco = json.loads((COCO / "annotations" / "instances_val2017.json").read_text(encoding="utf-8"))
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cid = {c["id"]: i for i, c in enumerate(cats)}
    by_img = defaultdict(list)
    for a in coco["annotations"]:
        if not a.get("iscrowd"):
            x, y, w, h = a["bbox"]
            by_img[a["image_id"]].append((cid[a["category_id"]], [x, y, x + w, y + h]))
    images = [im for im in coco["images"] if (COCO / "val2017" / im["file_name"]).is_file()]
    model = YOLO(config.YOLO_MODEL_PATH)
    gt = defaultdict(list)
    pred = {"shrunk to 320x240 (before)": defaultdict(list), "camera frame as it is (after)": defaultdict(list)}
    announce = {k: 0 for k in pred}
    for i, im in enumerate(images):
        img = cv2.imread(str(COCO / "val2017" / im["file_name"]))
        H, W = img.shape[:2]
        frame = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)          # what the camera delivers
        fx, fy = W / 640.0, H / 480.0
        for c, b in by_img.get(im["id"], []):
            gt[c].append((i, b))
        small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        for name, inp, sx, sy in (("shrunk to 320x240 (before)", small, 2 * fx, 2 * fy), ("camera frame as it is (after)", frame, fx, fy)):
            r = model.predict(inp, conf=0.15, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                    pred[name][int(c)].append((i, s, [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]))
                    announce[name] += int(s >= config.OBJECT_CONFIDENCE)
        if i % 1000 == 0:
            print(i, flush=True)
    res = {"date": "2026-10-13", "images": len(images), "model": Path(config.YOLO_MODEL_PATH).name, "paths": {}}
    for name in pred:
        res["paths"][name] = {"map50_all_80": round(ap_all(gt, pred[name], len(cats)), 4),
                              "detections_at_announce_confidence": announce[name]}
    # CPU time, four threads, 60 pictures each after warm-up
    torch.set_num_threads(4)
    cpu = YOLO(config.YOLO_MODEL_PATH)
    frames = [cv2.resize(cv2.imread(str(COCO / "val2017" / im["file_name"])), (640, 480)) for im in images[:64]]
    for name, prep in (("shrunk to 320x240 (before)", lambda f: cv2.resize(f, (320, 240), interpolation=cv2.INTER_AREA)),
                       ("camera frame as it is (after)", lambda f: f)):
        t = []
        for k, f in enumerate(frames):
            t0 = time.perf_counter()
            cpu.predict(prep(f), conf=config.OBJECT_CONFIDENCE, verbose=False, device="cpu")
            if k >= 4:
                t.append(time.perf_counter() - t0)
        res["paths"][name]["cpu_ms_median_4_threads"] = round(1000 * float(np.median(t)), 1)
    out = HERE.parent / "results" / "object_shrink_vs_full.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
