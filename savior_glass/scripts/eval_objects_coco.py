"""Object models on the whole COCO val2017 set (5,000 photographs), same AP@0.5 code as the earlier 250-image run
(scripts/publish_boost.py track_object), for three model sizes and two picture sizes.

Reported for all 80 classes and for the 30 classes below, chosen before the run as the ones a blind pedestrian or
a person at home meets most (declared here, not picked by their scores). No model is trained or changed.

    python scripts/eval_objects_coco.py
Output: results/object_coco_val2017_full.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from publish_boost import iou_xyxy, voc_ap  # noqa: E402

ROOT = HERE.parents[1]
COCO = ROOT / "data set" / "coco2017"
BLIND_USER = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign", "bench", "dog", "cat",
              "backpack", "umbrella", "handbag", "bottle", "cup", "bowl", "chair", "couch", "bed", "dining table", "toilet",
              "tv", "laptop", "cell phone", "book", "clock", "sink", "refrigerator", "potted plant"]
MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
SIZES = [640, 320]


def main():
    import torch
    from ultralytics import YOLO
    coco = json.loads((COCO / "annotations" / "instances_val2017.json").read_text(encoding="utf-8"))
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cid = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]
    assert all(n in names for n in BLIND_USER), [n for n in BLIND_USER if n not in names]
    by_img = defaultdict(list)
    for a in coco["annotations"]:
        if a.get("iscrowd"):
            continue
        x, y, w, h = a["bbox"]
        by_img[a["image_id"]].append((cid[a["category_id"]], [x, y, x + w, y + h]))
    images = [im for im in coco["images"] if (COCO / "val2017" / im["file_name"]).is_file()]
    out = HERE.parent / "results" / "object_coco_val2017_full.json"
    res = {"date": "2026-10-13", "images": len(images), "blind_user_classes": BLIND_USER,
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "runs": []}
    if out.exists():                       # finished runs are kept: a crash does not lose them
        res["runs"] = json.loads(out.read_text(encoding="utf-8"))["runs"]
    done = {(r["weights"], r["imgsz"]) for r in res["runs"]}
    for w in MODELS:
        local = HERE.parent / "models" / w
        model = YOLO(str(local) if local.is_file() else w)
        assert [model.names[i] for i in range(80)] == names, "class order differs"
        for size in SIZES:
            if (w, size) in done:
                continue
            gt, pred, t = defaultdict(list), defaultdict(list), []
            for i, im in enumerate(images):
                for c, box in by_img.get(im["id"], []):
                    gt[c].append((i, box))
                t0 = time.perf_counter()
                try:
                    r = model.predict(str(COCO / "val2017" / im["file_name"]), conf=0.15, verbose=False, imgsz=size)[0]
                except Exception as exc:       # GPU memory taken by another program: this picture on the CPU
                    torch.cuda.empty_cache()
                    print("retry on cpu:", im["file_name"], type(exc).__name__, flush=True)
                    r = model.predict(str(COCO / "val2017" / im["file_name"]), conf=0.15, verbose=False, imgsz=size, device="cpu")[0]
                t.append(time.perf_counter() - t0)
                if r.boxes is not None and len(r.boxes):
                    for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                        pred[int(c)].append((i, s, b))
            per = {}
            for c, name in enumerate(names):
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
                        if j in used:
                            continue
                        v = iou_xyxy(box, b)
                        if v > best:
                            best, bj = v, j
                    if best >= 0.5:
                        tp[k] = 1
                        used.add(bj)
                tpc = np.cumsum(tp)
                fpc = np.cumsum(1 - tp)
                per[name] = float(voc_ap(tpc / len(g), tpc / np.maximum(tpc + fpc, 1e-9))) if len(p) else 0.0
            run = {"weights": w, "imgsz": size, "map50_all_80": round(float(np.mean(list(per.values()))), 4),
                   "map50_blind_user_30": round(float(np.mean([per[n] for n in BLIND_USER])), 4),
                   "gpu_ms_median": round(1000 * float(np.median(t[20:])), 1),
                   "per_class_blind_user": {n: round(per[n], 3) for n in BLIND_USER}}
            res["runs"].append(run)
            out.write_text(json.dumps(res, indent=1), encoding="utf-8")
            print({k: v for k, v in run.items() if k != "per_class_blind_user"}, flush=True)
    print("written", out)


if __name__ == "__main__":
    main()
