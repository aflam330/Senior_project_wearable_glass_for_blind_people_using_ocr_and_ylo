"""UV note photographs: rename, label with a bounding box, split, set the threshold, test.

    python scripts/uv_dataset.py prepare     # rename to uv_<value>_<nnn>.png, write labels/ and previews/
    python scripts/uv_dataset.py evaluate    # threshold from the calibration part, results on the test part

Folder: <project>/data set/uv_notes/{200,500,1000}/  (photographs of genuine notes under a UV lamp)

Rules (same as the rest of the project):
  * The split is by time: the first 60 % of each value's photographs (by capture order) are the calibration part,
    the last 40 % the test part. Neighbouring photographs show the same note, so a random split would leak.
  * UV_MIN_FIBRES is chosen on the calibration part only; the test part is read once.
  * There are no counterfeit notes in this folder. The "no fibres" side is measured on patches of the plain surface
    beside the note in the same photographs (same lamp, same camera). That shows the check rejects plain paper; it
    does not show what it does with a real counterfeit.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config  # noqa: E402
from modes.uv_check import fibre_mask, inner, paper_ratios  # noqa: E402

ROOT = HERE.parents[1]
DATA = ROOT / "data set" / "uv_notes"
OUT = HERE.parent / "results" / "uv_check"
VALUES = ("200", "500", "1000")
DETECTOR = ROOT / "realtime_bangla_taka_detection" / "models" / "best.pt"


def read(path: Path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def images(value: str):
    return sorted(p for p in (DATA / value).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))


def prepare():
    from ultralytics import YOLO
    det = YOLO(str(DETECTOR))
    mapping, boxes = [], []
    done = DATA / "rename_map.csv"
    if done.exists():
        print("already renamed (rename_map.csv exists); only labels and previews are rebuilt")
    for cls, value in enumerate(VALUES):
        files = images(value)
        if not done.exists():
            # capture order: the phone's own time stamp is at the start of the name; screenshots sort after them
            tmp = []
            for i, f in enumerate(files, 1):
                t = f.with_name(f"__tmp_{i:03d}{f.suffix.lower()}")
                f.rename(t)
                tmp.append((f.name, t))
            for i, (old, t) in enumerate(tmp, 1):
                new = t.with_name(f"uv_{value}_{i:03d}.png")
                t.rename(new)
                mapping.append([value, old, new.name])
            files = images(value)
        (DATA / "labels" / value).mkdir(parents=True, exist_ok=True)
        (DATA / "previews" / value).mkdir(parents=True, exist_ok=True)
        n_cal = int(round(0.6 * len(files)))
        for i, f in enumerate(files):
            img = read(f)
            H, W = img.shape[:2]
            r = det.predict(img, conf=0.10, verbose=False, imgsz=640)[0]
            split = "calibrate" if i < n_cal else "test"
            label = DATA / "labels" / value / (f.stem + ".txt")
            if len(r.boxes):
                j = int(r.boxes.conf.argmax())
                x1, y1, x2, y2 = [float(v) for v in r.boxes.xyxy[j].cpu().numpy()]
                label.write_text(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
                boxes.append([value, f.name, split, int(x1), int(y1), int(x2), int(y2), round(float(r.boxes.conf[j]), 3)])
                vis = img.copy()
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), max(2, W // 300))
                mask, n, _ = fibre_mask(inner(img[int(y1):int(y2), int(x1):int(x2)]))
                cv2.putText(vis, f"{value} Taka  fibres {n}", (int(x1), max(30, int(y1) - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            W / 900, (0, 255, 255), max(2, W // 400))
            else:
                label.write_text("")
                boxes.append([value, f.name, split, "", "", "", "", ""])
                vis = img
            s = 640 / max(H, W)
            cv2.imwrite(str(DATA / "previews" / value / (f.stem + ".jpg")), cv2.resize(vis, None, fx=s, fy=s),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    if mapping:
        with open(done, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows([["value", "old_name", "new_name"]] + mapping)
    with open(DATA / "boxes.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["value", "file", "split", "x1", "y1", "x2", "y2", "detector_conf"]] + boxes)
    (DATA / "labels" / "classes.txt").write_text("200_taka\n500_taka\n1000_taka\n")
    found = sum(1 for b in boxes if b[3] != "")
    print(f"{len(boxes)} photographs; note box found on {found}; labels in {DATA / 'labels'}; previews in {DATA / 'previews'}")


def _surface_patch(img, x1, y1, x2, y2):
    """Plain surface beside the note, same rows; None when there is too little of it."""
    W = img.shape[1]
    bw, bh = x2 - x1, y2 - y1
    dy = int(0.10 * bh)
    left, right = x1, W - x2
    if max(left, right) < 0.25 * bw:
        return None
    p = img[y1 + dy:y2 - dy, int(x2 + 0.05 * bw):W - 4] if right >= left else img[y1 + dy:y2 - dy, 4:int(x1 - 0.05 * bw)]
    return p if p.size and p.shape[1] >= 40 and paper_ratios(p) is not None else None


def _variants(crop, rng):
    """The same crop under other conditions (none of them can add fibres)."""
    f = crop.astype(np.float32)
    yield "as photographed", crop
    yield "half as bright", np.clip(f * 0.5, 0, 255).astype(np.uint8)
    yield "1.5 times brighter", np.clip(f * 1.5, 0, 255).astype(np.uint8)
    yield "colour cast (blue x0.8, green x1.2)", np.clip(f * np.array([0.8, 1.2, 1.0], np.float32), 0, 255).astype(np.uint8)
    yield "slightly out of focus (sigma 1.5)", cv2.GaussianBlur(crop, (0, 0), 1.5)
    yield "sensor noise (sigma 6)", np.clip(f + rng.normal(0, 6, f.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 50])
    yield "JPEG quality 50", cv2.imdecode(enc, cv2.IMREAD_COLOR)
    h, w = crop.shape[:2]
    yield "half the resolution", cv2.resize(cv2.resize(crop, (w // 2, h // 2), interpolation=cv2.INTER_AREA), (w, h))


def evaluate():
    rows = list(csv.DictReader(open(DATA / "boxes.csv", encoding="utf-8")))
    rng = np.random.default_rng(42)
    notes, surf = {"calibrate": [], "test": []}, {"calibrate": [], "test": []}
    crops = []
    for r in rows:
        if r["x1"] == "":
            continue
        img = read(DATA / r["value"] / r["file"])
        x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
        crop = inner(img[y1:y2, x1:x2])
        _, n, ab = fibre_mask(crop)
        if ab is None:
            continue
        notes[r["split"]].append((r["value"], r["file"], n))
        if r["split"] == "test":
            crops.append((r["value"], crop))
        p = _surface_patch(img, x1, y1, x2, y2)
        if p is not None:
            surf[r["split"]].append(fibre_mask(p)[1])
    cal = np.array([n for _, _, n in notes["calibrate"]])
    cal_s = np.array(surf["calibrate"])
    # rule fixed before the test part is read: the largest threshold that still accepts at least 95 % of the
    # calibration notes, and never below (most fibres seen on plain surface) + 2
    floor = int(cal_s.max()) + 2 if len(cal_s) else 2
    thr = max(floor, int(np.floor(np.percentile(cal, 5))))
    thr = max(floor, min(thr, int(np.sort(cal)[int(np.floor(0.05 * len(cal)))])))
    te = np.array([n for _, _, n in notes["test"]])
    te_s = np.array(surf["test"])

    def wilson(k, n, z=1.96):
        if n == 0:
            return [None, None]
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return [round(float(c - h), 4), round(float(c + h), 4)]

    per_value = {}
    for v in VALUES:
        a = np.array([n for val, _, n in notes["test"] if val == v])
        per_value[v] = {"n": int(len(a)), "accepted": int((a >= thr).sum()), "median_fibres": float(np.median(a)) if len(a) else None}
    robust = {}
    for v, crop in crops:
        for name, img in _variants(crop, rng):
            n = fibre_mask(img)[1]
            k = robust.setdefault(name, [0, 0])
            k[0] += int(n >= thr)
            k[1] += 1
    res = {
        "date": "2026-10-12", "photographs": len(rows), "note_box_found": sum(1 for r in rows if r["x1"] != ""),
        "split": "by capture order per value: first 60 % calibrate, last 40 % test",
        "threshold_rule": "largest value accepting >= 95 % of calibration notes, at least (max surface fibres) + 2",
        "UV_MIN_FIBRES": int(thr),
        "calibrate": {"notes": int(len(cal)), "accepted": int((cal >= thr).sum()), "fibres_percentiles_5_50_95": np.percentile(cal, [5, 50, 95]).round(1).tolist(),
                      "surface_patches": int(len(cal_s)), "surface_max_fibres": int(cal_s.max()) if len(cal_s) else None},
        "test": {"notes": int(len(te)), "accepted": int((te >= thr).sum()), "accepted_rate": round(float((te >= thr).mean()), 4),
                 "accepted_wilson95": wilson(int((te >= thr).sum()), len(te)),
                 "fibres_percentiles_5_50_95": np.percentile(te, [5, 50, 95]).round(1).tolist(), "per_value": per_value,
                 "surface_patches": int(len(te_s)), "surface_rejected": int((te_s < thr).sum()),
                 "surface_rejected_wilson95": wilson(int((te_s < thr).sum()), len(te_s)),
                 "surface_max_fibres": int(te_s.max()) if len(te_s) else None},
        "test_under_other_conditions": {k: {"accepted": v[0], "n": v[1], "rate": round(v[0] / v[1], 4)} for k, v in robust.items()},
        "not_measured": ["counterfeit notes (none available)", "the glass camera and the lamp on the glass",
                         "notes of other values", "worn or washed notes as a separate group"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "uv_fibre_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"prepare": prepare, "evaluate": evaluate}[sys.argv[1] if len(sys.argv) > 1 else "evaluate"]()
