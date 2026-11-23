"""Training set for re-training the hand-held detector with NSTU-BDTAKA Recognition TRAIN photos.

NSTU Recognition has the denomination (folder) but no boxes; NSTU Detection has boxes but one class only. So each NSTU
train photo gets a weak box: the top box of the current hand-held detector (any class, conf >= 0.05), else the whole
picture; its CLASS is always the folder label, never the detector's guess.

Separation (test stays locked; validation is used only for choosing the epoch and threshold):
  * one image per original photo (Roboflow writes up to three augmented copies of each);
  * a train original is dropped if the same (denomination, original id) occurs in NSTU validation or test, or if its
    difference hash is within 6 bits of any validation/test photo of the same denomination (near duplicate);
  * to keep whole notes in scenes, 4,000 synthetic composites (currency_yolo_data/train) and 2,000 earlier close-ups
    (currency_closeups/train) are added, a fixed random sample (seed 0).
Labels are not changed and no validation/test photo is removed or relabelled.

Output: data set/nstu_weak_yolo/{train,valid}/{images,labels}, data/data_nstu_weak.yaml, results/nstu_weak/prep.json
    python scripts/dataset/make_nstu_weak.py
"""
from __future__ import annotations

import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT.parent / "data set"
NSTU = ROOT.parent / "data set for comparison" / "NSTU-BDTAKA dataset Mendeley" / "data" / "Recognition"
OUT = DS / "nstu_weak_yolo"
NAMES = ["2_taka", "5_taka", "10_taka", "20_taka", "50_taka", "100_taka", "200_taka", "500_taka", "1000_taka"]
IMG = {".jpg", ".jpeg", ".png"}


def oid(name):
    return re.split(r"_(?:jpg|jpeg|png)\.rf\.", name)[0]


def originals(split):
    items = {}
    for folder in sorted(p for p in (NSTU / split).iterdir() if p.is_dir()):
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMG:
                items.setdefault((folder.name, oid(p.name)), p)
    return items


def dhash(path, n=8):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (n + 1, n), interpolation=cv2.INTER_AREA)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())


def ham(a, b):
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())


def main():
    from ultralytics import YOLO
    det = YOLO(str(ROOT / "models" / "best_handheld.pt"))
    tr, va, te = originals("train"), originals("validation"), originals("test")
    held = {**{("v",) + k: p for k, p in va.items()}, **{("t",) + k: p for k, p in te.items()}}
    held_hash = {}
    for k, p in held.items():
        held_hash.setdefault(k[1], []).append(dhash(p))
    drop_id = drop_hash = 0
    keep = []
    for (cls, o), p in tr.items():
        if (cls, o) in va or (cls, o) in te:
            drop_id += 1
            continue
        h = dhash(p)
        if any(ham(h, x) <= 6 for x in held_hash.get(cls, [])):
            drop_hash += 1
            continue
        keep.append((cls, p))
    if OUT.exists():
        shutil.rmtree(OUT)
    stats = Counter()

    def write(split, cls, p, src_tag):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        r = det.predict(img, conf=0.05, imgsz=640, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            i = int(r.boxes.conf.argmax())
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            stats["box_from_detector"] += 1
        else:
            x1, y1, x2, y2 = 0, 0, w, h
            stats["box_whole_image"] += 1
        name = f"{src_tag}_{cls}_{p.stem}"
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, OUT / split / "images" / f"{name}{p.suffix}")
        cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
        (OUT / split / "labels" / f"{name}.txt").write_text(f"{NAMES.index(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    for cls, p in keep:
        write("train", cls, p, "nstu")
    for (cls, o), p in va.items():
        write("valid", cls, p, "nstuval")
    rnd = random.Random(0)
    for sub, n in (("currency_yolo_data", 4000), ("currency_closeups", 2000)):
        imgs = sorted((DS / sub / "train" / "images").glob("*.jpg"))
        for p in rnd.sample(imgs, min(n, len(imgs))):
            lab = DS / sub / "train" / "labels" / f"{p.stem}.txt"
            (OUT / "train" / "images").mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, OUT / "train" / "images" / f"{sub}_{p.name}")
            shutil.copy2(lab, OUT / "train" / "labels" / f"{sub}_{p.stem}.txt") if lab.exists() else \
                (OUT / "train" / "labels" / f"{sub}_{p.stem}.txt").write_text("")
            stats[f"added_{sub}"] += 1
    yaml = ROOT / "data" / "data_nstu_weak.yaml"
    yaml.write_text("# hand-held detector re-training: NSTU train (weak boxes, folder labels) + synthetic sample\n"
                    f"path: {OUT.as_posix()}\ntrain: train/images\nval: valid/images\n\nnc: 9\nnames:\n"
                    + "".join(f"  - {n}\n" for n in NAMES), encoding="utf-8")
    res = {"nstu_train_originals": len(tr), "dropped_same_id_in_val_or_test": drop_id,
           "dropped_near_duplicate_of_val_or_test": drop_hash, "nstu_train_kept": len(keep),
           "nstu_val_originals": len(va), "nstu_test_originals_locked": len(te),
           "per_class_kept": dict(Counter(c for c, _ in keep)), **stats}
    (ROOT / "results" / "nstu_weak").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "nstu_weak" / "prep.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
