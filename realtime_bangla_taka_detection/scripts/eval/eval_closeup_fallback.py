"""Hand-held close-ups (NSTU-BDTAKA Recognition): does the MobileNet fallback help?

Pipelines (deployed weights, nothing trained here):
  P1 YOLO only (current app): top box class, else abstain
  P2 YOLO, and if YOLO finds no note, the MobileNet close-up classifier (answers only at
     softmax >= 0.65, the app's CURRENCY_CONFIDENCE), else abstain
  P3 MobileNet only (same 0.65 rule)

Rule fixed before any test number: pick the pipeline with the highest score on the NSTU
*train* split used as development data, score = correct - wrong (abstain = 0), one image per
original id. Then report every pipeline on NSTU test once, excluding the 88 test originals that
also occur in NSTU train, and on Bangla Money. Writes results/external/closeup_fallback.json.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WS / "savior_glass"))
EXT = WS / "data set for comparison"
NAMES = {0: 2, 1: 5, 2: 10, 3: 20, 4: 50, 5: 100, 6: 200, 7: 500, 8: 1000}
REC = EXT / "NSTU-BDTAKA dataset Mendeley/data/Recognition"


def oid(p: Path) -> str:
    return re.split(r"_(?:jpg|jpeg|png)\.rf\.", p.name)[0]


def one_per_original(root: Path, value, exclude=frozenset(), limit=None, seed=0):
    items = {}
    for folder in sorted(x for x in root.iterdir() if x.is_dir()):
        for p in sorted(folder.glob("*")):
            key = (folder.name, oid(p))
            if key not in exclude and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.setdefault(key, (p, value(folder.name)))
    rows = list(items.values())
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", default=None, help="MobileNet checkpoint (default: the app's currency_mobilenet.pt)")
    ap.add_argument("--dev", choices=["nstu_train_sample", "closeup_val"], default="nstu_train_sample",
                    help="closeup_val = the held-out 10%% originals of train_closeup_classifier.py (seed 42)")
    ap.add_argument("--out", default="closeup_fallback.json")
    args = ap.parse_args()
    import config
    from modes.currency_mode import CurrencyMode
    from ultralytics import YOLO
    yolo = YOLO(str(ROOT / "models/best.pt"))
    if args.classifier:
        config.CURRENCY_MODEL_PATH = args.classifier
    cm = CurrencyMode()
    cm._load_classifier()
    assert cm._classifier is not None, "MobileNet classifier missing"
    thr = config.CURRENCY_CONFIDENCE

    def run(rows):
        counts = {p: {"correct": 0, "wrong": 0, "abstain": 0} for p in ("P1", "P2", "P3")}
        for path, true in rows:
            img = cv2.imread(str(path))
            r = yolo.predict(img, conf=0.25, imgsz=640, verbose=False)[0]
            y = NAMES[int(r.boxes.cls[int(r.boxes.conf.argmax())])] if r.boxes is not None and len(r.boxes) else None
            m, s = cm._classify(img)
            mob = m if s >= thr else None
            for name, pred in (("P1", y), ("P2", y if y is not None else mob), ("P3", mob)):
                counts[name]["abstain" if pred is None else ("correct" if pred == true else "wrong")] += 1
        n = len(rows)
        return {k: {**v, "n": n, "accuracy": round(v["correct"] / n, 4), "score": v["correct"] - v["wrong"],
                    "precision_when_answering": round(v["correct"] / max(v["correct"] + v["wrong"], 1), 4)}
                for k, v in counts.items()}

    val = lambda name: int(name.split("_")[0])
    train_keys = {(f.name, oid(p)) for f in (REC / "train").iterdir() if f.is_dir() for p in f.glob("*")}
    if args.dev == "closeup_val":
        sys.path.insert(0, str(ROOT / "scripts" / "train"))
        from train_closeup_classifier import CLASSES, split_train
        dev = run([(p, CLASSES[y]) for p, y in split_train(42)[1]])
    else:
        dev = run(one_per_original(REC / "train", val, limit=1500))
    chosen = max(dev, key=lambda k: dev[k]["score"])
    test = run(one_per_original(REC / "test", val, exclude=train_keys))
    bm = run([r for r in one_per_original(EXT / "Bangla Money dataset Kaggle/bangla_banknote_v2/Training", int) if r[1] != 1])
    out = {"classifier": args.classifier or config.CURRENCY_MODEL_PATH, "dev": args.dev,
           "rule": "max(correct - wrong) on the dev set, one image per original; threshold 0.65 = app default",
           "dev_nstu_train": dev, "chosen": chosen, "test_nstu_excluding_88_leaked_originals": test,
           "bangla_money_training_folders": bm}
    dst = ROOT / "results/external" / args.out
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
