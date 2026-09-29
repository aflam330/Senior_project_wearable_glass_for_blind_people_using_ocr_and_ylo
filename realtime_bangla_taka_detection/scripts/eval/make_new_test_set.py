"""New detector test set: test-split source notes on never-used COCO train2017 backgrounds.

- Foregrounds: only the BanglaTaka photos that the original generator put in the TEST split
  (recovered from data set/currency_yolo_data/test/images file names), so no training or
  validation source photo is reused.
- Backgrounds: COCO train2017 (the training composites used val2017 only).
- New random seed (900000+). Same compositing code as generate_synthetic_dataset.py.
- One composite per source photo, plus 10 % background-only negatives.
Then scores models/best.pt ONCE with Ultralytics val and saves the metrics.

Output: data set/currency_yolo_newtest/{images,labels}, results/new_test/detector_newtest.json
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
import generate_synthetic_dataset as gen  # noqa: E402

OLD_TEST = WS / "data set" / "currency_yolo_data" / "test" / "images"
DEST = WS / "data set" / "currency_yolo_newtest"
PAT = re.compile(r"^(\d+_taka)_(.+)_(\d+)\.jpg$")
SEED0 = 900000


def main() -> None:
    names = {v[1]: (k, v[0]) for k, v in gen.CLASSES.items()}  # class_name -> (folder, idx)
    sources = set()
    for f in OLD_TEST.iterdir():
        m = PAT.match(f.name)
        if m and m.group(1) in names:
            sources.add((m.group(1), m.group(2)))
    print("test-split source photos:", len(sources), flush=True)

    gen.BG_DIR = WS / "data set" / "coco2017" / "train2017"
    gen._load_bg_pool()
    for sub in ("images", "labels"):
        (DEST / sub).mkdir(parents=True, exist_ok=True)

    seed, n_pos, n_neg = SEED0, 0, 0
    for class_name, stem in sorted(sources):
        folder, idx = names[class_name]
        src = next((p for p in (gen.SOURCE_DIR / folder).iterdir() if p.stem == stem), None)
        if src is None:
            continue
        seed += 1
        random.seed(seed)
        np.random.seed(seed)
        comp, (xc, yc, bw, bh) = gen.make_composite(gen.load_source(src))
        out = f"{class_name}_{stem}_new"
        comp.save(DEST / "images" / f"{out}.jpg", quality=90)
        (DEST / "labels" / f"{out}.txt").write_text(f"{idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        n_pos += 1
        if random.random() < gen.NEGATIVE_RATIO:
            gen.make_negative().save(DEST / "images" / f"neg_{out}.jpg", quality=90)
            (DEST / "labels" / f"neg_{out}.txt").write_text("")
            n_neg += 1
    print("composites", n_pos, "negatives", n_neg, flush=True)

    order = sorted(gen.CLASSES.values())
    yaml = ROOT / "results" / "new_test" / "newtest.yaml"
    yaml.parent.mkdir(parents=True, exist_ok=True)
    yaml.write_text(f"path: {DEST.as_posix()}\ntrain: images\nval: images\ntest: images\nnc: {len(order)}\nnames:\n"
                    + "".join(f"  - {n}\n" for _, n in order), encoding="utf-8")

    from ultralytics import YOLO
    model = YOLO(str(ROOT / "models" / "best.pt"))
    m = model.val(data=str(yaml), split="test", plots=False, verbose=False)
    res = {"n_images": n_pos + n_neg, "n_notes": n_pos, "n_negatives": n_neg,
           "foregrounds": "BanglaTaka photos of the original test split only",
           "backgrounds": "COCO train2017 (never used in training)", "seed_base": SEED0,
           "precision": float(m.box.mp), "recall": float(m.box.mr),
           "map50": float(m.box.map50), "map50_95": float(m.box.map),
           "scored": "once, after generation; no model or threshold was changed"}
    (ROOT / "results" / "new_test" / "detector_newtest.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
