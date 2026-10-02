"""Task 4: EasyOCR parameter search, validation only.

Pipeline: Task 1 region choice + Task 2 preprocessing. 30 random configurations are scored on val seed 0;
the 4 best on seed 0 (and the current defaults) are re-scored on all three val seeds; the best of those wins.
Detector: CRAFT only (DBNet18 needs a C++ extension that cannot be compiled here). Recognizer: EasyOCR has one
Bangla recognition model (the CRNN in bengali.pth); there is no Bangla 'transformer' recognizer to try.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_bench import bench as B, preprocess as PP, text_region as T  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "ocr_bench"
T2 = json.loads((OUT / "task2_best.json").read_text(encoding="utf-8"))
T1 = T2["task1"]
GRID = {"text_threshold": [0.5, 0.6, 0.7, 0.8], "low_text": [0.3, 0.4, 0.5], "link_threshold": [0.3, 0.4, 0.5],
        "mag_ratio": [1.0, 1.5, 2.0], "contrast_ths": [0.1, 0.3, 0.5], "adjust_contrast": [0.3, 0.5, 0.7],
        "decoder": ["greedy", "beamsearch"], "slope_ths": [0.1, 0.2, 0.4], "canvas_size": [1280, 2560], "min_conf": [0.05, 0.1, 0.2]}
DEFAULT = {"text_threshold": 0.7, "low_text": 0.4, "link_threshold": 0.4, "mag_ratio": 1.0, "contrast_ths": 0.1, "adjust_contrast": 0.5,
           "decoder": "beamsearch", "slope_ths": 0.1, "canvas_size": 2560, "min_conf": T1["min_conf"]}


def prep_fn(name):
    if name.startswith("deskew+"):
        from ocr_bench.run_task2 import _deskew_bgr
        base = PP.VARIANTS[name.split("+", 1)[1]]
        return lambda im: base(_deskew_bgr(im))
    return PP.VARIANTS[name]


def reader_for(cfg):
    prep = prep_fn(T2["best"])
    kw = {k: v for k, v in cfg.items() if k != "min_conf"}
    return lambda im: T.read_region(im, T1["detector"], min_conf=cfg["min_conf"], second_pass=T1["second_pass"], prep=prep, **kw)


def main() -> None:
    rng = random.Random(42)
    cands = [dict(DEFAULT)] + [{k: rng.choice(v) for k, v in GRID.items()} for _ in range(30)]
    stage1 = []
    for i, cfg in enumerate(cands):
        r = B.score(reader_for(cfg), "val", seeds=(0,))
        stage1.append((r["overall"]["cer"], i, cfg))
        print(f"[{i:02d}] seed0 CER {100*r['overall']['cer']:.2f} {cfg}", flush=True)
    top = sorted(stage1, key=lambda x: x[0])[:4]
    if 0 not in [i for _, i, _ in top]:
        top.append(stage1[0])  # always confirm the defaults on all seeds too
    stage2 = []
    for _, i, cfg in top:
        r = B.score(reader_for(cfg), "val")
        B.save(r, OUT / f"val_t4_cfg{i:02d}.json", {"set": "val", "task": 4, "config": cfg, "prep": T2["best"], "task1": T1})
        stage2.append((r["overall"]["cer"], i, cfg))
        print(B.line(f"cfg{i:02d} (3 seeds)", r), flush=True)
    best = min(stage2, key=lambda x: x[0])
    (OUT / "task4_best.json").write_text(json.dumps({"best_cfg_index": best[1], "val_cer": best[0], "config": best[2],
                                                     "default_val_cer": next(c for c, i, _ in stage2 if i == 0),
                                                     "stage1": [{"seed0_cer": c, "i": i, "config": k} for c, i, k in stage1],
                                                     "prep": T2["best"], "task1": T1}, indent=1), encoding="utf-8")
    print("BEST", best)


if __name__ == "__main__":
    main()
