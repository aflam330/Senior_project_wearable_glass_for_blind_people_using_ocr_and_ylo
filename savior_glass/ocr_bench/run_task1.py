"""Task 1 runner: text-region detection before OCR. All choices on the validation set (3 seeds).

Stage A: one-at-a-time sweeps on val (seed 0 + 1 + 2): line ordering, confidence threshold, region crop,
detector (CRAFT vs DBNet18). Stage B: the best val configuration is written to results/ocr_bench/task1_best.json.
The test set is not read here (run_test.py reads it once per finished method).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_bench import bench as B, pipelines as P, text_region as T  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "ocr_bench"


def run(name, fn, cfg):
    r = B.score(fn, "val")
    B.save(r, OUT / f"val_t1_{name}.json", {"set": "val", "task": 1, "config": cfg})
    print(B.line(name, r), flush=True)
    return r["overall"]["cer"], cfg


def main() -> None:
    res = []
    res.append(run("sorted_lines", lambda im: P.easyocr_read(P.glass_preprocess(im), sort=True), {"kind": "full", "sort": True}))
    for mc in (0.1, 0.3, 0.5):
        res.append(run(f"conf{mc}", lambda im, mc=mc: P.easyocr_read(P.glass_preprocess(im), sort=True, min_conf=mc),
                       {"kind": "full", "sort": True, "min_conf": mc}))
    for mc in (0.1, 0.3):
        for sp in (False, True):
            cfg = {"kind": "region", "detector": "craft", "min_conf": mc, "second_pass": sp}
            res.append(run(f"region_craft_c{mc}_{'2pass' if sp else '1pass'}",
                           lambda im, mc=mc, sp=sp: T.read_region(im, "craft", min_conf=mc, second_pass=sp), cfg))
    best = min(res, key=lambda r: r[0])
    cfg = dict(best[1])
    if cfg.get("kind") == "region":
        c2 = dict(cfg, detector="dbnet18")
        res.append(run(f"region_dbnet18_c{c2['min_conf']}_{'2pass' if c2['second_pass'] else '1pass'}",
                       lambda im: T.read_region(im, "dbnet18", min_conf=c2["min_conf"], second_pass=c2["second_pass"]), c2))
    best = min(res, key=lambda r: r[0])
    (OUT / "task1_best.json").write_text(json.dumps({"val_cer": best[0], "config": best[1],
                                                     "all": [{"val_cer": c, "config": k} for c, k in res]}, indent=1), encoding="utf-8")
    print("BEST", best)


if __name__ == "__main__":
    main()
