"""Task 2 runner: preprocessing variants on top of the Task 1 choice (region, CRAFT, conf 0.1, one pass).

Each variant replaces the glass preprocessing. Val set, 3 seeds. The best single variant is then combined
with deskew / unsharp if those helped alone (one greedy step), still on val.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_bench import bench as B, preprocess as PP, text_region as T  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "ocr_bench"
T1 = json.loads((OUT / "task1_best.json").read_text(encoding="utf-8"))["best"]["config"]


def reader_with(prep):
    return lambda im: T.read_region(im, T1["detector"], min_conf=T1["min_conf"], second_pass=T1["second_pass"], prep=prep)


def main() -> None:
    res = {}
    for name, fn in PP.VARIANTS.items():
        r = B.score(reader_with(fn), "val")
        B.save(r, OUT / f"val_t2_{name}.json", {"set": "val", "task": 2, "prep": name, "task1": T1})
        res[name] = r["overall"]["cer"]
        print(B.line(name, r), flush=True)
    base = min(res, key=res.get)
    if res["deskew"] < res["gray"] and base != "deskew":  # deskew helped alone: deskew first, then the best variant
        name = f"deskew+{base}"
        r = B.score(reader_with(lambda im, b=base: PP.VARIANTS[b](_deskew_bgr(im))), "val")
        B.save(r, OUT / f"val_t2_{name}.json", {"set": "val", "task": 2, "prep": name, "task1": T1})
        res[name] = r["overall"]["cer"]
        print(B.line(name, r), flush=True)
    best = min(res, key=res.get)
    (OUT / "task2_best.json").write_text(json.dumps({"best": best, "val_cer": res[best], "all": res, "task1": T1}, indent=1), encoding="utf-8")
    print("BEST", best, res[best])


def _deskew_bgr(bgr):
    import cv2
    from deskew import determine_skew
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ang = determine_skew(g)
    if ang is None or abs(ang) > 15:
        return bgr
    M = cv2.getRotationMatrix2D((bgr.shape[1] / 2, bgr.shape[0] / 2), ang, 1.0)
    return cv2.warpAffine(bgr, M, (bgr.shape[1], bgr.shape[0]), borderMode=cv2.BORDER_REPLICATE)


if __name__ == "__main__":
    main()
