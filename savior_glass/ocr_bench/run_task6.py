"""Task 6: multi-scale OCR. The preprocessed image is read at scales 1.0, 1.5 and 2.0 (same region rule,
Task 4 parameters). Ensembles: the most confident reading (character-weighted box confidence), or a vote:
the reading with the smallest total edit distance to the other two (medoid). Chosen on val; one pass per
scale is cached so every ensemble is scored on the same readings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_bench import bench as B, text_region as T  # noqa: E402
from ocr_bench.run_task4 import T1, T2, prep_fn  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "ocr_bench"
T4 = json.loads((OUT / "task4_best.json").read_text(encoding="utf-8"))["config"]
SCALES = (1.0, 1.5, 2.0)


def read_all_scales(im):
    base = prep_fn(T2["best"])(im)
    kw = {k: v for k, v in T4.items() if k != "min_conf"}
    outs = []
    for s in SCALES:
        img = base if s == 1.0 else cv2.resize(base, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        outs.append(T.read_region(None, T1["detector"], min_conf=T4["min_conf"], second_pass=T1["second_pass"],
                                  prep=lambda _x, img=img: img, return_conf=True, **kw))
    return outs


def medoid(texts):
    d = [sum(B.lev(a, b) for b in texts) for a in texts]
    return texts[int(np.argmin(d))]


def main() -> None:
    cache = {}

    def cached(im):
        k = im.tobytes()
        if k not in cache:
            cache[k] = read_all_scales(im)
        return cache[k]

    methods = {f"scale{s}": (lambda im, i=i: cached(im)[i][0]) for i, s in enumerate(SCALES)}
    methods["most_confident"] = lambda im: max(cached(im), key=lambda r: r[1])[0]
    methods["vote_medoid"] = lambda im: medoid([r[0] for r in cached(im)])
    res = {}
    for name, fn in methods.items():
        r = B.score(fn, "val")
        r["latency_ms_median"] = None  # per-method latency is not separable when readings are cached
        B.save(r, OUT / f"val_t6_{name}.json", {"set": "val", "task": 6, "method": name, "scales": SCALES})
        res[name] = r["overall"]["cer"]
        print(f"{name:16s} CER {100*r['overall']['cer']:.2f} BN {100*r['bn']['cer']:.2f} EN {100*r['en']['cer']:.2f} "
              f"scene {100*r['by_cond']['scene']['all']['cer']:.2f}", flush=True)
    best = min(res, key=res.get)
    (OUT / "task6_best.json").write_text(json.dumps({"best": best, "val_cer": res[best], "all": res, "single_scale_is_task4": "scale1.0"},
                                                    indent=1), encoding="utf-8")
    print("BEST", best, res[best])


if __name__ == "__main__":
    main()
