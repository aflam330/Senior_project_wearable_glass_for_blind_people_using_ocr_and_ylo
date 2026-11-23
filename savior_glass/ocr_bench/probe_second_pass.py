"""Which second text-detection pass recovers the VALIDATION words the glass reads nothing on? (design probe; the test
words are not read). For each failed validation word (bucket no_box or filtered in diagnose_real_val.json) and each
pass setting, the detector is run again and its boxes recognised, as _read_v2 would; the CER on these words is
reported, and the number of false readings on 40 text-free validation pictures of the rendered benchmark's COCO half
(backgrounds without the sign) shows whether a pass makes the glass invent text on scenes.

    python -m ocr_bench.probe_second_pass
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from . import bench as B
from .page_bench import OUT, _ocr_mode
from .real_words import frame_of
from .real_words_tune import fetch
from .train_real_words import split

PASSES = {
    "mag0.5": {"mag_ratio": 0.5},
    "mag0.5_low": {"mag_ratio": 0.5, "text_threshold": 0.6, "low_text": 0.25},
    "low": {"text_threshold": 0.5, "low_text": 0.25, "link_threshold": 0.3},
    "mag1.5_low": {"mag_ratio": 1.5, "text_threshold": 0.6, "low_text": 0.25},
    "native_res": {"_no_upscale": True},
    "native_res_low": {"_no_upscale": True, "text_threshold": 0.6, "low_text": 0.25},
}


def main():
    import torch
    import config
    from ocr_text_region import clean_text, dominant_block, order_lines
    torch.set_num_threads(8)
    diag = json.loads((OUT / "diagnose_real_val.json").read_text(encoding="utf-8"))["rows"]
    _, va, _ = split()
    items = fetch(va)
    failed = [it for it, r in zip(items, diag) if r["bucket"] != "read"]
    assert all(it["gt"] == r["gt"] for it, r in zip(items, diag))
    coco = sorted(B.COCO.glob("*.jpg"))[:len(sorted(B.COCO.glob("*.jpg"))) // 2][:40]   # validation half of COCO
    m = _ocr_mode()
    base = dict(config.OCR_V2_PARAMS)

    def second(frame, p):
        h, w = frame.shape[:2]
        if not p.get("_no_upscale") and w < 1200:
            frame = cv2.resize(frame, (1200, int(h * 1200 / w)), interpolation=cv2.INTER_CUBIC)
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        kw = {k: v for k, v in base.items() if k in ("text_threshold", "low_text", "link_threshold", "mag_ratio", "slope_ths")}
        kw.update({k: v for k, v in p.items() if not k.startswith("_")})
        det = m._reader.readtext(gray, detail=1, paragraph=False, width_ths=0.7, height_ths=0.7, decoder="greedy",
                                 canvas_size=config.OCR_CANVAS_SIZE, contrast_ths=0.1, adjust_contrast=0.7, **kw)
        block = dominant_block(det, config.OCR_V2_MIN_CONF)
        conf = float(np.mean([b["c"] for b in block])) if block else 0.0
        return clean_text(order_lines(block)), conf

    res = {"failed_val_words": {l: sum(it["lang"] == l for it in failed) for l in ("bn", "en")}}
    for name, p in PASSES.items():
        rows = [(it["lang"], it["gt"], *second(frame_of(it["img"]), p)) for it in failed]
        scenes = [second(cv2.resize(cv2.imread(str(c)), (640, 480)), p) for c in coco]
        out = {}
        for thr in (0.0, 0.2, 0.3, 0.4, 0.5):
            r = {}
            for l in ("bn", "en"):
                rs = [(g, h if c >= thr else "") for lang, g, h, c in rows if lang == l]
                r[l] = {"cer_on_failed": round(float(np.mean([B.cer(g, h) for g, h in rs])), 4),
                        "recovered": sum(bool(h) for _, h in rs)}
            r["scenes_with_invented_text"] = sum(bool(h) and c >= thr for h, c in scenes)
            out[f"conf>={thr}"] = r
        res[name] = out
        print(name, json.dumps(out), flush=True)
    (OUT / "probe_second_pass.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
