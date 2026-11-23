"""Where does the pipeline lose real words? VALIDATION words only (150 + 150 of split()); the test words are not read.

For each validation word, framed as the glass sees it (real_words.frame_of):
  * boxes: how many text boxes the detector finds with the glass's settings
  * pipeline: the glass's reading (OCRMode._read_v2)
  * crop_only: the recogniser alone on the word crop (as the training curves measure it)
  * bucket: no_box (detector found nothing) / filtered (boxes found, nothing kept) / read
CER per bucket shows how much of the error comes from finding text and how much from reading it.

    python -m ocr_bench.diagnose_real_val
Output: results/ocr_bench/diagnose_real_val.json
"""
from __future__ import annotations

import json
from collections import Counter

import cv2
import numpy as np

from . import bench as B
from .page_bench import OUT, _ocr_mode
from .real_words import frame_of
from .real_words_tune import fetch
from .train_real_words import split


def main(threads=8):
    import torch
    import config
    from ocr_text_region import clean_text
    torch.set_num_threads(threads)
    _, va, _ = split()
    items = fetch(va)
    m = _ocr_mode()
    m.speak_line = None
    rows = []
    for it in items:
        fr = frame_of(it["img"])
        h, w = fr.shape[:2]
        big = cv2.resize(fr, (1200, int(h * 1200 / w)), interpolation=cv2.INTER_CUBIC)
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY))
        kw = {k: v for k, v in config.OCR_V2_PARAMS.items() if k in ("text_threshold", "low_text", "link_threshold", "mag_ratio", "slope_ths")}
        hl, fl = m._reader.detect(gray, width_ths=0.7, height_ths=0.7, canvas_size=config.OCR_CANVAS_SIZE, **kw)
        nbox = len(hl[0]) + len(fl[0]) if hl else 0
        hyp = clean_text(" ".join(m._read_v2(fr)))
        g = cv2.cvtColor(it["img"], cv2.COLOR_BGR2GRAY)
        crop = m._reader.recognize(g, horizontal_list=[[0, g.shape[1], 0, g.shape[0]]], free_list=[], detail=0,
                                   decoder="greedy")
        crop_txt = clean_text(" ".join(crop))
        upright = it["img"].shape[1] >= it["img"].shape[0]
        bucket = "no_box" if nbox == 0 else ("filtered" if not hyp else "read")
        rows.append({"lang": it["lang"], "gt": it["gt"], "pipeline": hyp, "crop_only": crop_txt, "boxes": nbox,
                     "bucket": bucket, "upright": upright, "word_h": int(it["img"].shape[0]), "word_w": int(it["img"].shape[1]),
                     "cer_pipeline": B.cer(it["gt"], hyp), "cer_crop": B.cer(it["gt"], crop_txt)})
    res = {}
    for lang in ("bn", "en"):
        rs = [r for r in rows if r["lang"] == lang]
        res[lang] = {"n": len(rs), "cer_pipeline": float(np.mean([r["cer_pipeline"] for r in rs])),
                     "cer_crop_only": float(np.mean([r["cer_crop"] for r in rs])),
                     "buckets": dict(Counter(r["bucket"] for r in rs)),
                     "cer_by_bucket": {b: float(np.mean([r["cer_pipeline"] for r in rs if r["bucket"] == b]))
                                       for b in ("no_box", "filtered", "read") if any(r["bucket"] == b for r in rs)},
                     "crop_cer_where_pipeline_read_nothing": float(np.mean([r["cer_crop"] for r in rs if not r["pipeline"]] or [0])),
                     "rotated": sum(not r["upright"] for r in rs),
                     "share_of_pipeline_error": {b: float(sum(r["cer_pipeline"] for r in rs if r["bucket"] == b) / max(sum(r["cer_pipeline"] for r in rs), 1e-9))
                                                 for b in ("no_box", "filtered", "read")}}
    (OUT / "diagnose_real_val.json").write_text(json.dumps({"summary": res, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
