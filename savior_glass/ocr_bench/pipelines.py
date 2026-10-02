"""OCR pipelines compared in the benchmark. Each returns read(bgr) -> text."""
from __future__ import annotations

import os
import sys
from functools import lru_cache

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DEVICE_GPU = os.environ.get("OCR_BENCH_GPU", "1") == "1"


@lru_cache(maxsize=4)
def reader(detector: str = "craft", gpu: bool = DEVICE_GPU):
    import easyocr
    kw = {} if detector == "craft" else {"detect_network": detector}
    return easyocr.Reader(["bn", "en"], gpu=gpu, verbose=False, **kw)


def glass_preprocess(bgr):
    """The current glass preprocessing (modes/ocr_mode.py OCRMode._preprocess)."""
    from modes.ocr_mode import OCRMode
    return OCRMode._preprocess(None, bgr)


def join_lines(det, sort=True) -> str:
    """Join EasyOCR boxes. sort=True orders boxes into lines (top to bottom, then left to right)."""
    items = [d for d in det if not isinstance(d, str) and len(d) > 1]
    if not items:
        return ""
    if not sort:
        return " ".join(str(d[1]) for d in items).strip()
    boxes = []
    for b, t, c in items:
        b = np.asarray(b, np.float32)
        boxes.append((b[:, 1].mean(), b[:, 0].min(), b[:, 1].max() - b[:, 1].min(), t, c))
    boxes.sort(key=lambda r: r[0])
    lines, cur = [], [boxes[0]]
    for r in boxes[1:]:
        if abs(r[0] - np.mean([x[0] for x in cur])) < 0.5 * max(np.median([x[2] for x in cur]), 1):
            cur.append(r)
        else:
            lines.append(cur)
            cur = [r]
    lines.append(cur)
    return " ".join(" ".join(x[3] for x in sorted(l, key=lambda r: r[1])) for l in lines).strip()


def easyocr_read(img, detector="craft", sort=True, min_conf=0.0, **kw):
    params = dict(detail=1, paragraph=False, decoder="beamsearch", beamWidth=5, width_ths=0.7, height_ths=0.7, canvas_size=2560)
    params.update(kw)
    det = reader(detector).readtext(img, **params)
    det = [d for d in det if len(d) < 3 or d[2] >= min_conf]
    return join_lines(det, sort)


def baseline(bgr, repair=False):
    """Current glass: preprocess, EasyOCR beamsearch, canvas 2560, boxes in EasyOCR's order; repair optional."""
    txt = easyocr_read(glass_preprocess(bgr), sort=False)
    if repair:
        from ocr_repair import repair_ocr_text
        txt = repair_ocr_text(txt)
    return txt


def glass_app(bgr):
    """The OCR mode of the app as shipped before this work (modes/ocr_mode.py process_frame, EasyOCR path):
    glass preprocess, beamsearch, reading-order sort, confidence >= config.OCR_CONFIDENCE, >= OCR_MIN_CHARS,
    _looks_like_text filter, then lexicon repair. (Tesseract / Ekush fallbacks only run when EasyOCR finds nothing.)"""
    import config
    from modes.ocr_mode import _looks_like_text, _reading_order_key
    from ocr_repair import repair_ocr_text
    det = reader("craft").readtext(glass_preprocess(bgr), detail=1, paragraph=False, decoder="beamsearch", beamWidth=5,
                                   width_ths=0.7, height_ths=0.7, canvas_size=2560, slope_ths=0.1)
    texts = []
    for _b, t, c in sorted(det, key=_reading_order_key):
        t = t.strip()
        if c < config.OCR_CONFIDENCE or len(t) < config.OCR_MIN_CHARS or not _looks_like_text(t):
            continue
        texts.append(t)
    return repair_ocr_text(" ".join(texts)) if texts else ""
