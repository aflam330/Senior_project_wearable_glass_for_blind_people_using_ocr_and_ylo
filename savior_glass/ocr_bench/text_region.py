"""Task 1: find the main text region first, then read only that region.

Steps (each part switchable, chosen on the validation set only):
  1. detect text boxes on the whole frame (CRAFT or DBNet18, EasyOCR's detectors)
  2. drop boxes below a recognition confidence
  3. keep the dominant text block: boxes whose height is close to the tallest confident line,
     and which sit next to each other (the sign or label the user is facing)
  4. crop that block with a margin, upscale to 1200 px wide, read again
  5. order boxes into lines (top to bottom, left to right)
"""
from __future__ import annotations

import cv2
import numpy as np

from . import pipelines as P


def _boxes(det):
    out = []
    for b, t, c in det:
        b = np.asarray(b, np.float32)
        out.append({"x0": b[:, 0].min(), "x1": b[:, 0].max(), "y0": b[:, 1].min(), "y1": b[:, 1].max(),
                    "h": b[:, 1].max() - b[:, 1].min(), "t": t, "c": float(c)})
    return out


def dominant_block(boxes, h_ratio=0.6, gap=1.5):
    """Boxes of the most prominent text: start from the most confident tall box, grow by adjacency."""
    if not boxes:
        return []
    seed = max(boxes, key=lambda b: b["h"] * b["c"])
    block, rest = [seed], [b for b in boxes if b is not seed]
    grown = True
    while grown:
        grown = False
        for b in list(rest):
            hs = np.median([x["h"] for x in block])
            if not (h_ratio <= b["h"] / max(hs, 1) <= 1 / h_ratio):
                continue
            near = any(b["x0"] - x["x1"] < gap * hs and x["x0"] - b["x1"] < gap * hs and
                       b["y0"] - x["y1"] < gap * hs and x["y0"] - b["y1"] < gap * hs for x in block)
            if near:
                block.append(b)
                rest.remove(b)
                grown = True
    return block


def read_region(bgr, detector="craft", min_conf=0.3, h_ratio=0.6, gap=1.5, margin=0.25, second_pass=True,
                preprocess=True, prep=None, return_conf=False, **kw):
    """prep: optional callable bgr -> image for EasyOCR (Task 2); default is the glass preprocessing."""
    if prep is not None:
        img = prep(bgr)
    else:
        img = P.glass_preprocess(bgr) if preprocess else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    params = dict(detail=1, paragraph=False, decoder="beamsearch", beamWidth=5, width_ths=0.7, height_ths=0.7, canvas_size=2560)
    params.update(kw)
    det = P.reader(detector).readtext(img, **params)
    boxes = [b for b in _boxes(det) if b["c"] >= min_conf]
    block = dominant_block(boxes, h_ratio, gap)
    if not block:
        return ("", 0.0) if return_conf else ""
    if not second_pass:
        txt = P.join_lines([([[b["x0"], b["y0"]], [b["x1"], b["y0"]], [b["x1"], b["y1"]], [b["x0"], b["y1"]]], b["t"], b["c"])
                             for b in block])
        if return_conf:
            n = sum(len(b["t"]) for b in block)
            return txt, float(sum(b["c"] * len(b["t"]) for b in block) / max(n, 1))
        return txt
    hs = np.median([b["h"] for b in block])
    x0 = int(max(0, min(b["x0"] for b in block) - margin * 2 * hs))
    x1 = int(min(img.shape[1], max(b["x1"] for b in block) + margin * 2 * hs))
    y0 = int(max(0, min(b["y0"] for b in block) - margin * hs))
    y1 = int(min(img.shape[0], max(b["y1"] for b in block) + margin * hs))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return ""
    s = 1200 / crop.shape[1]
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    det2 = P.reader(detector).readtext(crop, **params)
    det2 = [d for d in det2 if d[2] >= min_conf] or det2
    return P.join_lines(det2)
