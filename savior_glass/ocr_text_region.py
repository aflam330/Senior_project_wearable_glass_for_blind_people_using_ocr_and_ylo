"""Text-region selection and cleanup for OCR mode, chosen on the OCR benchmark's validation set.

paper_evidence/OCR_TEXT_REGION.md and OCR_POSTPROCESSING.md. Pure Python + NumPy, no extra packages.

dominant_block  keep the text the user is facing: start from the most confident tall box, add boxes of a
                similar height that touch it; drops stray text from the background of a scene
order_lines     join boxes line by line (top to bottom, then left to right)
clean_text      NFC, invisible characters, stray symbols at word edges, digits in the script of their neighbours
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np

_BN = re.compile(r"[ঀ-৿]")
_BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
_EN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍﻿­"), None)
_EDGE = re.compile(r"^[\|\[\]\{\}\(\)<>~`'\"“”‘’_^*#@$%&;:,.!?\\/=+-]+|[\|\[\]\{\}\(\)<>~`'\"“”‘’_^*#@$%&;:,!?\\/=+-]+$")


def _boxes(det):
    out = []
    for item in det:
        if len(item) < 3:
            continue
        b, t, c = item
        b = np.asarray(b, np.float32)
        out.append({"x0": float(b[:, 0].min()), "x1": float(b[:, 0].max()), "y0": float(b[:, 1].min()),
                    "y1": float(b[:, 1].max()), "h": float(b[:, 1].max() - b[:, 1].min()), "t": t, "c": float(c)})
    return out


def dominant_block(det, min_conf=0.2, h_ratio=0.6, gap=1.5):
    boxes = [b for b in _boxes(det) if b["c"] >= min_conf]
    if not boxes:
        return []
    seed = max(boxes, key=lambda b: b["h"] * b["c"])
    block, rest = [seed], [b for b in boxes if b is not seed]
    grown = True
    while grown:
        grown = False
        for b in list(rest):
            hs = float(np.median([x["h"] for x in block]))
            if not (h_ratio <= b["h"] / max(hs, 1.0) <= 1 / h_ratio):
                continue
            if any(b["x0"] - x["x1"] < gap * hs and x["x0"] - b["x1"] < gap * hs and
                   b["y0"] - x["y1"] < gap * hs and x["y0"] - b["y1"] < gap * hs for x in block):
                block.append(b)
                rest.remove(b)
                grown = True
    return block


def order_lines(block) -> str:
    if not block:
        return ""
    rows = sorted(((b["y0"] + b["y1"]) / 2, b["x0"], b["h"], b["t"]) for b in block)
    lines, cur = [], [rows[0]]
    for r in rows[1:]:
        if abs(r[0] - np.mean([x[0] for x in cur])) < 0.5 * max(np.median([x[2] for x in cur]), 1):
            cur.append(r)
        else:
            lines.append(cur)
            cur = [r]
    lines.append(cur)
    return " ".join(" ".join(x[3] for x in sorted(l, key=lambda r: r[1])) for l in lines).strip()


def clean_text(text: str) -> str:
    t = unicodedata.normalize("NFC", text).translate(_INVISIBLE)
    toks = [w for w in (_EDGE.sub("", w) for w in t.split()) if w]
    out = []
    for i, w in enumerate(toks):
        if all(c.isdigit() for c in w):
            nb = [x for x in (toks[i - 1] if i else "", toks[i + 1] if i + 1 < len(toks) else "") if x and not x.isdigit()]
            if nb and all(_BN.search(x) for x in nb):
                w = w.translate(_BN_DIGITS)
            elif nb and not any(_BN.search(x) for x in nb):
                w = w.translate(_EN_DIGITS)
        out.append(w)
    return " ".join(out)
