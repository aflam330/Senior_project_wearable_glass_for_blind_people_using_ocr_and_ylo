"""Validation-only follow-ups. Does not open the 24-phrase set or the 80-image set.

The cited pipeline is already slope_ths=0.2 (results/ocr_improve_summary.json).
Nothing in this file is allowed to replace that choice, because those two sets
were already read.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ocr_improve import (  # noqa: E402
    VAL_BN, VAL_EN, agg, glass_gray, join_text, prepare, read_kwargs, samples,
)
from eval_ocr_offline import levenshtein  # noqa: E402

OUT = ROOT / "results" / "ocr_improve_extra.json"


def niblack(gray, window=25, k=-0.2):
    gray_f = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f, -1, (window, window))
    sq = cv2.boxFilter(gray_f * gray_f, -1, (window, window))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    thresh = mean + k * std
    return np.where(gray_f > thresh, 255, 0).astype(np.uint8)


def perspective(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    pts = cv2.findNonZero(inv)
    if pts is None or len(pts) < 20:
        return bgr
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(np.float32)
    w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    h = int(max(np.linalg.norm(box[1] - box[2]), np.linalg.norm(box[3] - box[0])))
    if w < 8 or h < 8:
        return bgr
    if w < h:
        w, h = h, w
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    # order box points clockwise from top-left
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).ravel()
    ordered = np.array([box[np.argmin(s)], box[np.argmin(diff)], box[np.argmax(s)], box[np.argmax(diff)]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def gray_resize(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] < 1200:
        s = 1200 / gray.shape[1]
        gray = cv2.resize(gray, (1200, max(1, int(gray.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
    return gray


def clahe_only(bgr):
    gray = gray_resize(bgr)
    return cv2.createCLAHE(2.5, (8, 8)).apply(gray)


def image_for(name, bgr):
    if name == "niblack":
        return niblack(gray_resize(bgr))
    if name == "perspective":
        return glass_gray(perspective(bgr))
    if name == "gray_only":
        return gray_resize(bgr)
    if name == "clahe_only":
        return clahe_only(bgr)
    return prepare(name if name in ("baseline", "sauvola", "unsharp", "deskew", "text_crop") else "baseline", bgr)


def score(reader, data, name, kwargs):
    rows, t0 = [], time.perf_counter()
    for gt, bgr, lang in data:
        det = reader.readtext(image_for(name, bgr), **kwargs)
        hyp = join_text(det)[0]
        rows.append({"gt": gt, "hyp": hyp, "lang": lang, "cer": levenshtein(gt, hyp) / max(len(gt), 1)})
    secs = time.perf_counter() - t0
    summary = agg([{"lang": r["lang"], "cer": r["cer"], "wer": 0.0} for r in rows])
    summary["seconds"] = round(secs, 2)
    return summary, rows


def substitutions(gt, hyp):
    n, m = len(gt), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (gt[i - 1] != hyp[j - 1]))
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (gt[i - 1] != hyp[j - 1]):
            if gt[i - 1] != hyp[j - 1]:
                pairs.append((hyp[j - 1], gt[i - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs


def apply_map(text, mapping):
    return "".join(mapping.get(ch, ch) for ch in text)


def cross_seed(rows):
    """Fit single-character replacements on two seeds, score the third. Three folds."""
    folds = []
    for hold in range(3):
        fit = [r for r in rows if r["seed"] != hold]
        counts = Counter()
        for r in fit:
            counts.update(substitutions(r["gt"], r["hyp"]))
        mapping = {}
        for (src, dst), n in counts.items():
            if src == dst or src.isspace() or dst.isspace():
                continue
            rev = counts.get((dst, src), 0)
            if n >= 8 and rev * 4 < n and src not in mapping:
                mapping[src] = dst
        held = [r for r in rows if r["seed"] == hold]
        raw = float(np.mean([r["cer"] for r in held]))
        fixed = []
        for r in held:
            hyp = apply_map(r["hyp"], mapping)
            fixed.append(levenshtein(r["gt"], hyp) / max(len(r["gt"]), 1))
        folds.append({"hold_seed_index": hold, "n_rules": len(mapping), "rules": {a: b for a, b in mapping.items()},
                      "raw_cer": raw, "corrected_cer": float(np.mean(fixed)) if fixed else None})
    return folds


def main() -> None:
    font_note = "Nirmala via eval_ocr_improve.samples"
    val_blocks = [samples(VAL_BN, VAL_EN, 24, seed) for seed in (9000, 9100, 9200)]
    val = [row for block in val_blocks for row in block]
    glass_gray.ocr = __import__("modes.ocr_mode", fromlist=["OCRMode"]).OCRMode()
    import easyocr
    reader = easyocr.Reader(["bn", "en"], gpu=True, verbose=False)
    base_kw = read_kwargs("baseline")
    slope_kw = read_kwargs("slope_0_2")
    report = {"device": "cuda", "note": "validation only; does not replace slope_ths=0.2",
              "font": font_note, "methods": []}

    def add(name, summary):
        summary = dict(summary)
        summary["method"] = name
        report["methods"].append(summary)
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(name, round(summary["overall"]["cer"], 4), "bn", round(summary["bn"]["cer"], 4), flush=True)

    # per-seed for the cited comparison
    per_seed = []
    slope_rows = []
    for si, block in enumerate(val_blocks):
        bsum, _ = score(reader, block, "baseline", base_kw)
        ssum, rows = score(reader, block, "baseline", slope_kw)
        for r in rows:
            r["seed"] = si
        slope_rows.extend(rows)
        per_seed.append({"seed": [9000, 9100, 9200][si], "n": len(block),
                         "baseline_cer": bsum["overall"]["cer"], "baseline_bn": bsum["bn"]["cer"],
                         "slope_cer": ssum["overall"]["cer"], "slope_bn": ssum["bn"]["cer"]})
        print("seed", per_seed[-1], flush=True)
    report["per_seed"] = per_seed
    report["cross_seed_char_rules"] = cross_seed(slope_rows)

    for name, kw in (
        ("niblack", slope_kw),
        ("perspective", slope_kw),
        ("gray_only", slope_kw),
        ("clahe_only", slope_kw),
        ("slope_0_15", {**base_kw, "slope_ths": 0.15}),
        ("slope_0_30", {**base_kw, "slope_ths": 0.3}),
    ):
        print("scoring", name, flush=True)
        try:
            summary, _ = score(reader, val, name, kw)
            add(name, summary)
        except Exception as exc:
            add(name, {"overall": {"cer": None}, "bn": {"cer": None}, "en": {"cer": None}, "error": str(exc)})

    print("loading dbnet18", flush=True)
    try:
        db = easyocr.Reader(["bn", "en"], gpu=True, detect_network="dbnet18", verbose=False)
        for name, kw in (("dbnet18", base_kw), ("dbnet18_slope_0_2", slope_kw)):
            summary, _ = score(db, val, "baseline", kw)
            add(name, summary)
    except Exception as exc:
        report["dbnet18_error"] = traceback.format_exc()
        print("dbnet18 failed", exc, flush=True)

    engines = {}
    for mod in ("paddleocr", "pytesseract", "transformers", "kenlm", "bnlp"):
        try:
            __import__(mod)
            engines[mod] = "import_ok"
        except Exception as exc:
            engines[mod] = f"{type(exc).__name__}: {exc}"
    report["engine_imports"] = engines
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
