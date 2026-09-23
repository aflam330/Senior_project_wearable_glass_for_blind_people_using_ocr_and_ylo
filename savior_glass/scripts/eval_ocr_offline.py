"""Measure offline OCR: live glass preprocess + lexicon vs previous eval path.

Does not use currency images. n=80 synthetic BN+EN, same phrases as publish_boost.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modes.ocr_mode import OCRMode  # noqa: E402
from ocr_repair import repair_ocr_text  # noqa: E402


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _token_wer(gt: str, hyp: str) -> float:
    a, b = gt.split(), hyp.split()
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(a)


def _ocr_font() -> str:
    for p in (
        Path(r"C:\Windows\Fonts\Nirmala.ttf"),
        Path(r"C:\Windows\Fonts\vrinda.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ):
        if p.is_file():
            return str(p)
    return ""


def _render_text(text: str, font_path: str, wild: bool) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    rng = np.random.default_rng()
    w, h = 640, 160
    bg = int(rng.integers(200, 255)) if wild else 255
    img = Image.new("RGB", (w, h), (bg, bg, bg))
    draw = ImageDraw.Draw(img)
    size = int(rng.integers(28, 44) if wild else 36)
    try:
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    draw.text((24, 48), text, fill=(int(rng.integers(0, 40)),) * 3, font=font)
    if wild:
        img = img.rotate(float(rng.uniform(-6, 6)), expand=0, fillcolor=(bg, bg, bg))
        img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.0, 1.1))))
    arr = np.array(img)
    if wild and rng.random() < 0.5:
        arr = cv2.resize(arr, (0, 0), fx=0.55, fy=0.55)
        arr = cv2.resize(arr, (w, h))
    return arr

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def _cer(gt: str, hyp: str) -> float:
    return levenshtein(gt, hyp) / max(len(gt), 1)


def _read(reader, gray) -> str:
    try:
        det = reader.readtext(
            gray,
            detail=1,
            paragraph=False,
            decoder="beamsearch",
            beamWidth=5,
            width_ths=0.7,
            height_ths=0.7,
        )
    except TypeError:
        det = reader.readtext(gray, detail=1, paragraph=False, width_ths=0.7, height_ths=0.7)
    pieces = []
    for t in det:
        if isinstance(t, str):
            pieces.append(t)
        elif t and len(t) > 1:
            pieces.append(str(t[1]))
    return " ".join(pieces).strip()


def main() -> None:
    import easyocr

    bn = [
        "বাংলাদেশ ব্যাংক", "দশ টাকা", "বিশ টাকা", "এই পথ বন্ধ",
        "স্টেশন ১২", "হাসপাতাল", "ঔষধের দোকান", "বাস স্টপ",
        "ডানদিকে যান", "বাঁদিকে যান", "মূল ফটক", "টিকিট কাউন্টার",
    ]
    en = [
        "EXIT 12", "Bus Stop", "Hospital Gate", "Ticket Counter",
        "Main Entrance", "Pharmacy", "Turn Right", "Turn Left",
        "Platform 3", "Danger Keep Out", "Glycemic", "Digestive Biscuit",
    ]
    font = _ocr_font()
    reader = easyocr.Reader(["bn", "en"], gpu=False, verbose=False)
    ocr = OCRMode()

    samples = []
    for lang, pool in (("bn", bn), ("en", en)):
        for i in range(40):
            text = pool[i % len(pool)]
            wild = i % 2 == 1
            img = _render_text(text, font, wild=wild)
            samples.append((text, img, lang, wild))

    rows = []
    for gt, img_rgb, lang, wild in samples:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = ocr._preprocess(bgr)
        raw = _read(reader, gray)
        fixed = repair_ocr_text(raw)
        rows.append({
            "gt": gt,
            "raw": raw,
            "hyp": fixed,
            "lang": lang,
            "wild": wild,
            "cer_raw": _cer(gt, raw),
            "cer": _cer(gt, fixed),
            "wer_raw": _token_wer(gt, raw),
            "wer": _token_wer(gt, fixed),
        })

    def agg(subset, key_cer="cer", key_wer="wer"):
        return {
            "n": len(subset),
            "cer": float(np.mean([r[key_cer] for r in subset])) if subset else 1.0,
            "wer": float(np.mean([r[key_wer] for r in subset])) if subset else 1.0,
        }

    out = {
        "n": len(rows),
        "engine": "easyocr-bn-en + glass preprocess + lexicon (offline)",
        "font": font,
        "tesseract": False,
        "raw": {
            "overall": agg(rows, "cer_raw", "wer_raw"),
            "bn": agg([r for r in rows if r["lang"] == "bn"], "cer_raw", "wer_raw"),
            "en": agg([r for r in rows if r["lang"] == "en"], "cer_raw", "wer_raw"),
        },
        "repaired": {
            "overall": agg(rows),
            "bn": agg([r for r in rows if r["lang"] == "bn"]),
            "en": agg([r for r in rows if r["lang"] == "en"]),
        },
        "samples_head": rows[:12],
    }
    path = RESULTS / "ocr_offline_repaired.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", path)
    print("RAW     ", out["raw"])
    print("REPAIRED", out["repaired"])


if __name__ == "__main__":
    main()
