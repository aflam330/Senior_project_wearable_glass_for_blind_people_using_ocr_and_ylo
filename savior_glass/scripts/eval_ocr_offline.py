"""Measure offline OCR: live glass preprocess + lexicon vs previous eval path.

Does not use currency images. n=80 synthetic BN+EN, same phrases as publish_boost.

Images are drawn with a fixed seed per sample, so every run sees the same 80 images.
Canvas sizes: `python scripts/eval_ocr_offline.py 2560 640` scores each size on those same images and
writes results/ocr_offline_repaired_canvas<size>.json plus results/ocr_canvas_comparison.json (paired).
With no argument it uses config.OCR_CANVAS_SIZE and writes results/ocr_offline_repaired.json as before.
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


def _render_text(text: str, font_path: str, wild: bool, seed: int | None = None) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    rng = np.random.default_rng(seed)
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


def _read(reader, gray, canvas_size: int | None = None) -> str:
    import config as _config
    if canvas_size is None:
        canvas_size = getattr(_config, "OCR_CANVAS_SIZE", 2560)
    try:
        det = reader.readtext(
            gray,
            detail=1,
            paragraph=False,
            decoder="beamsearch",
            beamWidth=5,
            width_ths=0.7,
            height_ths=0.7,
            canvas_size=canvas_size,
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
    # --select: a SELECTION set for choosing settings (canvas size). New phrases that are NOT in
    # assets/ocr_lexicon.txt (so lexicon repair cannot help) and different image seeds. Choose on its
    # RAW CER; the default 80-image set is then read once to report the chosen setting.
    select = "--select" in sys.argv
    if select:
        sys.argv.remove("--select")
        bn = ["সাবধান", "প্রবেশ নিষেধ", "শৌচাগার", "খোলা আছে", "পানি", "দুধ",
              "চাল পাঁচ কেজি", "লিফট", "সিঁড়ি", "জরুরি নির্গমন", "নামাজের ঘর", "অপেক্ষা কক্ষ"]
        en = ["Lift", "Stairs", "Toilet", "No Entry", "Open", "Closed",
              "Milk", "Rice 5 kg", "Fire Exit", "Waiting Room", "Push", "Pull"]
    font = _ocr_font()
    reader = easyocr.Reader(["bn", "en"], gpu=False, verbose=False)
    ocr = OCRMode()

    samples = []
    for lang, pool in (("bn", bn), ("en", en)):
        for i in range(40):
            text = pool[i % len(pool)]
            wild = i % 2 == 1
            img = _render_text(text, font, wild=wild, seed=1000 * (lang == "en") + i + (5000 if select else 0))
            samples.append((text, img, lang, wild))

    import time

    import config as _config
    sizes = [int(a) for a in sys.argv[1:]] or [None]
    per_size = {}
    for size in sizes:
        cs = size or getattr(_config, "OCR_CANVAS_SIZE", 2560)
        rows, t0 = [], time.perf_counter()
        for gt, img_rgb, lang, wild in samples:
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            gray = ocr._preprocess(bgr)
            raw = _read(reader, gray, cs)
            fixed = repair_ocr_text(raw)
            rows.append({"gt": gt, "raw": raw, "hyp": fixed, "lang": lang, "wild": wild,
                         "cer_raw": _cer(gt, raw), "cer": _cer(gt, fixed),
                         "wer_raw": _token_wer(gt, raw), "wer": _token_wer(gt, fixed)})
        secs = time.perf_counter() - t0

        def agg(subset, key_cer="cer", key_wer="wer"):
            return {"n": len(subset),
                    "cer": float(np.mean([r[key_cer] for r in subset])) if subset else 1.0,
                    "wer": float(np.mean([r[key_wer] for r in subset])) if subset else 1.0}

        out = {"n": len(rows), "engine": "easyocr-bn-en + glass preprocess + lexicon (offline)", "font": font,
               "tesseract": False, "canvas_size": cs, "seeded_images": True,
               "seconds_total_this_host": round(secs, 1), "host_note": "timing is from the machine that ran this, not the Pi",
               "raw": {"overall": agg(rows, "cer_raw", "wer_raw"),
                       "bn": agg([r for r in rows if r["lang"] == "bn"], "cer_raw", "wer_raw"),
                       "en": agg([r for r in rows if r["lang"] == "en"], "cer_raw", "wer_raw")},
               "repaired": {"overall": agg(rows), "bn": agg([r for r in rows if r["lang"] == "bn"]),
                            "en": agg([r for r in rows if r["lang"] == "en"])},
               "samples_head": rows[:12], "per_sample_cer": [r["cer"] for r in rows]}
        name = "ocr_offline_repaired.json" if size is None else f"ocr_offline_repaired_canvas{cs}.json"
        if select:
            name = f"ocr_select_canvas{cs}.json"
            out["set"] = "selection (phrases not in the lexicon, seeds +5000)"
        (RESULTS / name).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        per_size[cs] = out
        print("wrote", RESULTS / name, "canvas", cs)
        print("RAW     ", out["raw"])
        print("REPAIRED", out["repaired"])
    if len(per_size) == 2:
        from scipy.stats import wilcoxon
        (a, ra), (b, rb) = per_size.items()
        d = np.array(ra["per_sample_cer"]) - np.array(rb["per_sample_cer"])
        cmp = {"canvas_a": a, "canvas_b": b, "n": len(d), "same_images": True,
               "repaired_cer": {str(a): ra["repaired"]["overall"]["cer"], str(b): rb["repaired"]["overall"]["cer"]},
               "repaired_wer": {str(a): ra["repaired"]["overall"]["wer"], str(b): rb["repaired"]["overall"]["wer"]},
               "bn_cer": {str(a): ra["repaired"]["bn"]["cer"], str(b): rb["repaired"]["bn"]["cer"]},
               "en_cer": {str(a): ra["repaired"]["en"]["cer"], str(b): rb["repaired"]["en"]["cer"]},
               "samples_a_better": int((d < 0).sum()), "samples_b_better": int((d > 0).sum()), "samples_equal": int((d == 0).sum()),
               "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}
        (RESULTS / ("ocr_select_comparison.json" if select else "ocr_canvas_comparison.json")).write_text(json.dumps(cmp, indent=1), encoding="utf-8")
        print(json.dumps(cmp, indent=1))


if __name__ == "__main__":
    main()
