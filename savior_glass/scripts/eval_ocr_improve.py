"""Choose an OCR pipeline on a fresh validation phrase list, then read the held-out sets once.

Validation phrases are not in the 80-sample list and not in the 24-phrase list.
Three image seeds (9000, 9100, 9200). The kept pipeline is the lowest mean raw
character error on that validation pool. Ties keep the faster pipeline.

Held-out reads, after the choice is fixed:
  select  — the 24 phrases that are not in the lexicon (the 13.3% set)
  report  — the original 80 images whose phrases are in the lexicon

Lexicon repair is reported and is not used to choose the pipeline.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ocr_offline import _cer, _ocr_font, _render_text, _token_wer  # noqa: E402
from modes.ocr_mode import OCRMode  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "ocr_improve_val.json"

VAL_BN = [
    "আসন নম্বর", "দরজা বন্ধ", "জানালা খুলুন", "বিদ্যুৎ চলে গেছে", "ধোঁয়া দেখা", "মেঝে ভেজা",
    "ছাদে উঠুন", "রাস্তা মেরামত", "নদীর ঘাট", "স্কুল গেট", "বাজার বন্ধ", "সেতু পার",
]
VAL_EN = [
    "Seat 4", "Door Closed", "Window Open", "Power Off", "Smoke Alarm", "Wet Floor",
    "Roof Access", "Road Work", "River Gate", "School Gate", "Market Closed", "Bridge Ahead",
]
SELECT_BN = ["সাবধান", "প্রবেশ নিষেধ", "শৌচাগার", "খোলা আছে", "পানি", "দুধ",
             "চাল পাঁচ কেজি", "লিফট", "সিঁড়ি", "জরুরি নির্গমন", "নামাজের ঘর", "অপেক্ষা কক্ষ"]
SELECT_EN = ["Lift", "Stairs", "Toilet", "No Entry", "Open", "Closed",
              "Milk", "Rice 5 kg", "Fire Exit", "Waiting Room", "Push", "Pull"]
REPORT_BN = [
    "বাংলাদেশ ব্যাংক", "দশ টাকা", "বিশ টাকা", "এই পথ বন্ধ",
    "স্টেশন ১২", "হাসপাতাল", "ঔষধের দোকান", "বাস স্টপ",
    "ডানদিকে যান", "বাঁদিকে যান", "মূল ফটক", "টিকিট কাউন্টার",
]
REPORT_EN = [
    "EXIT 12", "Bus Stop", "Hospital Gate", "Ticket Counter",
    "Main Entrance", "Pharmacy", "Turn Right", "Turn Left",
    "Platform 3", "Danger Keep Out", "Glycemic", "Digestive Biscuit",
]


def samples(bn, en, per_lang: int, seed_base: int):
    font = _ocr_font()
    rows = []
    for lang, pool in (("bn", bn), ("en", en)):
        for i in range(per_lang):
            text = pool[i % len(pool)]
            wild = i % 2 == 1
            img = _render_text(text, font, wild=wild, seed=seed_base + 1000 * (lang == "en") + i)
            rows.append((text, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), lang))
    return rows


def agg(rows):
    def one(subset):
        if not subset:
            return {"n": 0, "cer": None, "wer": None}
        return {"n": len(subset),
                "cer": float(np.mean([r["cer"] for r in subset])),
                "wer": float(np.mean([r["wer"] for r in subset]))}
    return {"overall": one(rows), "bn": one([r for r in rows if r["lang"] == "bn"]),
            "en": one([r for r in rows if r["lang"] == "en"])}


def join_text(det):
    pieces, conf = [], []
    for item in det:
        if isinstance(item, str):
            pieces.append(item)
        elif item and len(item) > 1:
            pieces.append(str(item[1]))
            if len(item) > 2 and isinstance(item[2], (int, float)):
                conf.append(float(item[2]))
    return " ".join(pieces).strip(), (float(np.mean(conf)) if conf else 0.0)


def glass_gray(bgr):
    ocr = glass_gray.ocr
    return ocr._preprocess(bgr)


def text_crop(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(inv)
    if pts is None:
        return bgr
    x, y, w, h = cv2.boundingRect(pts)
    pad = 8
    y0, y1 = max(0, y - pad), min(bgr.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(bgr.shape[1], x + w + pad)
    crop = bgr[y0:y1, x0:x1]
    return crop if crop.size else bgr


def sauvola(gray, window=25, k=0.34):
    gray_f = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f, -1, (window, window))
    sq = cv2.boxFilter(gray_f * gray_f, -1, (window, window))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    thresh = mean * (1 + k * (std / 128.0 - 1))
    return np.where(gray_f > thresh, 255, 0).astype(np.uint8)


def deskew(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    pts = cv2.findNonZero(inv)
    if pts is None or len(pts) < 20:
        return bgr
    angle = cv2.minAreaRect(pts)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.4 or abs(angle) > 12:
        return bgr
    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def unsharp(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    return cv2.addWeighted(gray, 1.6, blur, -0.6, 0)


def prepare(name, bgr):
    if name == "text_crop":
        bgr = text_crop(bgr)
    if name == "deskew":
        bgr = deskew(bgr)
    if name == "sauvola":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] < 1200:
            s = 1200 / gray.shape[1]
            gray = cv2.resize(gray, (1200, int(gray.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
        return sauvola(gray)
    gray = glass_gray(bgr)
    if name == "unsharp":
        return unsharp(gray)
    return gray


def read_kwargs(name):
    kw = {"detail": 1, "paragraph": False, "decoder": "beamsearch", "beamWidth": 5,
          "width_ths": 0.7, "height_ths": 0.7, "canvas_size": 2560, "mag_ratio": 1.0}
    if name == "canvas_640":
        kw["canvas_size"] = 640
    elif name == "greedy":
        kw["decoder"] = "greedy"
        kw.pop("beamWidth", None)
    elif name == "mag_1_5":
        kw["mag_ratio"] = 1.5
    elif name == "mag_2_0":
        kw["mag_ratio"] = 2.0
    elif name == "low_threshold":
        kw.update(text_threshold=0.5, low_text=0.3, link_threshold=0.3)
    elif name == "slope_0_2":
        kw["slope_ths"] = 0.2
    elif name == "contrast":
        kw.update(contrast_ths=0.05, adjust_contrast=0.7)
    return kw


METHODS = (
    "baseline", "canvas_640", "text_crop", "sauvola", "unsharp", "deskew",
    "greedy", "mag_1_5", "mag_2_0", "low_threshold", "slope_0_2", "contrast", "multiscale",
)


def recognize(reader, name, bgr):
    if name == "multiscale":
        kw = read_kwargs("baseline")
        best_text, best_conf = "", -1.0
        for scale in (1.0, 1.5):
            img = bgr if scale == 1.0 else cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            det = reader.readtext(prepare("baseline", img), **kw)
            text, conf = join_text(det)
            if conf > best_conf:
                best_text, best_conf = text, conf
        return best_text
    det = reader.readtext(prepare(name, bgr), **read_kwargs(name))
    return join_text(det)[0]


def score_method(reader, name, data):
    rows, t0 = [], time.perf_counter()
    for gt, bgr, lang in data:
        hyp = recognize(reader, name, bgr)
        rows.append({"gt": gt, "hyp": hyp, "lang": lang, "cer": _cer(gt, hyp), "wer": _token_wer(gt, hyp)})
    secs = time.perf_counter() - t0
    summary = agg(rows)
    summary["seconds"] = round(secs, 2)
    summary["seconds_per_image"] = round(secs / max(len(data), 1), 3)
    per_seed = []
    # samples are concatenated seed blocks of equal length
    nseed = summary.get("_nseed")
    return summary, rows


def main() -> None:
    overlap = set(VAL_BN + VAL_EN) & set(SELECT_BN + SELECT_EN + REPORT_BN + REPORT_EN)
    if overlap:
        raise SystemExit(f"validation phrases overlap a held-out list: {overlap}")
    font = _ocr_font()
    val = []
    for seed in (9000, 9100, 9200):
        val.extend(samples(VAL_BN, VAL_EN, 24, seed))
    print("validation images", len(val), "font", font, flush=True)
    import easyocr
    try:
        reader = easyocr.Reader(["bn", "en"], gpu=True, verbose=False)
        device = "cuda"
    except Exception as exc:
        print("gpu reader failed", exc, flush=True)
        reader = easyocr.Reader(["bn", "en"], gpu=False, verbose=False)
        device = "cpu"
    glass_gray.ocr = OCRMode()
    done = {}
    if OUT.is_file():
        done = {row["method"]: row for row in json.loads(OUT.read_text(encoding="utf-8")).get("methods", [])}
    methods = []
    for name in METHODS:
        if name in done and "error" not in done[name]:
            methods.append(done[name])
            print("loaded", name, done[name]["overall"]["cer"], flush=True)
            continue
        print("scoring", name, flush=True)
        try:
            summary, _ = score_method(reader, name, val)
        except Exception as exc:
            summary = {"method": name, "error": str(exc)}
            methods.append(summary)
            OUT.write_text(json.dumps({"device": device, "methods": methods}, ensure_ascii=False, indent=2), encoding="utf-8")
            print("FAILED", name, exc, flush=True)
            continue
        summary["method"] = name
        methods.append(summary)
        OUT.write_text(json.dumps({"device": device, "n_val_images": len(val), "seeds": [9000, 9100, 9200],
                                   "methods": methods}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(name, round(summary["overall"]["cer"], 4), "bn", round(summary["bn"]["cer"], 4),
              "sec", summary["seconds"], flush=True)

    ranked = [m for m in methods if "overall" in m]
    ranked.sort(key=lambda m: (m["overall"]["cer"], m["bn"]["cer"], m["seconds"]))
    winner = ranked[0]["method"]
    print("WINNER", winner, ranked[0]["overall"]["cer"], flush=True)

    from ocr_repair import repair_ocr_text

    heldout = {}
    for label, bn, en, per, seed in (
        ("select24", SELECT_BN, SELECT_EN, 40, 5000),
        ("report80", REPORT_BN, REPORT_EN, 40, 0),
    ):
        data = samples(bn, en, per, seed)
        block = {}
        for name in ("baseline", winner):
            rows = []
            t0 = time.perf_counter()
            for gt, bgr, lang in data:
                hyp = recognize(reader, name, bgr)
                fixed = repair_ocr_text(hyp)
                rows.append({"lang": lang, "cer": _cer(gt, hyp), "wer": _token_wer(gt, hyp),
                             "cer_repaired": _cer(gt, fixed), "wer_repaired": _token_wer(gt, fixed)})
            secs = time.perf_counter() - t0
            raw_rows = [{"lang": r["lang"], "cer": r["cer"], "wer": r["wer"]} for r in rows]
            rep_rows = [{"lang": r["lang"], "cer": r["cer_repaired"], "wer": r["wer_repaired"]} for r in rows]
            block[name] = {"raw": agg(raw_rows), "repaired": agg(rep_rows), "seconds": round(secs, 2)}
            print(label, name, block[name]["raw"]["overall"]["cer"], "bn", block[name]["raw"]["bn"]["cer"], flush=True)
        heldout[label] = block
    payload = {"device": device, "selection": "lowest validation raw CER; repaired CER not used",
               "winner": winner, "ranking": [{k: m[k] for k in ("method", "overall", "bn", "en", "seconds", "seconds_per_image")} for m in ranked],
               "heldout_read_once": heldout}
    (RESULTS / "ocr_improve_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", RESULTS / "ocr_improve_summary.json")


if __name__ == "__main__":
    main()
