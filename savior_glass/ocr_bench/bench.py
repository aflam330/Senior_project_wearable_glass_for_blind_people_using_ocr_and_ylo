"""OCR benchmark for the glass: correctly shaped Bangla + English text, held-out phrase sets, 3 seeds.

Phrase sets (no phrase appears in two sets):
  lexicon  the 24 phrases of assets/ocr_lexicon.txt (old 80-image set; repair favours them; legacy only)
  select   the 24 phrases used once to choose the canvas size (2026-10-01)
  val      48 new phrases: every setting is chosen here
  test     48 new phrases: never used for any choice; read once per finished method
Fonts are split the same way: VAL_FONTS for val, TEST_FONTS for test / lexicon / select, and
TRAIN_FONTS reserved for fine-tuning (never scored).
Conditions per phrase and seed:
  clean  dark text on a light page, 640 px wide
  wild   rotation +-6 deg, blur, low resolution, grey page (as the old "wild" images)
  scene  the text printed on a sign pasted into a real COCO photo, 640 x 480 (the glass frame size)
Seeds 0, 1, 2 change every random draw (font, size, position, distortion, background).
Metrics: CER = edit distance / reference length, per image, then averaged; WER on words; exact match.
"""
from __future__ import annotations

import json
import random
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WS = ROOT.parent
FONTS = HERE / "fonts"
WIN = Path(r"C:\Windows\Fonts")
COCO = WS / "data set" / "coco2017" / "val2017"
CACHE = HERE / "cache"

BN_FONT = {
    "val": [(str(WIN / "Nirmala.ttc"), 0), (str(FONTS / "NotoSerifBengali.ttf"), 0)],
    "test": [(str(WIN / "Nirmala.ttc"), 0), (str(FONTS / "NotoSansBengali.ttf"), 0), (str(FONTS / "TiroBangla-Regular.ttf"), 0)],
    "train": [(str(FONTS / f), 0) for f in ("HindSiliguri-Regular.ttf", "AnekBangla.ttf", "BalooDa2.ttf", "Galada-Regular.ttf",
                                             "Mina-Regular.ttf", "Atma-Regular.ttf")],
}
EN_FONT = {
    "val": [(str(WIN / "georgia.ttf"), 0), (str(WIN / "verdana.ttf"), 0)],
    "test": [(str(WIN / "arial.ttf"), 0), (str(WIN / "times.ttf"), 0), (str(WIN / "calibri.ttf"), 0)],
    "train": [(str(WIN / f), 0) for f in ("segoeui.ttf", "tahoma.ttf", "trebuc.ttf", "cour.ttf")],
}

PHRASES = {
    "lexicon": {
        "bn": ["বাংলাদেশ ব্যাংক", "দশ টাকা", "বিশ টাকা", "এই পথ বন্ধ", "স্টেশন ১২", "হাসপাতাল", "ঔষধের দোকান", "বাস স্টপ",
               "ডানদিকে যান", "বাঁদিকে যান", "মূল ফটক", "টিকিট কাউন্টার"],
        "en": ["EXIT 12", "Bus Stop", "Hospital Gate", "Ticket Counter", "Main Entrance", "Pharmacy", "Turn Right", "Turn Left",
               "Platform 3", "Danger Keep Out", "Glycemic", "Digestive Biscuit"],
    },
    "select": {
        "bn": ["সাবধান", "প্রবেশ নিষেধ", "শৌচাগার", "খোলা আছে", "পানি", "দুধ", "চাল পাঁচ কেজি", "লিফট", "সিঁড়ি", "জরুরি নির্গমন",
               "নামাজের ঘর", "অপেক্ষা কক্ষ"],
        "en": ["Lift", "Stairs", "Toilet", "No Entry", "Open", "Closed", "Milk", "Rice 5 kg", "Fire Exit", "Waiting Room", "Push", "Pull"],
    },
    "val": {
        "bn": ["ধূমপান নিষেধ", "রান্নাঘর", "মেয়াদ উত্তীর্ণের তারিখ", "দাম ৮৫ টাকা", "সরকারি প্রাথমিক বিদ্যালয়", "ডাকঘর",
               "ভিতরে আসুন", "জুতা খুলে প্রবেশ করুন", "চিনি এক কেজি", "পরীক্ষার হল", "ব্যবস্থাপক", "ঢাকা মেডিকেল কলেজ",
               "নিরাপদ পানীয় জল", "সোমবার বন্ধ", "রক্ত পরীক্ষা কেন্দ্র", "আলু", "বিদ্যুৎ বিল", "শিশুদের খেলার মাঠ",
               "ক্যাশ কাউন্টার", "সাবান", "ভাড়া দেওয়া হবে", "চক্ষু বিভাগ", "লবণ", "ট্রেনের সময়সূচি"],
        "en": ["Emergency Room", "Do Not Disturb", "Best Before 2027", "Price 120 Taka", "Library", "Reception", "Wet Floor",
               "Parking", "Post Office", "Exam Hall", "Manager", "Drinking Water", "Closed on Monday", "Blood Bank", "Potato",
               "Electricity Bill", "Playground", "Cash Counter", "Soap", "To Let", "Eye Clinic", "Salt", "Train Schedule",
               "Paracetamol 500 mg"],
    },
    "test": {
        "bn": ["মহিলাদের জন্য সংরক্ষিত", "ফার্মেসি", "উৎপাদনের তারিখ", "মূল্য ২৫০ টাকা", "জেলা প্রশাসকের কার্যালয়", "থানা",
               "বাইরে যাওয়ার পথ", "গাড়ি রাখা নিষেধ", "ডাল আধা কেজি", "শ্রেণিকক্ষ", "প্রধান শিক্ষক", "রাজশাহী বিশ্ববিদ্যালয়",
               "অগ্নি নির্বাপক", "শুক্রবার খোলা", "জরুরি বিভাগ", "পেঁয়াজ", "গ্যাস সংযোগ", "বৃদ্ধাশ্রম", "টাকা জমা দিন",
               "তেল", "বিক্রয় হবে", "দন্ত চিকিৎসা", "মরিচ", "বাসের ভাড়া"],
        "en": ["Ladies Only", "Pharmacy Open", "Manufactured On", "Price 250 Taka", "District Office", "Police Station",
               "Way Out", "No Parking", "Lentils Half kg", "Classroom", "Head Teacher", "Rajshahi University",
               "Fire Extinguisher", "Open on Friday", "Accident Ward", "Onion", "Gas Connection", "Old Age Home",
               "Deposit Money Here", "Cooking Oil", "For Sale", "Dental Care", "Chilli", "Bus Fare 30 Taka"],
    },
}
CONDITIONS = ("clean", "wild", "scene")
SEEDS = (0, 1, 2)


def nfc(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).split())


def lev(a, b) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(gt: str, hyp: str) -> float:
    gt, hyp = nfc(gt), nfc(hyp)
    return lev(gt, hyp) / max(len(gt), 1)


def wer(gt: str, hyp: str) -> float:
    a, b = nfc(gt).split(), nfc(hyp).split()
    return lev(a, b) / max(len(a), 1)


def _text_image(text, lang, split, rng, ink=None, bg=None):
    from .shaped_text import render_mask  # needs uharfbuzz; only for building new sets
    fonts = (BN_FONT if lang == "bn" else EN_FONT)["val" if split == "val" else "train" if split == "train" else "test"]
    path, idx = fonts[rng.integers(len(fonts))]
    size = int(rng.integers(30, 46))
    m = render_mask(text, path, size, idx).astype(np.float32) / 255
    ink = ink if ink is not None else float(rng.integers(0, 50))
    bg = bg if bg is not None else float(rng.integers(215, 256))
    img = bg * (1 - m) + ink * m
    return img.astype(np.uint8), path


def make_image(text, lang, split, cond, rng, coco_paths):
    img, font = _text_image(text, lang, split, rng)
    pad = int(rng.integers(12, 30))
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=int(img[0, 0]))
    if cond in ("clean", "wild"):
        w = 640
        h = max(160, img.shape[0] + 20)
        page = np.full((h, w), int(img[0, 0]), np.uint8)
        s = min(1.0, (w - 20) / img.shape[1])
        t = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        y, x = (h - t.shape[0]) // 2, int(rng.integers(0, max(1, w - t.shape[1])))
        page[y:y + t.shape[0], x:x + t.shape[1]] = t
        out = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
        if cond == "wild":
            M = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-6, 6)), 1.0)
            out = cv2.warpAffine(out, M, (w, h), borderValue=(int(img[0, 0]),) * 3)
            out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.3, 1.1)))
            if rng.random() < 0.5:
                out = cv2.resize(cv2.resize(out, None, fx=0.55, fy=0.55, interpolation=cv2.INTER_AREA), (w, h))
        return out, font
    # scene: sign on a real photo, perspective tilt, camera blur and noise
    bgimg = cv2.imread(str(coco_paths[rng.integers(len(coco_paths))]))
    bgimg = cv2.resize(bgimg, (640, 480), interpolation=cv2.INTER_AREA)
    sign = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    tw = int(rng.uniform(0.40, 0.85) * 640)
    s = min(tw / sign.shape[1], 200 / sign.shape[0])
    sign = cv2.resize(sign, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sh, sw = sign.shape[:2]
    x0, y0 = int(rng.integers(0, 640 - sw + 1)), int(rng.integers(0, 480 - sh + 1))
    j = lambda: rng.uniform(-0.06, 0.06)
    src = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    dst = np.float32([[x0 + j() * sw, y0 + j() * sh], [x0 + sw + j() * sw, y0 + j() * sh],
                      [x0 + sw + j() * sw, y0 + sh + j() * sh], [x0 + j() * sw, y0 + sh + j() * sh]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(sign, M, (640, 480))
    mask = cv2.warpPerspective(np.full((sh, sw), 255, np.uint8), M, (640, 480)) > 0
    out = bgimg.copy()
    out[mask] = warped[mask]
    out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.4, 1.2)))
    out = np.clip(out.astype(np.float32) * rng.uniform(0.7, 1.15) + rng.normal(0, 4, out.shape), 0, 255).astype(np.uint8)
    return out, font


def build(name: str, seed: int):
    """Images of phrase set `name` for one seed: every phrase x condition. Cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}_seed{seed}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return list(z["items"])
    split = "val" if name == "val" else "test"
    coco = sorted(COCO.glob("*.jpg"))
    coco = coco[: len(coco) // 2] if split == "val" else coco[len(coco) // 2:]
    rng = np.random.default_rng(10_000 * (1 + list(PHRASES).index(name)) + seed)
    items = []
    for lang in ("bn", "en"):
        for text in PHRASES[name][lang]:
            for cond in CONDITIONS:
                img, font = make_image(text, lang, split, cond, rng, coco)
                ok, enc = cv2.imencode(".png", img)
                items.append({"gt": text, "lang": lang, "cond": cond, "font": Path(font).name, "png": enc.tobytes()})
    np.savez_compressed(f, items=np.array(items, dtype=object))
    return items


def decode(item) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(item["png"], np.uint8), cv2.IMREAD_COLOR)


def score(read_fn, name: str, seeds=SEEDS, limit=None, conds=CONDITIONS) -> dict:
    """read_fn(bgr) -> text. Returns averages overall, per language, per condition, per seed, and latency."""
    rows = []
    for seed in seeds:
        items = [it for it in build(name, seed) if it["cond"] in conds][:limit]
        for it in items:
            img = decode(it)
            t0 = time.perf_counter()
            hyp = read_fn(img)
            ms = (time.perf_counter() - t0) * 1000
            rows.append({"seed": seed, "lang": it["lang"], "cond": it["cond"], "gt": it["gt"], "hyp": hyp,
                         "cer": cer(it["gt"], hyp), "wer": wer(it["gt"], hyp), "exact": nfc(it["gt"]) == nfc(hyp), "ms": ms})
    return summarise(rows)


def summarise(rows) -> dict:
    def agg(rs):
        if not rs:
            return None
        return {"n": len(rs), "cer": float(np.mean([r["cer"] for r in rs])), "wer": float(np.mean([r["wer"] for r in rs])),
                "exact": float(np.mean([r["exact"] for r in rs]))}
    out = {"overall": agg(rows), "bn": agg([r for r in rows if r["lang"] == "bn"]), "en": agg([r for r in rows if r["lang"] == "en"])}
    out["by_cond"] = {c: {"all": agg([r for r in rows if r["cond"] == c]), "bn": agg([r for r in rows if r["cond"] == c and r["lang"] == "bn"])}
                      for c in sorted({r["cond"] for r in rows})}
    seeds = sorted({r["seed"] for r in rows})
    per = [agg([r for r in rows if r["seed"] == s]) for s in seeds]
    out["per_seed_cer"] = [p["cer"] for p in per]
    out["per_seed_bn_cer"] = [(agg([r for r in rows if r["seed"] == s and r["lang"] == "bn"]) or {}).get("cer") for s in seeds]
    out["cer_seed_sd"] = float(np.std(out["per_seed_cer"], ddof=1)) if len(per) > 1 else 0.0
    out["latency_ms_median"] = float(np.median([r["ms"] for r in rows]))
    out["rows"] = rows
    return out


def save(result: dict, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in result.items() if k != "rows"}
    keep["meta"] = meta
    keep["per_sample"] = [{k: r[k] for k in ("seed", "lang", "cond", "gt", "hyp", "cer")} for r in result["rows"]]
    path.write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")


def line(name: str, r: dict) -> str:
    return (f"{name:38s} CER {100*r['overall']['cer']:5.1f}  BN {100*r['bn']['cer']:5.1f}  EN {100*r['en']['cer']:5.1f}  "
            f"WER {100*r['overall']['wer']:5.1f}  clean {100*r['by_cond'].get('clean',{}).get('all',{}).get('cer',float('nan')) if r['by_cond'].get('clean') else float('nan'):5.1f}  "
            f"scene {100*r['by_cond']['scene']['all']['cer'] if 'scene' in r['by_cond'] else float('nan'):5.1f}  "
            f"seeds {[round(100*x,1) for x in r['per_seed_cer']]}  {r['latency_ms_median']:.0f} ms")
