"""Text mode on signs and labels: the AI HAT text detector (OCR_DETECTOR=hailo) against the Pi's CPU detector
(8-bit CRAFT), with the same letter recogniser and the same app code (OCRMode.process_frame, plain path).

60 pictures at the camera's 640 x 480, made with a fixed seed: 1 to 4 lines of 1 to 3 common words (Bangla from
ocr_bench/lexicon/bn_50k.txt in one of the nine Bangla fonts of ocr_bench/fonts, English in DejaVu Sans; 25 % of
lines English, a few with a number), dark on light or light on dark, tilted up to 4 degrees, blurred 0.5 to 1.3 px,
uneven light and camera noise. Character error against the rendered text in reading order.

python hailo_ocr_sign_eval.py <out.json> [n]   (run from savior_glass/)
"""
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

os.environ["OCR_DETECTOR"] = "hailo"
HERE = Path(__file__).resolve()
GLASS = HERE.parents[2] / "savior_glass"
sys.path.insert(0, str(GLASS))
sys.path.insert(0, str(HERE.parent))
os.chdir(GLASS)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import config  # noqa: E402
from hailo_ocr_glass_eval import cer  # noqa: E402

FONTS = sorted(str(p) for p in (GLASS / "ocr_bench" / "fonts").glob("*.ttf"))
EN_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def words(path, n):
    out = []
    for line in open(path, encoding="utf-8"):
        w = line.split()[0]
        if len(w) >= 2 and w.isalpha() if path.endswith("en_50k.txt") else len(w) >= 2 and not any(c in w for c in "।,.!?"):
            out.append(w)
        if len(out) >= n:
            break
    return out


def make(rng, bn, en):
    lines = []
    for _ in range(int(rng.integers(1, 5))):
        english = rng.random() < 0.25
        pool = en if english else bn
        ws = [pool[int(rng.integers(len(pool)))] for _ in range(int(rng.integers(1, 4)))]
        if english:
            ws = [w.capitalize() for w in ws]
        if rng.random() < 0.15:
            ws.append(str(int(rng.integers(2, 999))) if english else "".join("০১২৩৪৫৬৭৮৯"[int(d)] for d in str(int(rng.integers(2, 999)))))
        lines.append(" ".join(ws))
    W, H = 640, 480
    bn_font = FONTS[int(rng.integers(len(FONTS)))]
    dark = rng.random() < 0.2
    bg, fg = (int(rng.integers(20, 60)), int(rng.integers(200, 245))) if dark else (int(rng.integers(200, 245)), int(rng.integers(10, 60)))
    margin = int(0.08 * W)
    size = int(min(0.6 * (H - 2 * margin) / len(lines), 0.11 * W))
    while True:
        fonts = [ImageFont.truetype(EN_FONT if t.isascii() else bn_font, size, layout_engine=ImageFont.Layout.RAQM) for t in lines]
        if max(f.getlength(t) for f, t in zip(fonts, lines)) <= W - 2 * margin or size <= 16:
            break
        size -= 2
    page = Image.new("L", (W, H), bg)
    d = ImageDraw.Draw(page)
    step = (H - 2 * margin) / len(lines)
    for i, (f, t) in enumerate(zip(fonts, lines)):
        d.text((margin + int(rng.integers(0, 20)), margin + i * step + 0.5 * (step - size * 1.25)), t, font=f, fill=fg)
    img = np.asarray(page).astype(np.float32)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), float(rng.uniform(-4, 4)), 1.0)
    img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    yy, xx = np.mgrid[0:H, 0:W]
    img *= 0.8 + 0.3 * (xx / W if rng.random() < 0.5 else yy / H)         # uneven light
    img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.5, 1.3)))
    img += rng.normal(0, 4, img.shape)
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), " ".join(lines)


def main():
    out = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    rng = np.random.default_rng(20261123)
    bn = words(str(GLASS / "ocr_bench/lexicon/bn_50k.txt"), 5000)
    en = words(str(GLASS / "ocr_bench/lexicon/en_50k.txt"), 5000)
    cases = [make(rng, bn, en) for _ in range(n)]
    pics = out.parent / "sign_eval_pictures"
    pics.mkdir(exist_ok=True)
    for i, (f, _) in enumerate(cases):
        cv2.imwrite(str(pics / f"sign_{i:02d}.png"), f)

    from modes.ocr_mode import OCRMode
    m = OCRMode()
    m.activate()
    r = m._reader
    hailo_detect = r.__dict__.get("detect")
    assert hailo_detect is not None and r.__dict__.get("_hailo_det") is not None, "Hailo text detector did not install"
    cpu_detect = type(r).detect.__get__(r)
    m.speak_line = None
    r.detect = cpu_detect
    m.process_frame(cases[0][0])                      # warm-up
    rows = []
    for i, (frame, truth) in enumerate(cases):
        row = {"i": i, "truth": truth}
        for name, fn in (("hailo", hailo_detect), ("cpu", cpu_detect)):
            r.detect = fn
            t0 = time.perf_counter()
            m.process_frame(frame)
            row[name] = {"s": round(time.perf_counter() - t0, 3), "text": m._stored_text or "",
                         "cer": round(cer(m._stored_text or "", truth), 2)}
        rows.append(row)
        print(i, truth, "| hailo", row["hailo"]["cer"], row["hailo"]["s"], "| cpu", row["cpu"]["cer"], row["cpu"]["s"], flush=True)

    def summ(k):
        c = [x[k]["cer"] for x in rows]
        return {"cer_mean": round(st.mean(c), 2), "cer_median": round(st.median(c), 2),
                "exact": sum(v == 0 for v in c), "time_s_median": round(st.median(x[k]["s"] for x in rows), 2),
                "time_s_mean": round(st.mean(x[k]["s"] for x in rows), 2)}
    res = {"n": n, "seed": 20261123, "summary": {"hailo": summ("hailo"), "cpu": summ("cpu")},
           "hailo_better": sum(x["hailo"]["cer"] < x["cpu"]["cer"] for x in rows),
           "cpu_better": sum(x["cpu"]["cer"] < x["hailo"]["cer"] for x in rows), "rows": rows}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("summary", "hailo_better", "cpu_better")}, indent=1))
    m.cleanup()


if __name__ == "__main__":
    main()
