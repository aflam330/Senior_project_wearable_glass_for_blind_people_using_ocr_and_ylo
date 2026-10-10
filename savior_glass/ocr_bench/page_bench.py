"""Whole-page OCR speed and accuracy on the CPU (the Raspberry Pi has no GPU).

A "page" is 12 lines of benchmark phrases (Bangla and English mixed), rendered with correct shaping and
photographed at the glass camera size. Only VAL phrases and VAL fonts are used, because settings are
chosen on these pages; the TEST split is not read here.

    python -m ocr_bench.page_bench profile          # where the time goes (detection vs recognition)
    python -m ocr_bench.page_bench compare          # speed and CER of each candidate setting
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import bench as B

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[0] / "results" / "ocr_bench"
LINES = 12


def make_page(seed: int, width: int = 1280, height: int = 720):
    """A printed page photographed by the camera: lines of val phrases on a light page, slight blur and noise."""
    rng = np.random.default_rng(500 + seed)
    phrases = [(t, "bn") for t in B.PHRASES["val"]["bn"]] + [(t, "en") for t in B.PHRASES["val"]["en"]]
    pick = [phrases[i] for i in rng.choice(len(phrases), LINES, replace=False)]
    rows, gt = [], []
    for text, lang in pick:
        img, _ = B._text_image(text, lang, "val", rng, ink=20.0, bg=235.0)
        rows.append(img)
        gt.append(text)
    pw = max(r.shape[1] for r in rows) + 80
    lh = max(r.shape[0] for r in rows) + 14
    page = np.full((lh * LINES + 80, pw), 235, np.uint8)
    for i, r in enumerate(rows):
        page[40 + i * lh: 40 + i * lh + r.shape[0], 40: 40 + r.shape[1]] = r
    s = min(width / page.shape[1], height / page.shape[0])
    page = cv2.resize(page, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width), 225, np.uint8)
    y, x = (height - page.shape[0]) // 2, (width - page.shape[1]) // 2
    canvas[y:y + page.shape[0], x:x + page.shape[1]] = page
    out = cv2.GaussianBlur(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), (0, 0), 0.6)
    out = np.clip(out.astype(np.float32) + rng.normal(0, 3, out.shape), 0, 255).astype(np.uint8)
    return out, " ".join(gt)


def _ocr_mode():
    sys.path.insert(0, str(HERE.parents[0]))
    import os
    os.environ["OCR_GPU"] = "0"          # the Pi's path
    import config  # noqa: F401
    from modes.ocr_mode import OCRMode
    m = OCRMode()
    m.activate()
    return m


def profile(sizes=((640, 480), (1280, 720))):
    m = _ocr_mode()
    r = m._reader
    for w, h in sizes:
        img, _ = make_page(0, w, h)
        m.process_frame(img)            # warm-up
        t0 = time.perf_counter()
        m.process_frame(img)
        total = time.perf_counter() - t0
        g = cv2.cvtColor(cv2.resize(img, (1200, int(h * 1200 / w))) if w < 1200 else img, cv2.COLOR_BGR2GRAY)
        t0 = time.perf_counter()
        hl, fl = r.detect(g, canvas_size=2560, mag_ratio=1.0, text_threshold=0.8, low_text=0.3, link_threshold=0.4)
        det = time.perf_counter() - t0
        print(f"{w}x{h}: whole capture {total:.2f} s; detection alone {det:.2f} s; {len(hl[0])} text boxes")


CANDIDATES = {
    # name: (width the frame is scaled to before detection, recognition batch size, decoder)
    "current (1200 px, batch 1, beam)": (1200, 1, "beamsearch"),
    "batch 16": (1200, 16, "beamsearch"),
    "batch 16 + greedy": (1200, 16, "greedy"),
    "960 px + batch 16": (960, 16, "beamsearch"),
    "native size + batch 16": (0, 16, "beamsearch"),
}


def reader_fn(m, width, batch, decoder):
    """The glass's own v2 pipeline (ocr_mode._read_v2) with the three speed settings exposed."""
    import config
    from ocr_text_region import clean_text, dominant_block, order_lines

    def read(frame):
        h, w = frame.shape[:2]
        if width and w != width and (w < width or w > 1600):
            frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_CUBIC if w < width else cv2.INTER_AREA)
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        kw = dict(config.OCR_V2_PARAMS)
        kw.update(decoder=decoder, canvas_size=config.OCR_CANVAS_SIZE, batch_size=batch)
        if decoder == "beamsearch":
            kw["beamWidth"] = config.OCR_BEAM_WIDTH
        det = m._reader.readtext(gray, detail=1, paragraph=False, width_ths=0.7, height_ths=0.7, **kw)
        return clean_text(order_lines(dominant_block(det, config.OCR_V2_MIN_CONF)))
    return read


def compare(threads=4):
    import torch
    torch.set_num_threads(threads)      # the Pi 5 has four cores
    m = _ocr_mode()
    pages = [make_page(s, w, h) for s in (0, 1, 2) for (w, h) in ((640, 480), (1280, 720))]
    res = {}
    for name, (width, batch, dec) in CANDIDATES.items():
        fn = reader_fn(m, width, batch, dec)
        fn(pages[0][0])                 # warm-up
        times, cers = [], []
        for img, gt in pages:
            t0 = time.perf_counter()
            hyp = fn(img)
            times.append(time.perf_counter() - t0)
            cers.append(B.cer(gt, hyp))
        val = B.score(fn, "val")        # the 48-phrase validation set, 3 seeds, 432 images
        res[name] = {"page_s_median": float(np.median(times)), "page_cer": float(np.mean(cers)),
                     "val_cer": val["overall"]["cer"], "val_cer_bn": val["bn"]["cer"], "val_cer_en": val["en"]["cer"], "val_ms_median": val["latency_ms_median"]}
        r = res[name]
        print(f"{name:34s} page {r['page_s_median']:5.2f} s  page CER {100*r['page_cer']:5.1f}%  "
              f"val CER {100*r['val_cer']:4.1f}% (bn {100*r['val_cer_bn']:4.1f}, en {100*r['val_cer_en']:4.1f})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "page_speed_compare.json").write_text(json.dumps({"threads": threads, "pages": len(pages), "results": res},
                                                            indent=1, ensure_ascii=False), encoding="utf-8")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "profile"
    if cmd == "profile":
        profile()
    elif cmd == "compare":
        compare()


if __name__ == "__main__":
    main()
