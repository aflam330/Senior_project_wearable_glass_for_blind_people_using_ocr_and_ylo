"""Checks of the text reader after the single-detector-pass change (needs EasyOCR and its models; about a minute).

  * the glass path (one detector pass, then recognition) gives the same text as EasyOCR's readtext on benchmark images;
  * the text detector is called once per capture, for a short sign, a blank picture and a page;
  * a blank picture answers "no text found" and does not invent letters;
  * a page is spoken line by line.

    python scripts/test_ocr_single_pass.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from ocr_bench import bench as B  # noqa: E402
from ocr_bench.page_bench import _ocr_mode, make_page, reader_fn  # noqa: E402
from ocr_text_region import clean_text  # noqa: E402

m = _ocr_mode()
calls = {"n": 0}
real_detect = m._reader.detect


def counting_detect(*a, **k):
    calls["n"] += 1
    return real_detect(*a, **k)


m._reader.detect = counting_detect
old = reader_fn(m, 1200, 1, config.OCR_V2_DECODER)            # EasyOCR's readtext, as the benchmark has always scored it
items = [it for it in B.build("val", 0)][::9]                 # 16 images: clean, distorted and photo scenes, both scripts
same = 0
for it in items:
    img = B.decode(it)
    m.speak_line = None
    new = clean_text(" ".join(m._read_v2(img)))
    same += int(new == old(img))
assert same == len(items), f"glass path differs from readtext on {len(items) - same} of {len(items)} images"

sign = B.decode(next(it for it in B.build("val", 0) if it["cond"] == "scene"))
spoken = []
m.speak_line = spoken.append                                  # as in the glass: the line-by-line path is armed
calls["n"] = 0
t = time.perf_counter(); out = m.process_frame(sign); t_sign = time.perf_counter() - t
assert calls["n"] == 1, f"short sign: detector ran {calls['n']} times"

blank = np.full((480, 640, 3), 128, np.uint8)
calls["n"] = 0
t = time.perf_counter(); out = m.process_frame(blank); t_blank = time.perf_counter() - t
assert calls["n"] == 1 and m._stored_text is None and "পাওয়া যায়নি" in out, (calls, out, m._stored_text)

page, gt = make_page(1, 1280, 720)
spoken.clear(); calls["n"] = 0
t = time.perf_counter(); m.process_frame(page); t_page = time.perf_counter() - t
assert calls["n"] == 1 and m.streamed and len(spoken) == 12 and B.cer(gt, m._stored_text or "") < 0.02, (calls, len(spoken))
print(f"single pass: all checks pass ({len(items)} images identical to readtext; sign {t_sign:.1f} s, blank {t_blank:.1f} s, "
      f"12-line page {t_page:.1f} s on this machine)")
