"""How far does a camera picture have to be enlarged before the text detector sees it?

The glass enlarges a 640-pixel-wide picture to 1,200 pixels before reading (config.OCR_READ_WIDTH). The detector's
time grows with the number of pixels, and on the Raspberry Pi 5 one detector pass took 11 to 18 s (13 October 2026).
Reading at the camera's own 640 pixels was measured on 11 October: faster, but +0.74 points of validation error, and
rejected. Widths in between were never measured. This script measures them.

Rule, fixed before the run (2026-10-14):
  For each detector (full-precision ONNX, and the 8-bit one the Pi uses), widths 1,040, 960 and 880 are scored on the
  rendered validation set (432 images, 640 pixels wide) and on the real validation words (150 + 150), through the
  glass's own reading function on the CPU. A width is admissible for a detector if its rendered-validation CER is
  within 0.5 points of the same detector at 1,200 AND its real-validation mean CER is not more than 0.5 points above
  the same detector at 1,200. The smallest admissible width is adopted for that detector. The test sets are read once,
  with the adopted width, only if a width below 1,200 is adopted.
  Timing here is on a laptop that is running another job; it is recorded but decides nothing.
  The widths are scored in the order 1,200, 960, 1,040, 880 and each result is saved as it finishes.

    python -u -m ocr_bench.width_study
Output: results/ocr_bench/width_study.json
"""
from __future__ import annotations

import json
import time

from . import bench as B
from .page_bench import OUT, _ocr_mode
from .real_words_tune import fetch, score_real
from .train_real_words import split

RES = OUT / "width_study.json"
WIDTHS = (1200, 1040, 960, 880)
DETECTORS = ("onnx", "onnx_int8")


def main(threads=4):
    import torch
    import config
    import ocr_detector_onnx
    from ocr_text_region import clean_text
    torch.set_num_threads(threads)
    _, va, te = split()
    val_items = fetch(va)
    m = _ocr_mode()
    m.speak_line = None

    def read(frame):
        return clean_text(" ".join(m._read_v2(frame)))

    res = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else {"date": "2026-10-14", "validation": {}}

    def use(det, width):
        config.OCR_DETECTOR = det
        used = ocr_detector_onnx.install(m._reader)
        assert used.endswith(".onnx"), used
        config.OCR_READ_WIDTH = width
        return used

    for w in (1200, 960, 1040, 880):                       # the most informative widths first; saved one by one
        for det in DETECTORS:
            key = f"{det} {w}"
            if key in res["validation"]:
                continue
            used = use(det, w)
            t0 = time.perf_counter()
            real = score_real(read, val_items)
            rend = B.score(read, "val")
            res["validation"][key] = {"detector_file": used, "width": w, "real_val": real,
                                      "rendered_val_cer": round(rend["overall"]["cer"], 4), "rendered_val_cer_bn": round(rend["bn"]["cer"], 4),
                                      "rendered_val_cer_en": round(rend["en"]["cer"], 4),
                                      "rendered_median_ms_busy_laptop": round(rend["latency_ms_median"]), "minutes": round((time.perf_counter() - t0) / 60, 1)}
            RES.write_text(json.dumps(res, indent=1), encoding="utf-8")
            print(key, res["validation"][key], flush=True)

    res["chosen"] = {}
    for det in DETECTORS:
        base = res["validation"][f"{det} 1200"]
        ok = [w for w in WIDTHS[1:] if f"{det} {w}" in res["validation"]
              if res["validation"][f"{det} {w}"]["rendered_val_cer"] - base["rendered_val_cer"] <= 0.005
              and res["validation"][f"{det} {w}"]["real_val"]["mean_cer"] - base["real_val"]["mean_cer"] <= 0.005]
        res["chosen"][det] = {"admissible": ok, "width": min(ok) if ok else 1200}
    RES.write_text(json.dumps(res, indent=1), encoding="utf-8")
    res.setdefault("test", {})
    for det in DETECTORS:
        w = res["chosen"][det]["width"]
        if w == 1200 or det in res["test"]:
            continue
        use(det, w)
        real_t = score_real(read, fetch(te))
        t = B.score(read, "test")
        res["test"][det] = {"width": w, "real_words": real_t, "rendered": {"cer": round(t["overall"]["cer"], 4), "cer_bn": round(t["bn"]["cer"], 4),
                                                                           "cer_en": round(t["en"]["cer"], 4), "wer": round(t["overall"]["wer"], 4)}}
        RES.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "validation"}, indent=1))


if __name__ == "__main__":
    main()
