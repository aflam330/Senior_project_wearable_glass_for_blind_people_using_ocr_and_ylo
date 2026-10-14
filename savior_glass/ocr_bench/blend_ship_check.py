"""Two checks of the recogniser chosen by ocr_bench/real_ft_eval.py (blend 0.5, seed 44), as it is shipped.

1. The shipped copy (models/ocr_recognizer_real_blend/: half precision in two parts, because GitHub refuses files
   over 100 MB) must read the validation sets as the full-precision copy did in real_ft_eval.py
   (real validation mean CER 0.3665, rendered validation CER 0.0263).
2. The Raspberry Pi's configuration (8-bit text detector, read width 1,200) with this recogniser, on the same
   validation sets, so the Pi's setting has its own validation figure. Nothing is chosen from it.

    python -u -m ocr_bench.blend_ship_check
Output: results/ocr_bench/blend_ship_check.json
"""
from __future__ import annotations

import json

from . import bench as B
from .page_bench import OUT, _ocr_mode
from .real_words_tune import fetch, score_real
from .train_real_words import split

RES = OUT / "blend_ship_check.json"
SHIPPED = "models/ocr_recognizer_real_blend"


def main(threads=4):
    import os
    import torch
    import config
    import ocr_detector_onnx
    import ocr_recognizer_ft as FT
    from ocr_text_region import clean_text
    torch.set_num_threads(threads)
    _, va, _ = split()
    val_items = fetch(va)
    m = _ocr_mode()
    m.speak_line = None
    assert FT.install(m._reader, os.path.join(config.BASE_DIR, SHIPPED)) != "released"

    def read(frame):
        return clean_text(" ".join(m._read_v2(frame)))

    res = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else {"date": "2026-10-14"}
    for name, det in (("shipped, full-precision detector, width 1200", "onnx"), ("Pi: shipped, 8-bit detector, width 1200", "onnx_int8")):
        if name in res:
            continue
        config.OCR_DETECTOR, config.OCR_READ_WIDTH = det, 1200
        ocr_detector_onnx.install(m._reader)
        rend = B.score(read, "val")
        res[name] = {"real_val": score_real(read, val_items), "rendered_val_cer": round(rend["overall"]["cer"], 4),
                     "rendered_val_cer_bn": round(rend["bn"]["cer"], 4), "rendered_val_cer_en": round(rend["en"]["cer"], 4)}
        RES.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(name, res[name], flush=True)


if __name__ == "__main__":
    main()
