"""Ship the OCR candidate chosen by ocr_bench/ocr_v4_eval.py (SV: second text-detection pass + version-2 recogniser
blended half and half with the released weights), and check the shipped copy.

1. build: models/ocr_recognizer_real_v2_blend/ - the blend in half precision, in two parts (GitHub refuses files over
   100 MB), as models/ocr_recognizer_real_blend/ was shipped.
2. check: the shipped copy, with the glass's second-pass settings, must read the validation sets as SV did in
   ocr_v4_eval.py (real validation mean CER and rendered validation CER each within 0.005). The Raspberry Pi's setting
   (8-bit text detector, read width 1,200) is also scored on validation so it has its own figure; nothing is chosen
   from it. Test words are not read here.

    python -u -m ocr_bench.ship_v4 build
    python -u -m ocr_bench.ship_v4 check
Output: results/ocr_bench/ship_v4.json
"""
from __future__ import annotations

import json
import os
import sys

from . import bench as B
from .ocr_v3_eval import score
from .ocr_v4_eval import RES as V4, SECOND, V2
from .page_bench import OUT, _ocr_mode
from .real_ft_eval import released_weights
from .real_words_tune import fetch
from .train_real_words import split

RES = OUT / "ship_v4.json"
SHIPPED = "models/ocr_recognizer_real_v2_blend"


def build():
    import torch
    import config
    m = _ocr_mode()
    base = released_weights(m._reader)
    ft = torch.load(V2, map_location="cpu")
    sd = {k: ((0.5 * base[k].float() + 0.5 * v.float()).half() if v.is_floating_point() else v) for k, v in ft.items()}
    keys = list(sd)
    total = sum(sd[k].numel() for k in keys)
    out = os.path.join(config.BASE_DIR, SHIPPED)
    os.makedirs(out, exist_ok=True)
    parts, acc = ({}, {}), 0
    for k in keys:
        parts[0 if acc < total / 2 else 1][k] = sd[k]
        acc += sd[k].numel()
    for i, p in enumerate(parts, 1):
        torch.save(p, os.path.join(out, f"part{i}.pth"))
    for f in sorted(os.listdir(out)):
        print(f, os.path.getsize(os.path.join(out, f)) // 1_000_000, "MB")


def check(threads=4):
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
    config.OCR_SECOND_PASS, config.OCR_SECOND_PASS_MIN_CONF = SECOND, 0.2

    def read(frame):
        return clean_text(" ".join(m._read_v2(frame)))

    ref = json.loads(V4.read_text(encoding="utf-8"))["validation"]["SV"]
    res = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else {"date": "2026-11-23"}
    runs = (("shipped, as evaluated", None), ("Pi: shipped, 8-bit detector, width 1200", ("onnx_int8", 1200)))
    for name, pi in runs:
        if name in res:
            continue
        if pi:
            config.OCR_DETECTOR, config.OCR_READ_WIDTH = pi
            ocr_detector_onnx.install(m._reader)
        rend = B.score(read, "val")
        res[name] = {"real_val": score(read, val_items), "rendered_val_cer": round(rend["overall"]["cer"], 4)}
        RES.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(name, res[name]["real_val"]["mean_cer"], res[name]["rendered_val_cer"], flush=True)
    s = res["shipped, as evaluated"]
    res["matches_evaluated"] = (abs(s["real_val"]["mean_cer"] - ref["real_val"]["mean_cer"]) <= 0.005
                                and abs(s["rendered_val_cer"] - ref["rendered_val_cer"]) <= 0.005)
    RES.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("matches evaluated SV:", res["matches_evaluated"], "(SV:", ref["real_val"]["mean_cer"], ref["rendered_val_cer"], ")")


if __name__ == "__main__":
    {"build": build, "check": check}[sys.argv[1]]()
