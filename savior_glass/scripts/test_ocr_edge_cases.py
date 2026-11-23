"""Edge cases of the text reader (OCRMode, pipeline v2, the glass's own settings; needs EasyOCR, about a minute).

  * empty picture (no text boxes): "no text found" is answered, nothing is stored, nothing is invented
  * very small pictures (12 x 40 and 1 x 1 pixels): no exception, an answer is returned
  * low contrast (grey 120 text on grey 140): no exception; high-contrast text in the same layout is read
  * rotated text: a sign turned by 90 degrees is read when the rotation retry is on (OCR_ROTATE_LOWCONF > 0)
  * debug mode (OCR_DEBUG_DIR): original, boxes, crops and result.json are written for one capture

    python scripts/test_ocr_edge_cases.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from ocr_bench.page_bench import _ocr_mode  # noqa: E402

NOT_FOUND = "পাওয়া যায়নি"


def sign(text, fg=20, bg=235, w=640, h=480, scale=2.2):
    img = np.full((h, w, 3), bg, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 4)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_DUPLEX, scale, (fg,) * 3, 4, cv2.LINE_AA)
    return img


def main() -> None:
    m = _ocr_mode()
    m.speak_line = None
    fails = []

    def check(name, ok, info=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name} {info}")
        if not ok:
            fails.append(name)

    out = m.process_frame(np.full((480, 640, 3), 128, np.uint8))
    check("empty picture", NOT_FOUND in (out or "") and m._stored_text is None, repr(out))

    for shape in ((12, 40, 3), (1, 1, 3)):
        try:
            out = m.process_frame(np.full(shape, 200, np.uint8))
            check(f"tiny picture {shape[1]}x{shape[0]}", isinstance(out, str), repr(out))
        except Exception as exc:  # noqa: BLE001
            check(f"tiny picture {shape[1]}x{shape[0]}", False, f"{type(exc).__name__}: {exc}")

    hi = " ".join(m._read_v2(sign("EXIT 12")))
    check("high contrast read", "EXIT" in hi.upper(), repr(hi))
    try:
        lo = " ".join(m._read_v2(sign("EXIT 12", fg=120, bg=140)))
        check("low contrast, no exception", True, repr(lo))
    except Exception as exc:  # noqa: BLE001
        check("low contrast, no exception", False, f"{type(exc).__name__}: {exc}")

    turned = cv2.rotate(sign("STOP HERE", w=800, h=300), cv2.ROTATE_90_CLOCKWISE)
    saved = config.OCR_ROTATE_LOWCONF, config.OCR_ROTATE_FALLBACK
    config.OCR_ROTATE_FALLBACK, config.OCR_ROTATE_LOWCONF = False, 0.3
    rot = " ".join(m._read_v2(turned))
    config.OCR_ROTATE_LOWCONF, config.OCR_ROTATE_FALLBACK = saved
    check("rotated sign read with rotation retry", "STOP" in rot.upper(), repr(rot))

    with tempfile.TemporaryDirectory() as d:
        os.environ["OCR_DEBUG_DIR"] = d
        try:
            m._read_v2(sign("EXIT 12"))
        finally:
            del os.environ["OCR_DEBUG_DIR"]
        runs = list(Path(d).iterdir())
        ok = len(runs) == 1 and all((runs[0] / f).exists() for f in ("original.jpg", "boxes.jpg", "result.json"))
        res = json.loads((runs[0] / "result.json").read_text(encoding="utf-8")) if ok else {}
        check("debug mode writes original, boxes, crops, result", ok and bool(res.get("boxes"))
              and len(list((runs[0] / "crops").glob("*.png"))) == len(res["boxes"]), str(res.get("text")))

    if fails:
        print("ocr edge cases: FAILED " + ", ".join(fails))
        sys.exit(1)
    print("ocr edge cases: all checks pass")


if __name__ == "__main__":
    main()
