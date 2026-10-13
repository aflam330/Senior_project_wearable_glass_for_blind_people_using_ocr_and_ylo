"""OCR response time on the device: how long the glass takes from the capture button to the text.

Times the glass's own OCRMode.process_frame() (the call main.py makes when ACTION is pressed in text mode) on
rendered signs and pages with known text, and splits each capture into text detection (CRAFT), recognition
(CRNN) and the rest (resize, CLAHE, block selection, clean-up). Two paths are timed:

  app    speak_line is set, as main.py sets it: pages of 4+ lines are read and spoken line by line, so the time
         to the first spoken line is recorded as well as the total.
  plain  speak_line is not set: the path scripts/benchmark_pi5.py times (its "ocr_mode" row).

    python ocr_response_bench.py <out.json> [phase.txt]

Run from anywhere inside the glass venv. No camera, no speech; the frames are rendered here.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
GLASS = HERE.parents[2] / "savior_glass"
sys.path.insert(0, str(GLASS))
sys.path.insert(0, str(GLASS / "scripts"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import config  # noqa: E402
from benchmark_pi5 import cpu_temp_c, host_info, rss_mb, text_frame, throttled  # noqa: E402
from ocr_bench.bench import cer  # noqa: E402

BN_FONT = str(GLASS / "ocr_bench" / "fonts" / "NotoSansBengali.ttf")
EN_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MIN_REPS, MAX_REPS, CASE_BUDGET_S = 3, 5, 60.0


def is_bn(text: str) -> bool:
    return any("ঀ" <= ch <= "৿" for ch in text)


def render(lines: list[str], width: int, height: int, seed: int = 0) -> np.ndarray:
    """Dark text on a light page at the camera size, with the slight blur and noise of a photographed print."""
    rng = np.random.default_rng(seed)
    margin = int(0.06 * width)
    size = int(min(0.72 * (height - 2 * margin) / len(lines), 0.11 * width))
    while True:  # shrink until the widest line fits
        fonts = [ImageFont.truetype(BN_FONT if is_bn(t) else EN_FONT, size, layout_engine=ImageFont.Layout.RAQM) for t in lines]
        if max(f.getlength(t) for f, t in zip(fonts, lines)) <= width - 2 * margin or size <= 12:
            break
        size -= 2
    page = Image.new("L", (width, height), 232)
    draw = ImageDraw.Draw(page)
    step = (height - 2 * margin) / len(lines)
    for i, (font, text) in enumerate(zip(fonts, lines)):
        draw.text((margin, margin + i * step + 0.5 * (step - size * 1.25)), text, font=font, fill=24)
    out = cv2.GaussianBlur(cv2.cvtColor(np.asarray(page), cv2.COLOR_GRAY2BGR), (0, 0), 0.6)
    return np.clip(out.astype(np.float32) + rng.normal(0, 3, out.shape), 0, 255).astype(np.uint8)


SIGN_BN = ["ধূমপান নিষেধ", "ডাকঘর", "নিরাপদ পানীয় জল"]
LABEL = ["Paracetamol 500 mg", "মেয়াদ উত্তীর্ণের তারিখ", "Best Before 2027", "দাম ৮৫ টাকা", "Price 120 Taka", "চিনি এক কেজি"]
PAGE = ["সরকারি প্রাথমিক বিদ্যালয়", "Emergency Room", "জুতা খুলে প্রবেশ করুন", "Closed on Monday", "ঢাকা মেডিকেল কলেজ",
        "Drinking Water", "রক্ত পরীক্ষা কেন্দ্র", "Electricity Bill", "শিশুদের খেলার মাঠ", "Train Schedule",
        "ভাড়া দেওয়া হবে", "Cash Counter"]


def cases() -> list[dict]:
    blank = np.clip(np.full((480, 640, 3), 128, np.float32) + np.random.default_rng(7).normal(0, 4, (480, 640, 3)), 0, 255)
    return [
        {"name": "sign_en_3_lines", "label": "English sign, 3 lines (640x480)", "frame": text_frame(),
         "truth": "EXIT 12 Bus Stop Main Entrance", "paths": ("plain", "app"), "note": "the frame scripts/benchmark_pi5.py uses"},
        {"name": "sign_bn_1_line", "label": "Bangla sign, 1 line (640x480)", "frame": render(SIGN_BN[:1], 640, 480, 1),
         "truth": SIGN_BN[0], "paths": ("plain", "app")},
        {"name": "sign_bn_3_lines", "label": "Bangla sign, 3 lines (640x480)", "frame": render(SIGN_BN, 640, 480, 2),
         "truth": " ".join(SIGN_BN), "paths": ("plain", "app")},
        {"name": "label_mixed_6_lines", "label": "Label, 6 lines Bangla + English (640x480)", "frame": render(LABEL, 640, 480, 3),
         "truth": " ".join(LABEL), "paths": ("app",)},
        {"name": "page_12_lines_640", "label": "Page, 12 lines Bangla + English (640x480)", "frame": render(PAGE, 640, 480, 4),
         "truth": " ".join(PAGE), "paths": ("app",)},
        {"name": "page_12_lines_1280", "label": "Page, 12 lines Bangla + English (1280x720)", "frame": render(PAGE, 1280, 720, 5),
         "truth": " ".join(PAGE), "paths": ("app",)},
        {"name": "no_text", "label": "No text in view (640x480)", "frame": blank.astype(np.uint8), "truth": None, "paths": ("app",)},
    ]


class StageTimer:
    """Adds up the time spent inside reader.detect and reader.recognize; readtext calls both through the instance."""

    def __init__(self, reader):
        self.s = {"detect": 0.0, "recognize": 0.0}
        self.calls = {"detect": 0, "recognize": 0}
        for name in self.s:
            setattr(reader, name, self._wrap(getattr(reader, name), name))

    def _wrap(self, fn, name):
        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.s[name] += time.perf_counter() - t0
                self.calls[name] += 1
        return timed

    def reset(self):
        for name in self.s:
            self.s[name], self.calls[name] = 0.0, 0


def med(values):
    return round(statistics.median(values), 1) if values else None


def main() -> None:
    out = Path(sys.argv[1])
    phase = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    def set_phase(name):
        if phase:
            phase.write_text(name)

    from modes.ocr_mode import OCRMode

    set_phase("ocr_load")
    m = OCRMode()
    t0 = time.perf_counter()
    m.activate()
    load_ms = (time.perf_counter() - t0) * 1000.0
    if m._reader is None:
        raise SystemExit("EasyOCR reader not loaded")
    detector = type(m._reader.detector).__name__
    timer = StageTimer(m._reader)
    spoken: list[float] = []
    all_cases = cases()

    set_phase("ocr_first_capture")
    t0 = time.perf_counter()
    m.process_frame(all_cases[0]["frame"])  # the first capture after loading is slower; kept apart from the rest
    first_ms = (time.perf_counter() - t0) * 1000.0

    results = []
    for case in all_cases:
        cv2.imwrite(str(out.parent / f"ocr_input_{case['name']}.png"), case["frame"])
        for path in case["paths"]:
            set_phase(f"ocr_{case['name']}_{path}")
            m.speak_line = (lambda t: spoken.append(time.perf_counter())) if path == "app" else None
            rows, started = [], time.perf_counter()
            while len(rows) < MIN_REPS or (len(rows) < MAX_REPS and time.perf_counter() - started < CASE_BUDGET_S):
                timer.reset()
                spoken.clear()
                t0 = time.perf_counter()
                m.process_frame(case["frame"])
                total = time.perf_counter() - t0
                rows.append({
                    "total_ms": total * 1000.0,
                    "detect_ms": timer.s["detect"] * 1000.0,
                    "recognize_ms": timer.s["recognize"] * 1000.0,
                    "other_ms": (total - timer.s["detect"] - timer.s["recognize"]) * 1000.0,
                    "first_line_ms": (spoken[0] - t0) * 1000.0 if spoken and m.streamed else None,
                    "detect_calls": timer.calls["detect"],
                    "recognize_calls": timer.calls["recognize"],
                })
            text = m._stored_text or ""
            first = [r["first_line_ms"] for r in rows if r["first_line_ms"] is not None]
            res = {
                "case": case["name"], "label": case["label"], "path": path, "reps": len(rows),
                "frame": f"{case['frame'].shape[1]}x{case['frame'].shape[0]}",
                "total_ms_median": med([r["total_ms"] for r in rows]),
                "total_ms_min": round(min(r["total_ms"] for r in rows), 1),
                "total_ms_max": round(max(r["total_ms"] for r in rows), 1),
                "detect_ms_median": med([r["detect_ms"] for r in rows]),
                "recognize_ms_median": med([r["recognize_ms"] for r in rows]),
                "other_ms_median": med([r["other_ms"] for r in rows]),
                "first_line_ms_median": med(first),
                "streamed": bool(m.streamed),
                "detect_calls": rows[-1]["detect_calls"], "recognize_calls": rows[-1]["recognize_calls"],
                "text_boxes": len(getattr(m, "last_boxes", []) or []),
                "truth": case["truth"], "read": text,
                "cer": None if case["truth"] is None else round(cer(case["truth"], text), 4),
                "cpu_temp_c_after": cpu_temp_c(), "throttled_after": throttled(),
                "runs": [{k: (None if v is None else round(v, 1)) for k, v in r.items()} for r in rows],
            }
            if case.get("note"):
                res["note"] = case["note"]
            results.append(res)
            print(f"{case['name']:22s} {path:5s} total {res['total_ms_median']/1000:6.2f} s  detect {res['detect_ms_median']/1000:5.2f} s"
                  f"  recognise {res['recognize_ms_median']/1000:5.2f} s  first line "
                  f"{'-' if res['first_line_ms_median'] is None else round(res['first_line_ms_median']/1000, 2)}  "
                  f"CER {res['cer']}  ({res['reps']} runs, {res['cpu_temp_c_after']} C)", flush=True)
            out.write_text(json.dumps({"partial": True, "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")

    import torch
    payload = {
        "host": host_info(),
        "settings": {
            "OCR_PIPELINE": config.OCR_PIPELINE, "OCR_V2_DECODER": config.OCR_V2_DECODER,
            "OCR_CANVAS_SIZE": config.OCR_CANVAS_SIZE, "OCR_DETECTOR": config.OCR_DETECTOR, "detector_class": detector,
            "OCR_STREAM_PAGES": config.OCR_STREAM_PAGES, "OCR_STREAM_MIN_LINES": config.OCR_STREAM_MIN_LINES,
            "OCR_RECOGNIZER_PATH": config.OCR_RECOGNIZER_PATH, "torch_threads": torch.get_num_threads(),
        },
        "load_ms": round(load_ms, 1),
        "first_capture_after_load_ms": round(first_ms, 1),
        "peak_rss_mb": rss_mb(),
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    set_phase("idle")
    print("wrote", out)


if __name__ == "__main__":
    main()
