"""Does the counterfeit safe-check slow down right after an ONNX Runtime model has run?

    python safecheck_after_onnx.py <out.json>

In the 30-minute run of benchmark_pi5.py the safe-check (PyTorch) always follows the watermark localizer (ONNX Runtime)
and takes about 84 ms, against 37 ms when it runs 100 times in a row. In the app it follows the ONNX note detector.
This times the safe-check, on the crop benchmark_pi5.py uses, in four orders:
  alone                          100 safe-checks in a row
  after_detector                 hand-held note detector (ONNX INT8), then the timed safe-check: the app's order
  after_detector_pause_50ms      the same with a 50 ms pause between the two
  after_pytorch_detector         the PyTorch note detector (best.pt) first, to see whether any model before it costs
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "savior_glass" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "savior_glass"))
import benchmark_pi5 as B  # noqa: E402

N = 60
notes = B.inputs()["notes"]
from modes.currency_mode import CurrencyMode  # noqa: E402
from ultralytics import YOLO  # noqa: E402

m = CurrencyMode()
m._load_auth()
models = B.WS / "realtime_bangla_taka_detection" / "models"
det = YOLO(str(models / "best_handheld_int8.onnx"), task="detect")
pt = YOLO(str(models / "best.pt"), task="detect")


def safe(f):
    return m._authenticity(f[120:360, 80:560])


def run(before, pause_s=0.0):
    ms = []
    for i in range(N + 3):
        f = notes[i % len(notes)]
        if before is not None:
            before.predict(f, imgsz=640, conf=0.25, verbose=False)
        if pause_s:
            time.sleep(pause_s)
        t0 = time.perf_counter()
        safe(f)
        if i >= 3:
            ms.append((time.perf_counter() - t0) * 1000)
    ms.sort()
    return {"n": len(ms), "median_ms": round(statistics.median(ms), 1), "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 1),
            "temp_c_after": B.cpu_temp_c(), "throttled_after": B.throttled()}


out = {"temp_c_start": B.cpu_temp_c(), "throttled_start": B.throttled()}
for name, before, pause in [("alone", None, 0), ("after_detector", det, 0), ("after_detector_pause_50ms", det, 0.05),
                            ("after_pytorch_detector", pt, 0), ("alone_again", None, 0)]:
    out[name] = run(before, pause)
    print(name, out[name], flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
