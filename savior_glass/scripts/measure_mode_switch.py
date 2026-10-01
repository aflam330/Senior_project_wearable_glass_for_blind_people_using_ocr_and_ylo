"""Time switching into currency mode and the first currency check after it (no camera, no GPIO).

Measures, on whatever machine runs it:
  activate_first_ms   first switch into currency mode (models load)
  activate_again_ms   a later switch back into currency mode
  warm_up_ms          CurrencyMode.warm_up(), when the code has it
  first_check_ms      the first process_frame after activation (+ warm-up, if any)
  steady_check_ms     median of the next 10 process_frame calls
Frames are BanglaTaka 500 / 1000 photos. The label in the output says which host ran it; a laptop
number is not a Pi number.
Usage: python scripts/measure_mode_switch.py [out.json]
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from modes.currency_mode import CurrencyMode  # noqa: E402

DATA = ROOT.parent / "data set" / "Bangladeshi_Paper_Currency_Raw" / "Bangladeshi_Paper_Currency_Raw"


def frames(n=12):
    paths = sorted((DATA / "500").glob("*.jpg"))[:n // 2] + sorted((DATA / "1000").glob("*.jpg"))[:n // 2]
    out = []
    for p in paths:
        im = cv2.imread(str(p))
        out.append(cv2.resize(im, (640, round(im.shape[0] * 640 / im.shape[1]))))
    return out


def ms(t0):
    return round((time.perf_counter() - t0) * 1000, 1)


def main() -> None:
    fr = frames()
    m = CurrencyMode()
    res = {"host": platform.node(), "raspberry_pi_5": config.ON_RASPBERRY_PI_5, "python": platform.python_version(),
           "watermark_check": bool(getattr(config, "WATERMARK_CHECK_ENABLED", False))}
    t0 = time.perf_counter(); m.activate(); res["activate_first_ms"] = ms(t0)
    if hasattr(m, "warm_up"):
        t0 = time.perf_counter(); m.warm_up(); res["warm_up_ms"] = ms(t0)
    t0 = time.perf_counter(); m.process_frame(fr[0]); res["first_check_ms"] = ms(t0)
    steady = []
    for f in fr[1:11]:
        t0 = time.perf_counter(); m.process_frame(f); steady.append((time.perf_counter() - t0) * 1000)
    res["steady_check_ms"] = round(statistics.median(steady), 1)
    m.deactivate()
    t0 = time.perf_counter(); m.activate(); res["activate_again_ms"] = ms(t0)
    print(json.dumps(res, indent=1))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
