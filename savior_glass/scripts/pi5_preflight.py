"""Check that the Pi 5 glass has the measured models, without timing them.

Run from savior_glass/:

    python scripts/pi5_preflight.py

Exits 0 when the INT8 watermark, watermark localizer, INT8 note detector, safe-check checkpoint and
both denomination templates are present and the two ONNX graphs run one dummy
input. A pass on a laptop does not produce a Pi latency number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT.parent
MODELS = WS / "realtime_bangla_taka_detection" / "models"
TEMPLATES = WS / "data set" / "Bangladeshi_Paper_Currency_Raw" / "Bangladeshi_Paper_Currency_Raw"

REQUIRED = {
    "watermark INT8": MODELS / "watermark_mobilenetv2_int8.onnx",
    "watermark localizer": MODELS / "watermark_localizer_seed42.onnx",
    "note detector INT8": MODELS / "best_int8.onnx",
    "safe-check checkpoint": WS / "realtime_bangla_taka_detection" / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt",
    "500 template": TEMPLATES / "500" / "500 Taka_0001.jpg",
    "1000 template": TEMPLATES / "1000" / "1000  Taka_001.jpg",
}


def main() -> int:
    missing = [name for name, path in REQUIRED.items() if not path.is_file()]
    for name, path in REQUIRED.items():
        if path.is_file():
            print(f"ok  {name}  {path.stat().st_size / 1e6:.2f} MB")
        else:
            print(f"MISSING  {name}  {path}")
    if missing:
        print("Pi bundle is incomplete.")
        return 1
    try:
        import onnxruntime as ort
    except ImportError:
        print("MISSING  onnxruntime  (pip install -r requirements.txt)")
        return 1
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    wm = ort.InferenceSession(str(REQUIRED["watermark INT8"]), sess_options=opts, providers=["CPUExecutionProvider"])
    det = ort.InferenceSession(str(REQUIRED["note detector INT8"]), sess_options=opts, providers=["CPUExecutionProvider"])
    wm.run(None, {"image": np.zeros((1, 3, 224, 224), np.float32)})
    loc = ort.InferenceSession(str(REQUIRED["watermark localizer"]), sess_options=opts, providers=["CPUExecutionProvider"])
    corners = loc.run(None, {"image": np.zeros((1, 3, 320, 320), np.float32)})[0]
    if corners.shape != (1, 8):
        print("BAD  watermark localizer output", corners.shape)
        return 1
    det_in = det.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in det_in.shape]
    if shape[2] in (0, None) or not isinstance(det_in.shape[2], int):
        shape = [1, 3, 640, 640]
    det.run(None, {det_in.name: np.zeros(shape, np.float32)})
    print("ok  INT8 watermark, watermark localizer and INT8 detector graphs each ran one dummy input")
    import subprocess
    if subprocess.run([sys.executable, str(ROOT / "scripts" / "test_capture_guide.py")], env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}).returncode:
        print("BAD  capture guide self-test")
        return 1
    print("Latency is not measured here. On the Pi: python scripts/benchmark_pi5.py --iters 100 --sustained 30")
    return 0


if __name__ == "__main__":
    sys.exit(main())
