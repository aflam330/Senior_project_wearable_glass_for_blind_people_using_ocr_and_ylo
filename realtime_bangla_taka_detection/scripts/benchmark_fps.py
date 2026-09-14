"""Time YOLO FP32 / ONNX / INT8 FPS and record file sizes (Pi + laptop)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fps_benchmark.json"
IMGSZ = 640
WARMUP = 8
ITERS = 40


def _dummy():
    return np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)


def bench_ultralytics(path: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(path))
    img = _dummy()
    for _ in range(WARMUP):
        model.predict(img, imgsz=IMGSZ, verbose=False)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        model.predict(img, imgsz=IMGSZ, verbose=False)
    dt = (time.perf_counter() - t0) / ITERS
    return {
        "path": str(path),
        "bytes": path.stat().st_size if path.is_file() else None,
        "ms": dt * 1000,
        "fps": 1.0 / dt,
    }


def bench_onnx(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        import onnxruntime as ort
    except Exception:
        return None
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    # NCHW float
    x = np.random.rand(1, 3, IMGSZ, IMGSZ).astype(np.float32)
    name = inp.name
    for _ in range(WARMUP):
        sess.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(ITERS):
        sess.run(None, {name: x})
    dt = (time.perf_counter() - t0) / ITERS
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "mb": path.stat().st_size / 1e6,
        "ms": dt * 1000,
        "fps": 1.0 / dt,
        "runtime": "onnxruntime-cpu",
    }


def main():
    models = ROOT / "models"
    report = {"imgsz": IMGSZ, "iters": ITERS, "hardware": "CPU"}
    for label, name in [
        ("pt", "best.pt"),
        ("onnx", "best.onnx"),
        ("int8_onnx", "best_int8.onnx"),
        ("tflite", "best_int8.tflite"),
        ("tflite2", "best.tflite"),
    ]:
        path = models / name
        if not path.is_file():
            # ultralytics sometimes writes best_saved_model/*.tflite
            alts = list(models.glob(f"**/{name}")) + list(models.glob("**/*.tflite"))
            path = alts[0] if alts else path
        if not path.is_file():
            continue
        print("bench", path)
        if path.suffix == ".onnx":
            report[label] = bench_onnx(path) or bench_ultralytics(path)
        else:
            try:
                report[label] = bench_ultralytics(path)
            except Exception as exc:
                report[label] = {"error": str(exc), "bytes": path.stat().st_size, "mb": path.stat().st_size / 1e6}
        if path.is_file():
            report.setdefault(label, {})
            if isinstance(report[label], dict):
                report[label]["mb"] = path.stat().st_size / 1e6
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
