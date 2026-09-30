"""Measure the glass on the device it runs on: per-mode latency, memory, temperature.

Run this ON the Raspberry Pi 5 (inside the glass venv) to fill the "Raspberry Pi 5:
NOT MEASURED" gap. It times each mode's real process_frame() on real images, the
Taka detector variants (PT / ONNX / INT8 ONNX), and the jaal check, then records
host, CPU temperature, throttling and peak memory. On any other host the output is
labelled raspberry_pi_5=false so a laptop run cannot be mistaken for a Pi number.

Usage (on the Pi):
  python3 scripts/benchmark_pi5.py                  # latency of every mode, ~5 min
  python3 scripts/benchmark_pi5.py --sustained 30   # + 30-minute run, logs temp/throttle
  python3 scripts/benchmark_pi5.py --iters 5        # quick check

Output: results/pi5_benchmark_<hostname>_<UTC time>.json (+ .csv for --sustained)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WS = ROOT.parent
DATA = WS / "data set"

import cv2  # noqa: E402
import numpy as np  # noqa: E402


# ---------------------------------------------------------------- host info
def pi_model() -> str:
    try:
        return Path("/proc/device-tree/model").read_text(errors="ignore").strip("\x00 \n")
    except OSError:
        return ""


def cpu_temp_c() -> float | None:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except (OSError, ValueError):
        return None


def throttled() -> str | None:
    """vcgencmd get_throttled -> e.g. '0x0'. Non-zero means under-voltage or thermal throttling happened."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5).stdout
        return out.strip().split("=")[-1] or None
    except (OSError, subprocess.SubprocessError):
        return None


def rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KiB on Linux
    except ImportError:
        return None


def host_info() -> dict:
    model = pi_model()
    info = {
        "hostname": socket.gethostname(),
        "device_model": model or platform.processor() or platform.machine(),
        "raspberry_pi_5": "Raspberry Pi 5" in model,
        "machine": platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["cuda_gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    for mod in ("ultralytics", "onnxruntime", "easyocr", "cv2"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    return info


# ---------------------------------------------------------------- inputs
def _first_images(folder: Path, n: int) -> list[np.ndarray]:
    if not folder.is_dir():
        return []
    out = []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            img = cv2.imread(str(p))
            if img is not None:
                out.append(img)
        if len(out) >= n:
            break
    return out


def camera_like(img: np.ndarray) -> np.ndarray:
    """Letterbox any image into a 640x480 frame, the glass camera size."""
    h, w = img.shape[:2]
    r = min(640 / w, 480 / h)
    small = cv2.resize(img, (int(w * r), int(h * r)))
    frame = np.full((480, 640, 3), 90, np.uint8)
    y, x = (480 - small.shape[0]) // 2, (640 - small.shape[1]) // 2
    frame[y:y + small.shape[0], x:x + small.shape[1]] = small
    return frame


def text_frame() -> np.ndarray:
    frame = np.full((480, 640, 3), 235, np.uint8)
    for i, line in enumerate(("EXIT 12", "Bus Stop", "Main Entrance")):
        cv2.putText(frame, line, (60, 150 + 90 * i), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 4)
    return frame


def inputs() -> dict[str, list[np.ndarray]]:
    notes = _first_images(WS / "samples", 4) + _first_images(
        DATA / "Bangladeshi_Paper_Currency_Raw" / "Bangladeshi_Paper_Currency_Raw", 4)
    scenes = _first_images(DATA / "coco2017" / "val2017", 4) or notes
    faces = _first_images(DATA / "RAF-DB" / "DATASET" / "test", 4)
    return {
        "notes": [camera_like(i) for i in notes] or [np.zeros((480, 640, 3), np.uint8)],
        "scenes": [camera_like(i) for i in scenes],
        "faces": [camera_like(i) for i in faces] or [np.zeros((480, 640, 3), np.uint8)],
        "text": [text_frame()],
    }


# ---------------------------------------------------------------- timing
def time_fn(fn, frames: list[np.ndarray], warmup: int, iters: int) -> dict:
    for i in range(warmup):
        fn(frames[i % len(frames)])
    ms = []
    for i in range(iters):
        t0 = time.perf_counter()
        fn(frames[i % len(frames)])
        ms.append((time.perf_counter() - t0) * 1000.0)
    ms.sort()
    med = statistics.median(ms)
    return {
        "median_ms": round(med, 2),
        "p95_ms": round(ms[min(len(ms) - 1, int(0.95 * len(ms)))], 2),
        "mean_ms": round(statistics.fmean(ms), 2),
        "fps_median": round(1000.0 / med, 2) if med > 0 else None,
        "iters": iters,
        "rss_mb_after": rss_mb(),
        "cpu_temp_c_after": cpu_temp_c(),
    }


def build_benchmarks() -> dict:
    """name -> (callable(frame), input key). Each is skipped with a reason if it cannot load."""
    benches, skipped = {}, {}

    def add(name, loader, key):
        try:
            t0 = time.perf_counter()
            fn = loader()
            benches[name] = (fn, key, round((time.perf_counter() - t0) * 1000.0, 1))
        except Exception as exc:  # noqa: BLE001
            skipped[name] = f"{type(exc).__name__}: {exc}"[:300]

    def currency_mode():
        from modes.currency_mode import CurrencyMode
        m = CurrencyMode()
        m.activate()
        if m._yolo is None:
            raise RuntimeError("no currency YOLO weights found")
        return m.process_frame

    def yolo_variant(path: Path):
        def load():
            if not path.is_file():
                raise FileNotFoundError(path)
            from ultralytics import YOLO
            model = YOLO(str(path), task="detect")
            return lambda f: model.predict(f, imgsz=640, conf=0.25, verbose=False)
        return load

    def jaal_only():
        from modes.currency_mode import CurrencyMode
        m = CurrencyMode()
        m._load_auth()
        if m._auth is None:
            raise RuntimeError("no jaal model")
        return lambda f: m._authenticity(f[120:360, 80:560])

    def prmvt_kview(k):
        """PRMVT on k real views of one JaalTaka note (repeats a note photo if the dataset is absent)."""
        def load():
            from PIL import Image
            from modes.currency_mode import CurrencyMode
            m = CurrencyMode()
            m._load_auth()
            if m._auth_kind != "qduig":
                raise RuntimeError("PRMVT (Q-DUIG) checkpoint not found")
            import torch
            note_dirs = sorted((DATA / "JaalTaka" / "real_notes").glob("*"))
            imgs = sorted(note_dirs[0].glob("*.jpg"))[:k] if note_dirs else []
            bgr = [cv2.imread(str(p)) for p in imgs] or [inputs()["notes"][0]] * k
            dev = next(m._auth.parameters()).device
            views = torch.stack([m._auth_tf(Image.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))) for b in bgr[:k]])
            views = views.unsqueeze(0).to(dev)
            mask = torch.ones(1, k, dtype=torch.long, device=dev)

            def run(_frame):
                with torch.inference_mode():
                    out = m._auth(views, mask)["prob"]
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    return out
            return run
        return load

    def object_mode():
        from modes.object_mode import ObjectMode
        m = ObjectMode()
        m.activate()
        if m._model is None:
            raise RuntimeError("no YOLOv8 COCO weights")
        return m.process_frame

    def ocr_mode():
        from modes.ocr_mode import OCRMode
        m = OCRMode()
        m.activate()
        if m._reader is None:
            raise RuntimeError("EasyOCR reader not loaded")
        return m.process_frame

    def emotion():
        from assistive import EmotionDetector
        d = EmotionDetector(download=False)
        if d.face is None:
            raise RuntimeError("no face detector (Haar cascade or YuNet model)")
        return d.predict

    def watermark(learned: bool):
        from modes.watermark_check import WatermarkChecker
        checker = WatermarkChecker()
        if learned and checker.localizer is None:
            raise RuntimeError("no learned localizer (config.WATERMARK_LOCALIZER_PATH)")
        if not learned:
            checker.localizer = None  # template + SIFT registration path

        def run(frame):
            prob = checker.genuine_prob(frame, "500_taka")
            if prob is None:
                prob = checker.genuine_prob(frame, "1000_taka")
            return checker.sentence(prob)

        return run

    models = WS / "realtime_bangla_taka_detection" / "models"
    add("currency_mode_full", currency_mode, "notes")
    add("taka_yolo_pt", yolo_variant(models / "best.pt"), "notes")
    add("taka_yolo_onnx_fp32", yolo_variant(models / "best.onnx"), "notes")
    add("taka_yolo_onnx_int8", yolo_variant(models / "best_int8.onnx"), "notes")
    add("watermark_int8_sift", lambda: watermark(False), "notes")
    add("watermark_int8_localizer", lambda: watermark(True), "notes")
    add("jaal_check_qduig", jaal_only, "notes")
    for k in range(1, 7):
        add(f"prmvt_{k}view", prmvt_kview(k), "notes")
    add("object_mode", object_mode, "scenes")
    add("ocr_mode", ocr_mode, "text")
    add("emotion", emotion, "faces")
    return benches, skipped


def sustained(benches: dict, frames: dict, minutes: float, csv_path: Path) -> dict:
    """Cycle through every mode for `minutes`, logging latency, temperature and throttling."""
    end = time.time() + minutes * 60
    rows, i = [], 0
    names = list(benches)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["elapsed_s", "mode", "latency_ms", "cpu_temp_c", "throttled", "rss_mb"])
        start = time.time()
        while time.time() < end:
            name = names[i % len(names)]
            fn, key, _ = benches[name]
            f = frames[key][i % len(frames[key])]
            t0 = time.perf_counter()
            fn(f)
            lat = (time.perf_counter() - t0) * 1000.0
            row = [round(time.time() - start, 1), name, round(lat, 2), cpu_temp_c(), throttled(), rss_mb()]
            w.writerow(row)
            rows.append(row)
            i += 1
    temps = [r[3] for r in rows if r[3] is not None]
    return {
        "minutes": minutes,
        "calls": len(rows),
        "max_cpu_temp_c": max(temps) if temps else None,
        "final_throttled": throttled(),
        "csv": csv_path.name,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--sustained", type=float, default=0.0, help="minutes of continuous mixed-mode running")
    p.add_argument("--only", nargs="*", default=None, help="subset of benchmark names")
    args = p.parse_args()

    info = host_info()
    if not info["raspberry_pi_5"]:
        print(f"NOTE: host is '{info['device_model']}', not a Raspberry Pi 5. "
              "Results are labelled raspberry_pi_5=false.", flush=True)
    info["cpu_temp_c_start"] = cpu_temp_c()
    info["throttled_start"] = throttled()

    frames = inputs()
    benches, skipped = build_benchmarks()
    if args.only:
        benches = {k: v for k, v in benches.items() if k in args.only}
    results = {}
    for name, (fn, key, load_ms) in benches.items():
        print(f"timing {name} on {len(frames[key])} {key} frame(s) ...", flush=True)
        r = time_fn(fn, frames[key], args.warmup, args.iters)
        r["load_ms"] = load_ms
        results[name] = r
        print(f"  median {r['median_ms']} ms  p95 {r['p95_ms']} ms  ({r['fps_median']} FPS)", flush=True)
    for name, why in skipped.items():
        print(f"  skipped {name}: {why}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    base = out_dir / f"pi5_benchmark_{info['hostname']}_{stamp}"
    payload = {"host": info, "latency": results, "skipped": skipped, "peak_rss_mb": rss_mb()}
    if args.sustained > 0 and benches:
        print(f"sustained run for {args.sustained} min ...", flush=True)
        payload["sustained"] = sustained(benches, frames, args.sustained, base.with_suffix(".csv"))
    payload["host"]["cpu_temp_c_end"] = cpu_temp_c()
    payload["host"]["throttled_end"] = throttled()
    base.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {base.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
