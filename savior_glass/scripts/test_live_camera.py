"""Live-camera loop test: real webcam frames through the glass's own modes.

Runs for --seconds on camera --index: every frame goes through CurrencyMode.detect_live
(Taka YOLO + jaal check), ObjectMode.process_frame, and every 5th frame through the
emotion detector. Records capture rate, per-stage latency and what was announced.
No frame, image or audio is saved (the camera may see people). Writes
results/live_camera_test.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def stats(ms):
    if not ms:
        return None
    ms = sorted(ms)
    return {"n": len(ms), "median_ms": round(statistics.median(ms), 1), "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=20)
    args = ap.parse_args()
    import torch
    from assistive import EmotionDetector
    from modes.currency_mode import CurrencyMode
    from modes.object_mode import ObjectMode

    cap = cv2.VideoCapture(args.index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        out = {"status": "NOT_MEASURED", "reason": f"camera {args.index} did not open"}
        (ROOT / "results/live_camera_test.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(out)
        return
    cur, obj = CurrencyMode(), ObjectMode()
    cur.activate()
    obj.activate()
    emo = EmotionDetector(download=False)
    t_end = time.time() + 2.0
    while time.time() < t_end:  # auto-exposure warm-up
        cap.read()

    lat = {"capture": [], "currency_detect_and_jaal": [], "object_mode": [], "emotion": [], "frame_total": []}
    brightness, notes, objects, emotions = [], {}, {}, {}
    frames = 0
    t_start = time.time()
    while time.time() - t_start < args.seconds:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            continue
        frames += 1
        brightness.append(float(frame.mean()))
        hits = cur.detect_live(frame)
        t2 = time.perf_counter()
        said = obj.process_frame(frame)
        t3 = time.perf_counter()
        for h in hits:
            notes[h["name"]] = notes.get(h["name"], 0) + 1
        if said:
            objects[said] = objects.get(said, 0) + 1
        if frames % 5 == 0:
            e = emo.predict(frame)
            lat["emotion"].append((time.perf_counter() - t3) * 1000)
            emotions[e["backend"]] = emotions.get(e["backend"], 0) + 1
        lat["capture"].append((t1 - t0) * 1000)
        lat["currency_detect_and_jaal"].append((t2 - t1) * 1000)
        lat["object_mode"].append((t3 - t2) * 1000)
        lat["frame_total"].append((time.perf_counter() - t0) * 1000)
    elapsed = time.time() - t_start
    cap.release()
    out = {
        "status": "ok",
        "camera": {"index": args.index, "resolution": list(frame.shape[:2][::-1]) if frames else None,
                   "mean_brightness_0_255": round(statistics.fmean(brightness), 1) if brightness else None},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seconds": round(elapsed, 1), "frames_processed": frames, "loop_fps": round(frames / elapsed, 2),
        "latency": {k: stats(v) for k, v in lat.items()},
        "notes_detected": notes, "object_announcements": objects, "emotion_backend_counts": emotions,
        "saved_media": "none (privacy)",
        "note_accuracy_on_live_notes": "NOT_MEASURED (no note was held in front of the camera during an unattended run)",
    }
    (ROOT / "results/live_camera_test.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
