"""Headless webcam diagnostic v2: longer warm-up, brightness stats,
and a brightness-corrected pass through the model.
"""

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
CAMERA_INDEX = 0
NUM_SHOTS = 3
INTERVAL_SEC = 5
CONF_THRESHOLD = 0.05
OUT_DIR = Path("results/webcam_diag2")


def brighten(frame, gain=3.0, gamma=0.5):
    f = frame.astype(np.float32) / 255.0
    f = np.clip(f * gain, 0, 1)
    f = np.power(f, gamma)
    return (f * 255).astype(np.uint8)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    print("Warming up camera (60 frames)...", flush=True)
    for _ in range(60):
        cap.read()
        time.sleep(0.02)

    for i in range(NUM_SHOTS):
        print(f"\n--- Shot {i+1}/{NUM_SHOTS} (waiting {INTERVAL_SEC}s) ---", flush=True)
        time.sleep(INTERVAL_SEC)

        ret, frame = cap.read()
        if not ret:
            print("  failed to read frame")
            continue

        mean_brightness = frame.mean()
        print(f"  mean brightness: {mean_brightness:.2f} / 255")

        raw_path = OUT_DIR / f"shot_{i+1}_raw.jpg"
        cv2.imwrite(str(raw_path), frame)

        bright = brighten(frame)
        bright_path = OUT_DIR / f"shot_{i+1}_brightened.jpg"
        cv2.imwrite(str(bright_path), bright)

        results = model.predict(bright, conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            print("  (brightened) No detections above conf=0.05")
        else:
            for b in boxes:
                cls_id = int(b.cls.item())
                conf = b.conf.item()
                print(f"  (brightened) {model.names[cls_id]}: conf={conf:.3f}")

        annotated = results[0].plot()
        ann_path = OUT_DIR / f"shot_{i+1}_brightened_annotated.jpg"
        cv2.imwrite(str(ann_path), annotated)

    cap.release()
    print(f"\nSaved shots to {OUT_DIR}/")


if __name__ == "__main__":
    main()
