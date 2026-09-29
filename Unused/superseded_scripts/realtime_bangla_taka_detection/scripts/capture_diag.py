"""Headless webcam diagnostic (no GUI window).

Captures several frames a few seconds apart, runs the model on each at a
very low confidence threshold, prints ALL detections (not just the best),
and saves raw + annotated images for visual review.
"""

import time
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
CAMERA_INDEX = 0
NUM_SHOTS = 6
INTERVAL_SEC = 4
CONF_THRESHOLD = 0.05
OUT_DIR = Path("results/webcam_diag")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    # let the camera warm up / auto-exposure settle
    for _ in range(10):
        cap.read()

    for i in range(NUM_SHOTS):
        print(f"\n--- Shot {i+1}/{NUM_SHOTS} (waiting {INTERVAL_SEC}s) ---", flush=True)
        time.sleep(INTERVAL_SEC)

        ret, frame = cap.read()
        if not ret:
            print("  failed to read frame")
            continue

        raw_path = OUT_DIR / f"shot_{i+1}_raw.jpg"
        cv2.imwrite(str(raw_path), frame)

        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            print("  No detections above conf=0.05")
        else:
            for b in boxes:
                cls_id = int(b.cls.item())
                conf = b.conf.item()
                print(f"  {model.names[cls_id]}: conf={conf:.3f}")

        annotated = results[0].plot()
        ann_path = OUT_DIR / f"shot_{i+1}_annotated.jpg"
        cv2.imwrite(str(ann_path), annotated)

    cap.release()
    print(f"\nSaved {NUM_SHOTS} raw + annotated shots to {OUT_DIR}/")


if __name__ == "__main__":
    main()
