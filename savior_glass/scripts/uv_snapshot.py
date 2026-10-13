"""Take one picture under the UV lamp, mark the fibres the check finds, and save it.

    python scripts/uv_snapshot.py                 # picture from the camera
    python scripts/uv_snapshot.py photo.jpg       # an existing picture
    python scripts/uv_snapshot.py --whole         # judge the whole picture (do not look for the note box first)

Stop the glass (main.py) first: only one program can use the camera.
Set UV_LIGHT_PIN=<pin> if a Pi pin switches the lamp; otherwise switch the lamp on by hand before running.
Output: results/uv_snapshots/uv_<time>.jpg (left: the picture with the note box, right: fibres in green) and a line
with the result. The check has not been tested on a counterfeit note; "not found" means "check by hand".
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config  # noqa: E402
from modes.uv_check import UVChecker, UVLamp, fibre_mask, inner  # noqa: E402

OUT = HERE.parent / "results" / "uv_snapshots"


def grab():
    import utils
    cam = utils.CameraManager()
    cam.start()
    lamp = UVLamp()
    lamp.set(True)
    try:
        frame, t0 = None, time.time()
        time.sleep(1.0)                                   # exposure settles on the UV picture
        while time.time() - t0 < 8:
            frame = cam.get_frame()
            if frame is not None:
                break
            time.sleep(0.1)
    finally:
        lamp.set(False)
        cam.stop()
    return frame


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    whole = "--whole" in sys.argv
    frame = cv2.imdecode(np.fromfile(args[0], np.uint8), cv2.IMREAD_COLOR) if args else grab()
    if frame is None:
        sys.exit("No picture: is the camera free (stop main.py) and connected?")
    crop, box = frame, None
    if not whole:
        try:
            from ultralytics import YOLO
            # config picks the INT8 graph on the Pi and the PyTorch weights on a laptop
            r = YOLO(config.CURRENCY_YOLO_PATH, task="detect").predict(frame, conf=0.10, verbose=False, imgsz=640)[0]
            if r.boxes is not None and len(r.boxes):
                i = int(r.boxes.conf.argmax())
                x1, y1, x2, y2 = [int(v) for v in r.boxes.xyxy[i].tolist()]
                box, crop = (x1, y1, x2, y2), frame[max(0, y1):y2, max(0, x1):x2]
        except Exception as exc:
            print("note detector not used:", exc)
    part = inner(crop) if box else crop
    res = UVChecker().check(part)
    mask, n, ab = fibre_mask(part)
    left = frame.copy()
    if box:
        cv2.rectangle(left, box[:2], box[2:], (0, 255, 255), max(2, frame.shape[1] // 300))
    if mask is not None:
        right = cv2.resize(part, (mask.shape[1], mask.shape[0])).astype(np.float32)
        right = (np.clip(right / max(np.percentile(right, 99.8), 1) * 255, 0, 255) * 0.6).astype(np.uint8)
        right[cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0] = (60, 255, 60)
    else:
        right = np.zeros((300, 600, 3), np.uint8)
    h = 480
    a = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
    b = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
    canvas = np.hstack([a, np.full((h, 8, 3), 255, np.uint8), b])
    said = {"uv_marks_found": "UV MARKS FOUND: likely genuine", "uv_marks_not_found": "UV marks NOT found: check by hand",
            "too_dark": "too dark: hold the note closer to the lamp", "no_note": "no note found"}[res["outcome"]]
    cv2.putText(canvas, f"{said}  (fibres {res.get('fibres', 0)}, need {config.UV_MIN_FIBRES})", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"uv_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"{said} | fibres {res.get('fibres', 0)} | note box {'found' if box else 'not used'} | saved {path}")


if __name__ == "__main__":
    main()
