"""Show which cameras this machine has and which one the glass will use. Run on the Pi:

    python scripts/check_camera.py

It lists ribbon (CSI) cameras seen by picamera2, the /dev/video* devices, and which OpenCV indices deliver frames,
then starts the app's own CameraManager and saves one frame to results/camera_check.jpg.
"""
from __future__ import annotations

import glob
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2  # noqa: E402

import config  # noqa: E402
from utils import CameraManager  # noqa: E402


def main() -> None:
    print("CAMERA_BACKEND =", config.CAMERA_BACKEND, "| CAMERA_INDEX =", config.CAMERA_INDEX)
    try:
        from picamera2 import Picamera2
        info = Picamera2.global_camera_info()
        print(f"picamera2: {len(info)} ribbon camera(s)", [c.get("Model") for c in info])
    except Exception as exc:  # noqa: BLE001
        print("picamera2: not usable ->", type(exc).__name__, str(exc)[:160])
    nodes = sorted(glob.glob("/dev/video*"), key=lambda p: int(p.rsplit("video", 1)[1]) if p.rsplit("video", 1)[1].isdigit() else 999)
    print("video devices:", nodes or "none listed (not Linux?)")
    cm = CameraManager()
    for idx in cm._candidate_indices():
        cap = cm._open_cv(idx)
        print(f"opencv index {idx}:", "frames OK" if cap is not None else "no frames")
        if cap is not None:
            cap.release()
    print("starting the app camera ...")
    try:
        cm.start()
    except RuntimeError as exc:
        sys.exit(f"FAILED: {exc}")
    time.sleep(0.5)
    frame = cm.get_frame()
    cm.stop()
    out = os.path.join(config.BASE_DIR, "results", "camera_check.jpg")
    cv2.imwrite(out, frame)
    print(f"OK: frame {frame.shape[1]}x{frame.shape[0]}, mean brightness {frame.mean():.0f}, saved to {out}")


if __name__ == "__main__":
    main()
