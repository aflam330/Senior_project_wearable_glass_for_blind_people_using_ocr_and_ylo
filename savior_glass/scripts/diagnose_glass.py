"""Find out WHY currency or text reading fails on the glass: capture the scene and run the app's own code on it.

Run on the Pi (stop main.py first, only one program can use the camera):

    python scripts/diagnose_glass.py currency     # hold a note in front of the camera
    python scripts/diagnose_glass.py ocr          # hold printed Bangla / English text in front of the camera
    python scripts/diagnose_glass.py ocr --shots 5

For each shot it takes ONE picture at the camera's higher resolution (1640 x 1232 on the Pi Camera v2), makes the
640 x 480 version the app uses today from the same picture, and runs the app's currency detector or OCR on both.
It prints, per shot: sharpness (variance of the Laplacian; low = blurred), brightness, and what each resolution gives.
Everything is saved in results/diagnostics/<time>/ (the pictures, and report.json) so the result can be looked at
afterwards. If the high-resolution column is right where the 640 column is wrong, the fix is to capture stills at
high resolution; if both are blurred (low sharpness), the lens focus or the distance is the problem.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402

HI = (1640, 1232)


def sharpness(bgr) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (640, int(g.shape[0] * 640 / g.shape[1])), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


class Cam:
    """High-resolution frames from the Pi Camera (picamera2), or from OpenCV on other machines."""

    def __init__(self):
        self.pc, self.cap = None, None
        try:
            from picamera2 import Picamera2
            if Picamera2.global_camera_info():
                self.pc = Picamera2()
                self.pc.configure(self.pc.create_still_configuration(main={"size": HI, "format": "RGB888"}))
                self.pc.start()
                time.sleep(1.0)
                return
        except Exception as exc:  # noqa: BLE001
            print("picamera2 not used:", str(exc)[:120])
        self.cap = cv2.VideoCapture(int(os.environ.get("CAMERA_INDEX", "0")))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        for _ in range(8):
            self.cap.read()

    def grab(self):
        if self.pc is not None:
            return self.pc.capture_array()
        ok, f = self.cap.read()
        return f if ok else None

    def close(self):
        if self.pc is not None:
            self.pc.stop()
            self.pc.close()
        if self.cap is not None:
            self.cap.release()


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "currency"
    shots = int(sys.argv[sys.argv.index("--shots") + 1]) if "--shots" in sys.argv else 3
    out = os.path.join(config.BASE_DIR, "results", "diagnostics", dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + what)
    os.makedirs(out, exist_ok=True)
    if what == "currency":
        from modes.currency_mode import CurrencyMode
        mode = CurrencyMode()
        mode.activate()

        def run(frame):
            hits = mode.detect_live(frame)
            return [{"name": h["name"], "conf": round(h["conf"], 3), "auth": h.get("auth")} for h in hits[:3]]
    else:
        from modes.ocr_mode import OCRMode
        from ocr_text_region import clean_text
        mode = OCRMode()
        mode.activate()

        def run(frame):
            t0 = time.perf_counter()
            text = clean_text(" ".join(mode._read_v2(frame)))
            return {"text": text, "seconds": round(time.perf_counter() - t0, 1)}
    cam = Cam()
    report = {"what": what, "device": open("/proc/device-tree/model").read().strip("\x00") if os.path.exists("/proc/device-tree/model") else "not a Pi",
              "app_resolution": [config.CAMERA_WIDTH, config.CAMERA_HEIGHT], "shots": []}
    try:
        for i in range(1, shots + 1):
            print(f"\nShot {i} of {shots}: hold the {'note' if what == 'currency' else 'text'} steady in front of the camera ... ", end="", flush=True)
            for s in (3, 2, 1):
                print(s, end=" ", flush=True)
                time.sleep(1)
            hi = cam.grab()
            if hi is None:
                sys.exit("\nno frame from the camera")
            lo = cv2.resize(hi, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out, f"shot{i}_high.jpg"), hi, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(os.path.join(out, f"shot{i}_app640.jpg"), lo, [cv2.IMWRITE_JPEG_QUALITY, 90])
            row = {"shot": i, "high_size": [int(hi.shape[1]), int(hi.shape[0])], "sharpness": round(sharpness(hi), 1),
                   "brightness": round(float(hi.mean()), 1), "app_640": run(lo), "high_res": run(hi)}
            report["shots"].append(row)
            print(f"\n  sharpness {row['sharpness']} (under about 100 = blurred), brightness {row['brightness']}")
            print("  app today (640x480):", json.dumps(row["app_640"], ensure_ascii=False))
            print(f"  high resolution ({hi.shape[1]}x{hi.shape[0]}):", json.dumps(row["high_res"], ensure_ascii=False))
    finally:
        cam.close()
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("\nsaved to", out)


if __name__ == "__main__":
    main()
