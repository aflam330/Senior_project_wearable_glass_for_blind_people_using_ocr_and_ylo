"""Is this picture taken under the UV lamp? The rule behind config.UV_FRAME_BG_RATIO.

Under a UV lamp the camera sees almost only blue: the frame's mean blue is several times its mean green. Ordinary
note photos are not like that. The threshold is set on two sets only, then read once on two others:

  set on:   ordinary photos   data set/currency_yolo_data/valid  (the highest blue/green among them)
            UV photos          data set/uv_notes, split "calibrate" in boxes.csv, each as the whole photo and as a
                               webcam-style frame (turned landscape, centre cut to 4:3, 640x480)  (the lowest blue/green)
            threshold = midway between the two
  read on:  ordinary photos   data set/currency_yolo_data/test   (how many would be taken for UV pictures)
            UV photos          data set/uv_notes, split "test", in both forms  (how many are recognised)

Version 2 (29 October 2026). Version 1 used whole UV photos only (threshold 2.181); as a webcam-style frame one
calibration photo (uv_500_024) then fell below it (2.170), so the frames were added to the calibration side. Among the
colour features compared on the calibration sets only (blue over green, over the larger of green and red, over their
geometric and arithmetic mean, over red), blue over green kept the widest margin.
The UV photos are phone photographs; the glass's C270 webcam was not measured.

    python scripts/uv_frame_rule.py      ->  results/uv_check/uv_frame_rule.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from modes.currency_mode import blue_green_ratio  # noqa: E402

ROOT = HERE.parent.parent
DATA = ROOT / "data set"
OUT = HERE.parent / "results" / "uv_check" / "uv_frame_rule.json"


def read(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def camera_frame(im: np.ndarray) -> np.ndarray:
    """The photo turned landscape, its centre cut to 4:3 and scaled to 640x480, as a webcam would frame it
    (no padding, which would change the colour and brightness of the frame)."""
    if im.shape[0] > im.shape[1]:
        im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    h, w = im.shape[:2]
    if w * 3 > h * 4:
        cw = h * 4 // 3
        im = im[:, (w - cw) // 2:(w - cw) // 2 + cw]
    else:
        ch = w * 3 // 4
        im = im[(h - ch) // 2:(h - ch) // 2 + ch]
    return cv2.resize(im, (640, 480), interpolation=cv2.INTER_AREA)


def ratios(paths: list[Path], as_frame: bool = False) -> list[float]:
    out = []
    for p in paths:
        im = read(p)
        if im is not None:
            out.append(blue_green_ratio(camera_frame(im) if as_frame else im))
    return out


def main() -> None:
    split = {r["file"]: r["split"] for r in csv.DictReader(open(DATA / "uv_notes" / "boxes.csv", encoding="utf-8"))}
    uv = sorted(p for p in (DATA / "uv_notes").glob("*/*.png"))
    cal = [p for p in uv if split.get(p.name) == "calibrate"]
    test = [p for p in uv if split.get(p.name) == "test"]
    uv_cal = ratios(cal) + ratios(cal, as_frame=True)
    normal_val = ratios(sorted((DATA / "currency_yolo_data" / "valid" / "images").glob("*")))
    thr = round((max(normal_val) + min(uv_cal)) / 2, 3)
    uv_test, uv_test_frames = ratios(test), ratios(test, as_frame=True)
    normal_test = ratios(sorted((DATA / "currency_yolo_data" / "test" / "images").glob("*")))
    res = {
        "version": 2,
        "rule": "mean blue / mean green of the frame > threshold (and mean blue > 20)",
        "threshold": thr,
        "set_on": {"ordinary_valid_n": len(normal_val), "ordinary_valid_max": round(max(normal_val), 4),
                   "uv_calibrate_photos": len(cal), "uv_calibrate_min_photo_or_frame": round(min(uv_cal), 4)},
        "test": {"uv_test_photos": len(test),
                 "recognised_whole_photo": int(sum(r > thr for r in uv_test)),
                 "recognised_webcam_frame": int(sum(r > thr for r in uv_test_frames)),
                 "uv_test_min": round(min(uv_test + uv_test_frames), 4),
                 "ordinary_test_n": len(normal_test), "ordinary_test_taken_for_uv": int(sum(r > thr for r in normal_test)),
                 "ordinary_test_max": round(max(normal_test), 4)},
        "version_1": {"threshold": 2.181, "set_on": "whole UV calibration photos only",
                      "why_replaced": "one calibration photo as a webcam-style frame fell below it (uv_500_024, 2.170)"},
        "not_measured": "the glass's C270 webcam under the lamp; a room with other lights on",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
