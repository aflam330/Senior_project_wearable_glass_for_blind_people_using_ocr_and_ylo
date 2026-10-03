"""Run the live algorithm panel once on a fixed picture (no camera, no window) and save what it would show.

The picture is a real 500 Taka photo (BanglaTaka) with a RAF-DB test face beside it, so every algorithm has
something to work on. Checks that each one ran and reported a time. Output: results/live_diag_check.png + .json
Run: python scripts/test_live_diag.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("FIELD_LOG_ENABLED", "0")

import config  # noqa: E402
import main as app_main  # noqa: E402
import preview  # noqa: E402

WS = os.path.abspath(os.path.join(ROOT, ".."))


def scene():
    note = cv2.imread(os.path.join(WS, "data set", "Bangladeshi_Paper_Currency_Raw", "Bangladeshi_Paper_Currency_Raw", "500", "500 Taka_0001.jpg"))
    canvas = np.full((720, 1280, 3), 90, np.uint8)
    nw = 760
    note = cv2.resize(note, (nw, int(note.shape[0] * nw / note.shape[1])))
    canvas[60:60 + note.shape[0], 40:40 + nw] = note[: 720 - 60]
    faces = sorted(glob.glob(os.path.join(WS, "data set", "RAF-DB", "DATASET", "test", "4", "*.jpg")))
    if faces:
        face = cv2.resize(cv2.imread(faces[0]), (300, 300))
        canvas[380:680, 900:1200] = face
    return canvas


def main() -> None:
    img = scene()
    app = app_main.SmartGlass()
    app._camera.get_frame = lambda: img.copy()
    view = preview.Preview(app)
    t0 = time.time()
    with app._model_lock:
        state = view._diag._run(img)          # loads the models on first use
    load_s = time.time() - t0
    with app._model_lock:
        state = view._diag._run(img)          # second pass: steady-state times
    view._diag.state, view._diag.enabled = state, True
    frame = img.copy()
    view._draw_diag(frame)
    out = np.hstack([frame, view._diag_panel(frame.shape[0])])
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    cv2.imwrite(os.path.join(ROOT, "results", "live_diag_check.png"), out)
    slim = json.loads(json.dumps(state, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    slim["first_pass_seconds_including_model_loading"] = round(load_s, 1)
    with open(os.path.join(ROOT, "results", "live_diag_check.json"), "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=1)
    ran = {k: ("ms" in (state.get(k) or {})) for k in ("detector", "safe", "guide", "watermark", "objects", "emotion")}
    print("ran:", ran)
    for k in ("detector", "safe", "guide", "watermark", "objects", "emotion"):
        v = dict(state.get(k) or {})
        v.pop("quad", None)
        print(f"  {k}: {json.dumps(v, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o), ensure_ascii=False)[:230]}")
    assert all(ran.values()), "an algorithm did not run"
    print("live panel: every algorithm ran; saved results/live_diag_check.png")


if __name__ == "__main__":
    main()
