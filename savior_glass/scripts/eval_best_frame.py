"""Does reading the sharpest frame of a short burst help with shaky hands? (frame_pick.py, config.OCR_BEST_FRAME_S)

Validation phrases only (ocr_bench.bench "val", seed 0, all three conditions: clean, wild, scene); the test phrases
are not read. Hand shake is simulated as motion blur of random length and direction on each frame of a burst of 8
(the press frame and about 0.8 s after it), with a little brightness change. Two cases, fixed before the run:
  press_shakier: the press frame blurred 6-16 px (the press moves the glass), the other frames 0-14 px
  same_shake:    every frame blurred 0-14 px
For each image: CER of the press frame, of the frame frame_pick picks, and of the least blurred frame (the best a
picker could do). The Pi's reading path (CPU, models as shipped).

    python scripts/eval_best_frame.py       ->  results/ocr_bench/best_frame.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("OCR_GPU", "0")
from frame_pick import pick_sharpest  # noqa: E402
from ocr_bench import bench as B  # noqa: E402

OUT = HERE.parent / "results" / "ocr_bench" / "best_frame.json"
BURST = 8


def motion_blur(img: np.ndarray, length: float, angle: float) -> np.ndarray:
    n = int(round(length))
    if n < 2:
        return img
    k = np.zeros((n, n), np.float32)
    k[n // 2, :] = 1.0
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), angle, 1.0), (n, n))
    return cv2.filter2D(img, -1, k / max(k.sum(), 1e-6))


def burst(img: np.ndarray, rng, press_shakier: bool) -> tuple[list[np.ndarray], list[float]]:
    lengths = [rng.uniform(6, 16) if (press_shakier and i == 0) else rng.uniform(0, 14) for i in range(BURST)]
    frames = []
    for L in lengths:
        f = motion_blur(img, L, rng.uniform(0, 180)).astype(np.float32) * rng.uniform(0.9, 1.1)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return frames, lengths


def main() -> None:
    from ocr_bench.page_bench import _ocr_mode
    from ocr_text_region import clean_text
    m = _ocr_mode()
    m.speak_line = None

    def read(frame):
        return clean_text(" ".join(m._read_v2(frame)))

    items = B.build("val", 0)
    res = {"set": "ocr_bench val, seed 0", "images": len(items), "burst": BURST}
    for case in ("press_shakier", "same_shake"):
        rng = np.random.default_rng(2026_10_29 + (case == "same_shake"))
        rows = []
        for it in items:
            frames, lengths = burst(B.decode(it), rng, case == "press_shakier")
            pick, _ = pick_sharpest(frames)
            best = int(np.argmin(lengths))
            texts = {0: read(frames[0])}
            for j in {pick, best} - {0}:
                texts[j] = read(frames[j])
            rows.append({"press": B.cer(it["gt"], texts[0]), "picked": B.cer(it["gt"], texts[pick]),
                         "least_blur": B.cer(it["gt"], texts[best]), "picked_is_least_blur": pick == best,
                         "lang": it["lang"], "cond": it["cond"]})
        mean = lambda k, rs=rows: round(float(np.mean([r[k] for r in rs])), 4)
        res[case] = {"cer_press_frame": mean("press"), "cer_picked": mean("picked"), "cer_least_blur": mean("least_blur"),
                     "picked_is_least_blur": int(sum(r["picked_is_least_blur"] for r in rows)),
                     "by_cond": {c: {"press": mean("press", [r for r in rows if r["cond"] == c]),
                                     "picked": mean("picked", [r for r in rows if r["cond"] == c])}
                                 for c in B.CONDITIONS}}
        print(case, json.dumps(res[case]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
