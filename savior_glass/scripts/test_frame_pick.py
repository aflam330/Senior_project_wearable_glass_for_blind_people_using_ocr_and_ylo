"""frame_pick picks the sharpest frame of a burst and never returns nothing when the press frame exists.

    python scripts/test_frame_pick.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frame_pick import best_frame, pick_sharpest  # noqa: E402

rng = np.random.default_rng(0)
sharp = np.zeros((480, 640, 3), np.uint8)
for i in range(12):
    cv2.putText(sharp, "Paracetamol 500 mg", (20, 40 + 36 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
blurred = [cv2.GaussianBlur(sharp, (0, 0), s) for s in (4, 3, 2.5, 5)]

i, scores = pick_sharpest(blurred[:2] + [sharp] + blurred[2:])
assert i == 2, scores

# the burst: press frame blurred, then the camera gives the sharp one
seq = iter(blurred[1:] + [sharp] * 20)
t = [0.0]
frame, info = best_frame(blurred[0], lambda: next(seq), 1.0, clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s))
assert frame is sharp and info["picked"] > 0, info
# no camera frames after the press: the press frame is read
frame, info = best_frame(blurred[0], lambda: None, 0.3, clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s))
assert frame is blurred[0] and info["frames"] == 1, info
print("frame pick: all checks pass")
