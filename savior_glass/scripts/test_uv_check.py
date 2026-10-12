"""Checks of the UV fibre check (modes/uv_check.py). Uses the UV photographs when they are present
(<project>/data set/uv_notes, not in the repository) and synthetic pictures otherwise.

    python scripts/test_uv_check.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config  # noqa: E402
from modes.uv_check import UVChecker, guided_uv_check  # noqa: E402

DATA = HERE.parents[1] / "data set" / "uv_notes"


class Lamp:
    def __init__(self):
        self.log = []

    def set(self, on):
        self.log.append(bool(on))


def synthetic(fibres: int, seed=0):
    """Dull blue 'paper' under a UV lamp, with thin greenish-white strokes as fibres."""
    rng = np.random.default_rng(seed)
    img = np.zeros((400, 760, 3), np.float32)
    img[..., 0] = 110 + rng.normal(0, 2, img.shape[:2])
    img[..., 1] = 0.22 * img[..., 0]
    img[..., 2] = 0.05 * img[..., 0]
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(fibres):
        x, y = int(rng.integers(110, 650)), int(rng.integers(60, 340))
        a = rng.uniform(0, np.pi)
        cv2.line(img, (x, y), (int(x + 22 * np.cos(a)), int(y + 22 * np.sin(a))), (200, 230, 210), 2, cv2.LINE_AA)
    return img


def run(frame, locate=lambda f: f):
    said, lamp, t = [], Lamp(), [0.0]

    def clock():
        t[0] += 0.3
        return t[0]
    ev = guided_uv_check(lambda: frame, locate, said.append, lamp=lamp, checker=UVChecker(), clock=clock, sleep=lambda s: None)
    assert lamp.log and lamp.log[0] is True and lamp.log[-1] is False, "the lamp must be switched on, then off"
    return ev, said


ev, said = run(synthetic(40))
assert ev["outcome"] == "uv_marks_found", ev
ev, said = run(synthetic(0))
assert ev["outcome"] == "uv_marks_not_found" and "জাল" not in said[-1], (ev, said)      # never "counterfeit" by default
ev, _ = run(np.zeros((400, 760, 3), np.uint8))
assert ev["outcome"] == "too_dark", ev
ev, _ = run(synthetic(40), locate=lambda f: None)
assert ev["outcome"] == "no_note", ev
config.UV_SAY_COUNTERFEIT = True
assert "জাল" in UVChecker.sentence({"outcome": "uv_marks_not_found"})
config.UV_SAY_COUNTERFEIT = False

# a lamp that fails while the check runs must still be switched off
lamp = Lamp()
try:
    guided_uv_check(lambda: (_ for _ in ()).throw(RuntimeError("camera")), lambda f: f, lambda s: None, lamp=lamp,
                    clock=lambda: 0.0, sleep=lambda s: None)
except RuntimeError:
    pass
assert lamp.log[-1] is False

real = ""
if (DATA / "boxes.csv").exists():
    rows = [r for r in csv.DictReader(open(DATA / "boxes.csv", encoding="utf-8")) if r["file"] == "uv_200_003.png" or
            (r["file"] == "uv_200_013.png" and r["x1"])]
    for r in rows:
        img = cv2.imdecode(np.fromfile(str(DATA / r["value"] / r["file"]), np.uint8), cv2.IMREAD_COLOR)
        crop = img if not r["x1"] else img[int(r["y1"]):int(r["y2"]), int(r["x1"]):int(r["x2"])]
        ev, _ = run(crop)
        assert ev["outcome"] == "uv_marks_found", (r["file"], ev)
    real = f"; {len(rows)} real photographs of genuine notes accepted"
print("uv check: all checks pass" + real)
