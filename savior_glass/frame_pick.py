"""Steadier text reading: after an ACTION press in text-reading mode, look at the camera for OCR_BEST_FRAME_S seconds
and read the sharpest frame instead of the one taken at the moment of the press, which a press shakes most.

Sharpness is the variance of the Laplacian on the grey frame at 640 px wide (larger = more edge detail). Measured on
the rendered validation set with simulated hand shake (scripts/eval_best_frame.py, results/ocr_bench/best_frame.json).
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import cv2
import numpy as np


def sharpness(frame: np.ndarray) -> float:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if g.shape[1] != 640:
        g = cv2.resize(g, (640, max(1, round(g.shape[0] * 640 / g.shape[1]))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def pick_sharpest(frames: list[np.ndarray]) -> tuple[int, list[float]]:
    """Index of the sharpest frame and every frame's score."""
    scores = [sharpness(f) for f in frames]
    return int(np.argmax(scores)), scores


def best_frame(first: Optional[np.ndarray], get_frame: Callable[[], Optional[np.ndarray]], seconds: float,
               clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> tuple:
    """(sharpest frame, info) among the press frame and frames taken every 0.1 s for `seconds` after it."""
    frames = [first] if first is not None else []
    t0 = clock()
    while clock() - t0 < seconds:
        sleep(0.1)
        f = get_frame()
        if f is not None:
            frames.append(f)
    if not frames:
        return None, {"frames": 0}
    i, scores = pick_sharpest(frames)
    return frames[i], {"frames": len(frames), "picked": i, "sharpness": round(scores[i], 1),
                       "press_sharpness": round(scores[0], 1)}
