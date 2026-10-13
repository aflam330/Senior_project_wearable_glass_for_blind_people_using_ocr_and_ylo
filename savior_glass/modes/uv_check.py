"""UV security-fibre check for 200, 500 and 1,000 Taka notes (excitation-normalised fluorescence residual, ENFR).

A genuine Taka note carries short fibres that glow under ultraviolet light, on paper that stays dull. Ordinary paper
has no such fibres. The check counts the fibres inside the note box of a picture taken under the UV lamp.

Idea. The lamp's own light reaches the camera as blue. Anything that fluoresces gives off light of another colour.
For the paper of one note, green and red are a fixed share of blue (a = median G/B, b = median R/B); what that share
does not explain is fluorescence:

    E = max(G - a*B, R - b*B)                       emission residual
    r = tophat(E) / (blur(B) + 1)                   thin bright structures, relative to the local lamp level

Because a, b and blur(B) are measured in the same picture, the lamp's brightness, its distance and the camera's
colour balance drop out. Fibres are the connected parts of r > UV_REL_THRESHOLD that are thin and elongated.

Measured on 118 phone photographs of genuine notes (scripts/uv_dataset.py, results/uv_check/): see config.py.
Not measured: counterfeit notes (none were available) and the glass camera. The check therefore never says
"counterfeit" unless config.UV_SAY_COUNTERFEIT is switched on by hand.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import cv2
import numpy as np

import config

logger = logging.getLogger("smart_glass.uv")

WIDTH = 600          # every note crop is brought to this width, so sizes below are in the same units for any camera


def paper_ratios(bgr: np.ndarray) -> Optional[tuple]:
    """Median green/blue and red/blue of the lit pixels; None when the picture is too dark to judge."""
    f = bgr.reshape(-1, 3).astype(np.float32)
    ok = (f[:, 0] > 25) & (f[:, 0] < 250)
    if ok.sum() < 200:
        return None
    return float(np.median(f[ok, 1] / f[ok, 0])), float(np.median(f[ok, 2] / f[ok, 0]))


def fibre_mask(bgr: np.ndarray, rel_threshold: Optional[float] = None):
    """(mask of fibres at WIDTH px, number of fibres, (a, b)) for one note crop; (None, 0, None) if too dark."""
    ab = paper_ratios(bgr)
    if ab is None:
        return None, 0, None
    thr = float(getattr(config, "UV_REL_THRESHOLD", 0.12) if rel_threshold is None else rel_threshold)
    s = WIDTH / float(bgr.shape[1])
    im = cv2.resize(bgr, (WIDTH, max(8, int(bgr.shape[0] * s))), interpolation=cv2.INTER_AREA).astype(np.float32)
    B, G, R = im[..., 0], im[..., 1], im[..., 2]
    E = np.maximum(G - ab[0] * B, R - ab[1] * B)
    top = cv2.morphologyEx(E, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    rel = top / (cv2.GaussianBlur(B, (0, 0), 6) + 1.0)
    raw = (rel > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(raw, 8)
    keep, count = np.zeros_like(raw), 0
    for i in range(1, n):
        w, h, area = st[i, 2], st[i, 3], st[i, 4]
        length = max(w, h)
        # a fibre is 5-70 px long at this scale and thin: it fills little of its bounding box
        if 5 <= length <= 70 and area >= 5 and area <= 0.45 * w * h + 3.0 * length:
            keep[lab == i] = 1
            count += 1
    return keep, count, ab


def inner(bgr: np.ndarray) -> np.ndarray:
    """The note box without its border, so that the surface around the note is not judged."""
    h, w = bgr.shape[:2]
    dx, dy = int(0.12 * w), int(0.10 * h)
    return bgr[dy:h - dy, dx:w - dx] if h - 2 * dy > 16 and w - 2 * dx > 16 else bgr


class UVChecker:
    """Counts UV fibres in a note crop and words the result for the user."""

    def __init__(self, min_fibres: Optional[int] = None):
        self.min_fibres = int(getattr(config, "UV_MIN_FIBRES", 5) if min_fibres is None else min_fibres)

    def check(self, note_bgr: Optional[np.ndarray]) -> dict:
        if note_bgr is None or note_bgr.size == 0:
            return {"outcome": "no_note", "fibres": 0}
        _, count, ab = fibre_mask(inner(note_bgr))
        if ab is None:
            return {"outcome": "too_dark", "fibres": 0}
        out = "uv_marks_found" if count >= self.min_fibres else "uv_marks_not_found"
        return {"outcome": out, "fibres": int(count), "paper_gb": round(ab[0], 4), "paper_rb": round(ab[1], 4)}

    @staticmethod
    def sentence(result: dict) -> str:
        o = result.get("outcome")
        if o == "uv_marks_found":
            return "ইউভি চিহ্ন পাওয়া গেছে। সম্ভবত আসল।"                     # UV marks found. Likely genuine.
        if o == "too_dark":
            return "ইউভি আলোতে ছবি অন্ধকার। নোটটি আলোর কাছে ধরুন।"          # too dark under the UV light: hold it closer
        if o == "no_note":
            return "ইউভি আলোতে নোট পাওয়া যায়নি। নোটটি ক্যামেরার সামনে ধরুন।"    # no note found under the UV light
        if getattr(config, "UV_SAY_COUNTERFEIT", False):
            return "ইউভি চিহ্ন পাওয়া যায়নি। নোটটি জাল হতে পারে।"              # no UV marks: the note may be counterfeit
        return "ইউভি চিহ্ন পাওয়া যায়নি। আসল কিনা হাতে যাচাই করুন।"           # no UV marks: check by hand


class UVLamp:
    """The UV lamp on a GPIO pin (through a transistor). Without a pin it does nothing: a lamp with its own switch."""

    def __init__(self, pin: Optional[int] = None):
        self.pin = getattr(config, "UV_LIGHT_PIN", None) if pin is None else pin
        self._gpio = None
        if self.pin is not None:
            try:
                import RPi.GPIO as GPIO  # rpi-lgpio provides this name on the Pi 5
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
                self._gpio = GPIO
            except Exception as exc:  # laptop, or the pin is busy
                logger.info("UV lamp pin not available (%s); the lamp must be switched by hand", exc)

    def set(self, on: bool) -> None:
        if self._gpio is not None:
            self._gpio.output(self.pin, self._gpio.HIGH if on else self._gpio.LOW)


def guided_uv_check(get_frame: Callable[[], Optional[np.ndarray]], locate: Callable[[np.ndarray], Optional[np.ndarray]],
                    speak: Callable[[str], None], lamp: Optional[UVLamp] = None, checker: Optional[UVChecker] = None,
                    clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> dict:
    """Lamp on, take pictures until one can be judged or the time is up, lamp off, speak the result.
    The best picture decides, because blur and a dark frame hide fibres. (Strong sensor noise can add a few false
    ones: in the test, added noise raised the count, which is one reason the threshold is not set near zero.)"""
    lamp = lamp or UVLamp()
    checker = checker or UVChecker()
    timeout = float(getattr(config, "UV_TIMEOUT_S", 6.0))
    best = {"outcome": "no_note", "fibres": 0}
    rank = {"no_note": 0, "too_dark": 1, "uv_marks_not_found": 2, "uv_marks_found": 3}
    # checking under the UV light: hold the note close to the camera and keep it still
    speak("ইউভি আলোতে নোটটি পরীক্ষা করা হচ্ছে। নোটটি ক্যামেরার কাছে স্থির করে ধরুন।")
    lamp.set(True)
    t0, frames = clock(), 0
    try:
        sleep(float(getattr(config, "UV_SETTLE_S", 0.4)))          # let the camera's exposure settle on the UV picture
        while clock() - t0 < timeout:
            frame = get_frame()
            if frame is None:
                sleep(0.05)
                continue
            frames += 1
            res = checker.check(locate(frame))
            if (rank[res["outcome"]], res["fibres"]) > (rank[best["outcome"]], best["fibres"]):
                best = res
            if best["outcome"] == "uv_marks_found":
                break
            sleep(0.05)
    finally:
        lamp.set(False)                                             # the lamp never stays on
    best = dict(best, seconds=round(clock() - t0, 3), frames=frames)
    speak(checker.sentence(best))
    return best
