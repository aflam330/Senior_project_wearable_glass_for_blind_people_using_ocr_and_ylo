"""Task 2: preprocessing variants. Each maps a BGR frame to the image EasyOCR reads."""
from __future__ import annotations

import cv2
import numpy as np


def _upscale(img, target=1200):
    h, w = img.shape[:2]
    if w < target:
        return cv2.resize(img, (target, int(h * target / w)), interpolation=cv2.INTER_CUBIC)
    if w > 1600:
        return cv2.resize(img, (1600, int(h * 1600 / w)), interpolation=cv2.INTER_AREA)
    return img


def gray(bgr):
    return cv2.cvtColor(_upscale(bgr), cv2.COLOR_BGR2GRAY)


def color(bgr):
    return _upscale(bgr)


def no_upscale(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def clahe(bgr):
    return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray(bgr))


def bilateral(bgr):
    return cv2.bilateralFilter(gray(bgr), 9, 75, 75)


def glass(bgr):
    return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(cv2.bilateralFilter(gray(bgr), 9, 75, 75))


def unsharp(bgr, amount=1.0, sigma=1.5):
    g = gray(bgr).astype(np.float32)
    return np.clip(g + amount * (g - cv2.GaussianBlur(g, (0, 0), sigma)), 0, 255).astype(np.uint8)


def glass_unsharp(bgr):
    g = glass(bgr).astype(np.float32)
    return np.clip(g + 0.8 * (g - cv2.GaussianBlur(g, (0, 0), 1.5)), 0, 255).astype(np.uint8)


def sauvola(bgr, window=31, k=0.2):
    from skimage.filters import threshold_sauvola
    g = gray(bgr)
    return ((g > threshold_sauvola(g, window_size=window, k=k)) * 255).astype(np.uint8)


def niblack(bgr, window=31, k=0.2):
    from skimage.filters import threshold_niblack
    g = gray(bgr)
    return ((g > threshold_niblack(g, window_size=window, k=k)) * 255).astype(np.uint8)


def deskewed(bgr):
    from deskew import determine_skew
    g = gray(bgr)
    ang = determine_skew(cv2.resize(g, None, fx=0.5, fy=0.5))
    if ang is None or abs(ang) > 15:
        return g
    M = cv2.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), ang, 1.0)
    return cv2.warpAffine(g, M, (g.shape[1], g.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def perspective(bgr):
    """Largest bright four-sided region (a sign or page), warped flat. Falls back to gray()."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 50, 150)
    cnts, _ = cv2.findContours(cv2.dilate(edges, None), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        a = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if len(a) == 4 and cv2.contourArea(a) > 0.05 * g.size:
            best = a.reshape(4, 2).astype(np.float32)
            break
    if best is None:
        return gray(bgr)
    s, d = best.sum(1), np.diff(best, axis=1).ravel()
    q = np.float32([best[s.argmin()], best[d.argmin()], best[s.argmax()], best[d.argmax()]])
    w = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[3] - q[2])))
    h = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
    if w < 40 or h < 15:
        return gray(bgr)
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return gray(cv2.warpPerspective(bgr, M, (w, h)))


VARIANTS = {"glass": glass, "gray": gray, "color": color, "no_upscale": no_upscale, "clahe": clahe, "bilateral": bilateral,
            "unsharp": unsharp, "glass_unsharp": glass_unsharp, "sauvola": sauvola, "niblack": niblack,
            "deskew": deskewed, "perspective": perspective}
