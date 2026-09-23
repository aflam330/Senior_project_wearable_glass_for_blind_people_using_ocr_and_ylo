"""Robustness corruptions. Extends camva.apply_corruption with contrast/glare/noise."""

from __future__ import annotations

import cv2
import numpy as np

from ..camva.data import apply_corruption as camva_corrupt


def apply_corruption(bgr: np.ndarray, name: str, severity: float) -> np.ndarray:
    if name == "contrast":
        img = bgr.astype(np.float32)
        mean = img.mean()
        return np.clip((img - mean) * float(severity) + mean, 0, 255).astype(np.uint8)
    if name == "glare":
        h, w = bgr.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        cy, cx = int(h * 0.3), int(w * 0.7)
        rad = max(h, w) * 0.25
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad ** 2))
        add = (blob * 255.0 * float(severity))[..., None]
        return np.clip(bgr.astype(np.float32) + add, 0, 255).astype(np.uint8)
    if name == "sensor_noise":
        noise = np.random.normal(0.0, float(severity), bgr.shape)
        return np.clip(bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return camva_corrupt(bgr, name, severity)
