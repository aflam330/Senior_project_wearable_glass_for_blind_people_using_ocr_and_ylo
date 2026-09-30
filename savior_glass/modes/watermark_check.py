"""Watermark check for a back-lit Taka note (500 / 1,000), research feature, off by default.

The user holds the note up to a light; the camera sees the note back-lit, so the watermark window
(portrait + denomination electrotype) shows. The note photo is registered to a whole-note front template
of its denomination (SIFT + RANSAC, as in realtime_bangla_taka_detection/scripts/eval/watermark_features.py),
the window is cropped and scored by a MobileNetV2 (models/watermark_mobilenetv2_int8.onnx, INT8, 2.6 MB;
same decisions as FP32 on the test crops).
Measured on JaalTaka back-lit photos of counterfeit prints unseen in training: accuracy 0.929, AUC 0.976,
2 / 98 genuine called counterfeit (results/watermark/mobilenetv2.json). NOT validated on the glass camera.
It never says "counterfeit": low scores become "watermark not clear, check by hand".
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

import config

logger = logging.getLogger("smart_glass.watermark")

_ROOT = os.path.abspath(os.path.join(config.BASE_DIR, ".."))
_TEMPLATES = {
    "500_taka": os.path.join(_ROOT, "data set", "Bangladeshi_Paper_Currency_Raw", "Bangladeshi_Paper_Currency_Raw", "500", "500 Taka_0001.jpg"),
    "1000_taka": os.path.join(_ROOT, "data set", "Bangladeshi_Paper_Currency_Raw", "Bangladeshi_Paper_Currency_Raw", "1000", "1000  Taka_001.jpg"),
}
_WM_BOX = (0.70, 0.33, 0.93, 0.85)  # below the upper-right serial; same box as the research pipeline
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class WatermarkChecker:
    def __init__(self, model_path: Optional[str] = None, min_inliers: int = 12):
        import onnxruntime as ort
        path = model_path or getattr(config, "WATERMARK_MODEL_PATH", "")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # INT8 graph prints harmless "unused initializer" warnings otherwise
        self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.min_inliers = min_inliers
        self.sift = cv2.SIFT_create(nfeatures=4000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        # learned localizer (scripts/train/train_watermark_localizer.py): no template, no SIFT; used when present
        self.localizer = None
        loc_path = getattr(config, "WATERMARK_LOCALIZER_PATH", "")
        if loc_path and os.path.exists(loc_path):
            self.localizer = ort.InferenceSession(loc_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.templates = {}
        for name, p in _TEMPLATES.items():
            img = cv2.imread(p)
            if img is None:
                continue
            g = cv2.cvtColor(cv2.resize(img, None, fx=1000 / img.shape[1], fy=1000 / img.shape[1]), cv2.COLOR_BGR2GRAY)
            self.templates[name] = (self.sift.detectAndCompute(g, None), g.shape)

    def _window(self, note_bgr: np.ndarray, denom: str) -> Optional[np.ndarray]:
        if denom not in self.templates:
            return None
        col = cv2.resize(note_bgr, None, fx=700 / note_bgr.shape[1], fy=700 / note_bgr.shape[1], interpolation=cv2.INTER_AREA)
        kp, des = self.sift.detectAndCompute(cv2.cvtColor(col, cv2.COLOR_BGR2GRAY), None)
        (tkp, tdes), (th, tw) = self.templates[denom]
        if des is None or len(kp) < 10:
            return None
        good = [m for m, n in (p for p in self.matcher.knnMatch(des, tdes, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 10:
            return None
        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([tkp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
        if H is None or int(mask.sum()) < self.min_inliers:
            return None
        warped = cv2.warpPerspective(col, H, (tw, th))
        cover = cv2.warpPerspective(np.ones(col.shape[:2], np.uint8), H, (tw, th))
        x0, y0, x1, y1 = _WM_BOX
        ys, xs = slice(int(y0 * th), int(y1 * th)), slice(int(x0 * tw), int(x1 * tw))
        if cover[ys, xs].mean() < 0.8:
            return None
        return cv2.resize(warped[ys, xs], (224, 224))

    def _window_learned(self, note_bgr: np.ndarray) -> Optional[np.ndarray]:
        col = cv2.resize(note_bgr, None, fx=700 / note_bgr.shape[1], fy=700 / note_bgr.shape[1], interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(cv2.resize(col, (320, 320), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
        c = self.localizer.run(None, {"image": x})[0].reshape(4, 2) * np.float32([col.shape[1], col.shape[0]])
        area = cv2.contourArea(c.astype(np.float32)) / (col.shape[0] * col.shape[1])
        if not cv2.isContourConvex(c.astype(np.float32)) or not 0.005 < area < 0.5:
            return None  # implausible window: ask the user to hold the note straight
        M = cv2.getPerspectiveTransform(c.astype(np.float32), np.float32([[0, 0], [224, 0], [224, 224], [0, 224]]))
        return cv2.warpPerspective(col, M, (224, 224))

    def genuine_prob(self, note_bgr: np.ndarray, denom: str) -> Optional[float]:
        win = self._window_learned(note_bgr) if self.localizer is not None else None
        if win is None:
            win = self._window(note_bgr, denom)
        if win is None:
            return None
        x = ((cv2.cvtColor(win, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - _MEAN) / _STD).transpose(2, 0, 1)[None]
        lo = self.session.run(None, {"image": x.astype(np.float32)})[0][0]
        e = np.exp(lo - lo.max())
        return float(e[1] / e.sum())

    def sentence(self, p: Optional[float]) -> str:
        if p is None:
            return "জলছাপের জায়গা পাওয়া যায়নি। নোটটি আলোর সামনে সোজা করে ধরুন।"  # window not found: hold straight to the light
        if p > getattr(config, "WATERMARK_CLEAR_THRESHOLD", 0.5):
            return "জলছাপ স্পষ্ট।"  # watermark clear
        return "জলছাপ স্পষ্ট নয়। আসল কিনা হাতে যাচাই করুন।"  # not clear: check by hand (never "counterfeit")
