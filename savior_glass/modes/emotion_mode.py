"""
Emotion mode: says the facial expression of the nearest (largest) face in front of the user.

Triggered by the ACTION button. Uses roboeye.fer_emotion.EmotionDetector (YuNet / Haar face detector, then the
RAF-DB EfficientNetV2-S classifier in models/emotion_faces.pt blended with FER+). If emotion_faces.pt is missing
the detector falls back to the small FER+ model alone, which is much weaker; activate() logs which one is in use.
Measured on the RAF-DB test set (3,068 faces): accuracy 86.5 % (paper_evidence/emotion/emotion_rafdb.json).
Not measured on the glass camera.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import numpy as np

import config
from .base_mode import BaseMode

logger = logging.getLogger("smart_glass.emotion_mode")

_BN = {
    "happiness": "খুশি", "happy": "খুশি",
    "sadness": "দুঃখিত", "sad": "দুঃখিত",
    "anger": "রাগান্বিত", "angry": "রাগান্বিত",
    "surprise": "অবাক",
    "fear": "ভীত",
    "disgust": "বিরক্ত",
    "contempt": "বিরক্ত",
    "neutral": "স্বাভাবিক",
}


class EmotionMode(BaseMode):

    def __init__(self) -> None:
        self._detector = None
        self.last = None   # last result dict (label, prob, face box, backend); read by the preview and the field log

    def activate(self) -> None:
        logger.info("Emotion mode activated")
        if self._detector is not None:
            return
        root = os.path.abspath(os.path.join(config.BASE_DIR, "..", "realtime_bangla_taka_detection"))
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from roboeye.fer_emotion import EmotionDetector
            det = EmotionDetector(download=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Emotion detector could not be loaded: %s", exc)
            return
        if det.face is None:
            logger.warning("Emotion mode: no face detector (YuNet model or Haar cascade missing)")
            return
        self._detector = det
        strong = det.rgb_net is not None
        logger.info("Emotion detector ready: face detector %s, classifier %s", det.face_kind,
                    "RAF-DB EfficientNetV2-S (models/emotion_faces.pt)" if strong else "FER+ only (emotion_faces.pt missing: weaker)")

    def warm_up(self) -> None:
        if self._detector is not None:
            try:
                self._detector.predict(np.full((480, 640, 3), 128, np.uint8))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Emotion warm-up failed: %s", exc)

    def deactivate(self) -> None:
        logger.info("Emotion mode deactivated")

    def cleanup(self) -> None:
        self._detector = None

    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        if frame is None:
            return "ক্যামেরা প্রস্তুত নয়"
        if self._detector is None:
            return "ইমোশন মডেল লোড হয়নি"            # emotion model not loaded
        try:
            res = self._detector.predict(frame)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Emotion inference failed: %s", exc)
            return "মুখের অভিব্যক্তি বোঝা যায়নি"      # could not read the expression
        self.last = {k: res.get(k) for k in ("label", "prob", "face", "backend")}
        if res.get("face") is None:
            return "কোনো মুখ পাওয়া যায়নি। ক্যামেরা মানুষের মুখের দিকে ধরুন।"   # no face found: point the camera at a face
        label = str(res.get("label", "neutral")).lower()
        logger.info("Emotion %s (%.0f%%) via %s", label, 100 * float(res.get("prob") or 0), res.get("backend"))
        return f"সামনের মানুষটিকে {_BN.get(label, label)} দেখাচ্ছে"   # the person in front looks <emotion>
