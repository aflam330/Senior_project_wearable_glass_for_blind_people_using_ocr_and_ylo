"""
Currency Detection Mode — Bangladeshi Taka (BDT) notes.

Two-stage detection (graceful degradation):

Stage 1 — HSV Color Analysis (always available, zero extra deps):
  Each Taka denomination has a dominant color family.
  We find the largest rectangle-like contour and match its HSV histogram
  against predefined color profiles.

Stage 2 — MobileNetV3-Small Classifier (optional, higher accuracy):
  If models/currency_mobilenet.pt exists (trained with scripts/train_currency.py),
  we crop the detected note region, classify, and override Stage 1 when
  confidence >= CURRENCY_CONFIDENCE.

Triggered only on ACTION button press — not continuous — because OCR
and object modes are already competing for CPU.
"""
import logging
import os
from typing import Optional

import cv2
import numpy as np
import torch

import config
from .base_mode import BaseMode

logger = logging.getLogger("smart_glass.currency_mode")

# Bangla names for each denomination
_DENOMINATION_BN: dict[int, str] = {
    10:   "দশ টাকার নোট",
    20:   "বিশ টাকার নোট",
    50:   "পঞ্চাশ টাকার নোট",
    100:  "একশত টাকার নোট",
    200:  "দুইশত টাকার নোট",
    500:  "পাঁচশত টাকার নোট",
    1000: "এক হাজার টাকার নোট",
}

DENOMINATIONS = [10, 20, 50, 100, 200, 500, 1000]

# HSV color profiles per denomination (hue_low, hue_high, sat_min, val_min)
# Approximate ranges — real notes vary by age and lighting.
_HSV_PROFILES: dict[int, list[tuple]] = {
    10:   [(35,  85,  50, 80)],                           # green
    20:   [(15,  35,  50, 80)],                           # brown/olive
    50:   [(120, 160, 40, 80)],                           # purple/violet
    100:  [(0,   10,  60, 80), (170, 180, 60, 80)],       # red (wraps around)
    200:  [(100, 130, 50, 70)],                           # blue/indigo
    500:  [(40,  80,  60, 50)],                           # dark green
    1000: [(80,  105, 30, 50)],                           # teal/grey-green
}

# Note aspect ratio: ~76mm × 156mm ≈ 1 : 2.05
_NOTE_ASPECT_MIN = 1.5
_NOTE_ASPECT_MAX = 2.8
_MIN_CONTOUR_AREA = 8000   # pixels² — ignore tiny blobs


class CurrencyMode(BaseMode):

    def __init__(self) -> None:
        self._classifier = None
        self._transform  = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        logger.info("Currency detection mode activated")
        self._load_classifier()

    def deactivate(self) -> None:
        logger.info("Currency detection mode deactivated")

    def cleanup(self) -> None:
        self._classifier = None

    # ------------------------------------------------------------------
    # Core processing  (called on ACTION button press)
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        if frame is None:
            return "ক্যামেরা প্রস্তুত নয়"

        note_roi, bbox = self._detect_note_region(frame)

        if note_roi is None:
            return "নোট সনাক্ত করা যায়নি। ক্যামেরার সামনে ধরুন।"

        # Stage 1: color-based guess
        color_result, color_score = self._color_match(note_roi)

        # Stage 2: override with classifier if available and confident
        if self._classifier is not None:
            cls_result, cls_score = self._classify(note_roi)
            if cls_score >= config.CURRENCY_CONFIDENCE:
                logger.info(
                    "Classifier: %d tk (%.0f%%) | Color: %s tk (%.0f%%)",
                    cls_result, cls_score * 100,
                    color_result, color_score * 100,
                )
                return _DENOMINATION_BN.get(cls_result, f"{cls_result} টাকার নোট")

        if color_result and color_score >= 0.30:
            logger.info("Color match: %d tk (%.0f%%)", color_result, color_score * 100)
            return _DENOMINATION_BN.get(color_result, f"{color_result} টাকার নোট")

        return "নোট নিশ্চিত করা যায়নি। আরও কাছে ধরুন।"

    # ------------------------------------------------------------------
    # Stage 1 — HSV color analysis
    # ------------------------------------------------------------------

    def _detect_note_region(self, frame: np.ndarray):
        """
        Find the largest rectangular contour with note-like aspect ratio.
        Returns (cropped_roi, bbox) or (None, None).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Build a broad mask that catches all possible note colors
        masks = []
        for profiles in _HSV_PROFILES.values():
            for (hl, hh, sl, vl) in profiles:
                lower = np.array([hl, sl, vl])
                upper = np.array([hh, 255, 255])
                masks.append(cv2.inRange(hsv, lower, upper))

        combined_mask = masks[0]
        for m in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, m)

        # Morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, None

        # Keep contours that are large enough and note-shaped
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < _MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = max(w, h) / max(min(w, h), 1)
            if _NOTE_ASPECT_MIN <= aspect <= _NOTE_ASPECT_MAX:
                candidates.append((area, (x, y, w, h), cnt))

        if not candidates:
            # Fallback: largest contour regardless of shape
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            bbox = (x, y, w, h)
        else:
            candidates.sort(key=lambda c: c[0], reverse=True)
            _, bbox, _ = candidates[0]

        x, y, w, h = bbox
        # Add small padding
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None
        return roi, bbox

    def _color_match(self, roi: np.ndarray) -> tuple[Optional[int], float]:
        """Score each denomination's HSV profile against the ROI."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]

        best_denom: Optional[int] = None
        best_score: float = 0.0

        for denom, profiles in _HSV_PROFILES.items():
            score = 0.0
            for (hl, hh, sl, vl) in profiles:
                lower = np.array([hl, sl, vl])
                upper = np.array([hh, 255, 255])
                mask = cv2.inRange(hsv, lower, upper)
                score += cv2.countNonZero(mask) / total_pixels

            if score > best_score:
                best_score = score
                best_denom = denom

        return best_denom, best_score

    # ------------------------------------------------------------------
    # Stage 2 — MobileNetV3-Small classifier
    # ------------------------------------------------------------------

    def _load_classifier(self) -> None:
        if not os.path.isfile(config.CURRENCY_MODEL_PATH):
            logger.info(
                "No currency classifier at %s — using color analysis only. "
                "Train one with: python3 scripts/train_currency.py",
                config.CURRENCY_MODEL_PATH,
            )
            return
        try:
            import torchvision.models as models
            import torchvision.transforms as T

            model = models.mobilenet_v3_small(weights=None)
            # Replace head to match 7 classes
            in_features = model.classifier[-1].in_features
            import torch.nn as nn
            model.classifier[-1] = nn.Linear(in_features, len(DENOMINATIONS))
            model.load_state_dict(torch.load(config.CURRENCY_MODEL_PATH, map_location="cpu"))
            model.eval()
            self._classifier = model

            self._transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
            logger.info("Currency classifier loaded from %s", config.CURRENCY_MODEL_PATH)
        except Exception as exc:
            logger.warning("Failed to load currency classifier: %s", exc)
            self._classifier = None

    def _classify(self, roi: np.ndarray) -> tuple[int, float]:
        """Run MobileNetV3 classifier on cropped note region."""
        try:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0)
            with torch.no_grad():
                logits = self._classifier(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            idx   = int(probs.argmax())
            score = float(probs[idx])
            return DENOMINATIONS[idx], score
        except Exception as exc:
            logger.warning("Classifier inference error: %s", exc)
            return DENOMINATIONS[0], 0.0
