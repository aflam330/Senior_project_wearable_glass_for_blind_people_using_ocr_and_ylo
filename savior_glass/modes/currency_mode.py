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
import sys
from pathlib import Path
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
        self._yolo = None
        self._auth = None
        self._auth_kind = None
        self._auth_tf = None
        self.last_bbox = None
        self.last_class = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        logger.info("Currency detection mode activated")
        self._load_yolo()
        self._load_classifier()
        self._load_auth()

    def deactivate(self) -> None:
        logger.info("Currency detection mode deactivated")

    def cleanup(self) -> None:
        self._classifier = None
        self._yolo = None
        self._auth = None

    # ------------------------------------------------------------------
    # Core processing  (called on ACTION button press)
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        if frame is None:
            return "ক্যামেরা প্রস্তুত নয়"

        # Stage 0 — trained Taka YOLO. Do not fall through to color when it is loaded:
        # the HSV guess was announcing the wrong denomination (often 100 or 1000).
        hits = self.detect_live(frame)
        if hits:
            self._buzz("detect")
            return hits[0]["text"]
        if self._yolo is not None:
            return "নোট সনাক্ত করা যায়নি। ক্যামেরার সামনে ধরুন।"

        note_roi, bbox = self._detect_note_region(frame)
        self.last_bbox = bbox
        self.last_class = None

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
    # Stage 0 — YOLOv8s (PT / ONNX / INT8 ONNX)
    # ------------------------------------------------------------------

    def _candidate_yolo_paths(self) -> list[str]:
        """Prefer PyTorch weights. ONNX stalls on first predict if onnxruntime is missing."""
        here = config.BASE_DIR
        sibling = os.path.abspath(os.path.join(here, "..", "realtime_bangla_taka_detection", "models"))
        pt_names = ("best.pt", "best_wild_ft.pt")
        onnx_names = ("best.onnx", "best_int8.onnx")
        paths = []
        for root in (sibling, os.path.join(here, "models"), getattr(config, "CURRENCY_YOLO_DIR", "")):
            if not root:
                continue
            for name in pt_names:
                paths.append(os.path.join(root, name))
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            onnx_names = ()
        for root in (os.path.join(here, "models"), sibling):
            for name in onnx_names:
                paths.append(os.path.join(root, name))
        extra = getattr(config, "CURRENCY_YOLO_PATH", "")
        if extra:
            paths.insert(0, extra)
        # de-dupe, keep order
        seen = set()
        unique = []
        for path in paths:
            key = os.path.normcase(os.path.abspath(path))
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    def _load_yolo(self) -> None:
        for path in self._candidate_yolo_paths():
            if not os.path.isfile(path):
                continue
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(path)
                logger.info("Currency YOLO loaded from %s", path)
                return
            except Exception as exc:
                logger.warning("Failed to load currency YOLO %s: %s", path, exc)
        logger.info("No currency YOLO weights — HSV/MobileNet only")

    def _load_auth(self) -> None:
        """Genuine vs jaal (counterfeit) on the detected note crop."""
        root = os.path.abspath(os.path.join(config.BASE_DIR, "..", "realtime_bangla_taka_detection"))
        if root not in sys.path:
            sys.path.insert(0, root)
        ckpt = os.path.join(root, "results", "qduig", "prefix_ft", "seed42", "checkpoint.pt")
        try:
            if os.path.isfile(ckpt):
                from roboeye.authenticity import default_transform
                from roboeye.qduig.engine import load_qduig
                self._auth = load_qduig(Path(ckpt))
                self._auth_tf = default_transform(train=False)
                self._auth_kind = "qduig"
                logger.info("Jaal detector loaded from %s", ckpt)
                return
        except Exception as exc:
            logger.warning("Q-DUIG jaal detector failed: %s", exc)
            self._auth = None
        try:
            from roboeye.authenticity import AuthenticityClassifier
            clf = AuthenticityClassifier()
            if clf.loaded:
                self._auth = clf
                self._auth_kind = "cnnvit"
                logger.info("Jaal detector loaded (CNN+ViT fallback)")
                return
        except Exception as exc:
            logger.warning("Authenticity model failed: %s", exc)
        logger.warning("No jaal/counterfeit model loaded")

    def _authenticity(self, crop: np.ndarray) -> tuple[str, float]:
        if self._auth is None:
            return "unknown", 0.5
        try:
            if self._auth_kind == "qduig":
                from PIL import Image
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                view = self._auth_tf(Image.fromarray(rgb)).unsqueeze(0).unsqueeze(0)
                device = next(self._auth.parameters()).device
                view = view.to(device)
                mask = torch.ones(1, 1, dtype=torch.long, device=device)
                with torch.inference_mode():
                    out = self._auth(view, mask)
                genuine = float(out["prob"].reshape(-1)[0].item())
            else:
                pred = self._auth.predict([crop])
                genuine = float(pred.get("genuine_prob", 0.5))
            if not np.isfinite(genuine):
                return "unknown", 0.5
            label = "genuine" if genuine >= 0.5 else "counterfeit"
            return label, genuine
        except Exception as exc:
            logger.warning("Jaal check failed: %s", exc)
            return "unknown", 0.5

    _BN = {
        "2_taka": "দুই টাকার নোট",
        "5_taka": "পাঁচ টাকার নোট",
        "10_taka": "দশ টাকার নোট",
        "20_taka": "বিশ টাকার নোট",
        "50_taka": "পঞ্চাশ টাকার নোট",
        "100_taka": "একশত টাকার নোট",
        "200_taka": "দুইশত টাকার নোট",
        "500_taka": "পাঁচশত টাকার নোট",
        "1000_taka": "এক হাজার টাকার নোট",
    }

    def detect_live(self, frame: np.ndarray) -> list[dict]:
        """Return note boxes for the live overlay. Clears pose state when nothing is found."""
        if self._yolo is None or frame is None:
            return []
        infer = frame
        if float(frame.mean()) < 80:
            f = frame.astype(np.float32) / 255.0
            f = np.clip(f * 3.0, 0, 1)
            infer = (np.power(f, 0.5) * 255).astype(np.uint8)
        try:
            results = self._yolo.predict(infer, conf=0.25, verbose=False, imgsz=640)
        except Exception as exc:
            logger.warning("YOLO currency infer failed: %s", exc)
            return []
        boxes = results[0].boxes
        hits = []
        if boxes is not None and len(boxes) > 0:
            order = boxes.conf.argsort(descending=True)
            for idx in order.tolist():
                name = str(results[0].names[int(boxes.cls[idx].item())])
                conf = float(boxes.conf[idx].item())
                xyxy = boxes.xyxy[idx].detach().cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                text = self._BN.get(name, name.replace("_", " "))
                hits.append({
                    "name": name,
                    "text": text,
                    "conf": conf,
                    "bbox": (x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
                    "auth": "unknown",
                    "auth_en": "",
                    "genuine_prob": 0.5,
                })
        for hit in hits[:2]:
            if hit["conf"] < 0.35:
                continue
            x, y, w, h = hit["bbox"]
            x1, y1 = max(0, x), max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < 24 or crop.shape[1] < 24:
                continue
            label, genuine = self._authenticity(crop)
            hit["auth"] = label
            hit["genuine_prob"] = genuine
            if label == "counterfeit":
                hit["text"] = hit["text"] + "। জাল টাকা"
                hit["auth_en"] = "JAAL"
            elif label == "genuine":
                hit["text"] = hit["text"] + "। আসল"
                hit["auth_en"] = "REAL"
        if hits:
            self.last_bbox = hits[0]["bbox"]
            self.last_class = hits[0]["name"]
            top = hits[0]
            logger.info(
                "YOLO %s (%.0f%%) %s genuine=%.0f%%",
                top["name"], top["conf"] * 100, top["auth"], top["genuine_prob"] * 100,
            )
        else:
            self.last_bbox = None
            self.last_class = None
        return hits

    def _buzz(self, _pattern: str) -> None:
        pin = getattr(config, "HAPTIC_PIN", None)
        if pin is None:
            return
        try:
            import RPi.GPIO as GPIO
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.HIGH)
            import time
            time.sleep(0.08)
            GPIO.output(pin, GPIO.LOW)
        except Exception:
            pass

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

            blob = torch.load(config.CURRENCY_MODEL_PATH, map_location="cpu", weights_only=False)
            classes = DENOMINATIONS
            state = blob
            if isinstance(blob, dict) and "state_dict" in blob:
                state = blob["state_dict"]
                saved = blob.get("classes")
                if saved:
                    classes = [int(c) for c in saved]
            model = models.mobilenet_v3_small(weights=None)
            in_features = model.classifier[-1].in_features
            import torch.nn as nn
            model.classifier[-1] = nn.Linear(in_features, len(classes))
            model.load_state_dict(state)
            model.eval()
            self._classifier = model
            self._classifier_classes = classes

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
            classes = getattr(self, "_classifier_classes", DENOMINATIONS)
            return int(classes[idx]), score
        except Exception as exc:
            logger.warning("Classifier inference error: %s", exc)
            return DENOMINATIONS[0], 0.0
