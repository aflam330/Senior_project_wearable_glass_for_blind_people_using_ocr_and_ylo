"""
Object Detection Mode — YOLOv8n (COCO 80 classes).

Runs continuous automatic scanning every OBJECT_SCAN_INTERVAL seconds.
ACTION button press forces an immediate announcement ignoring cooldowns.
Detected class names are translated to Bangla via assets/labels_bn.json.
"""
import json
import logging
import os
import time
from typing import Optional

import cv2
import numpy as np
import torch

import config
from .base_mode import BaseMode

logger = logging.getLogger("smart_glass.object_mode")

# Bangla fallback label when a class name isn't in the translation dict
_UNKNOWN_BN = "অজানা বস্তু"


class ObjectMode(BaseMode):

    def __init__(self) -> None:
        self._model = None
        self._labels_bn: dict[str, str] = {}
        self._last_announced: dict[str, float] = {}
        self._force_announce = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        logger.info("Object detection mode activated")
        self._load_model()
        self._load_labels()

    def deactivate(self) -> None:
        logger.info("Object detection mode deactivated")
        self._last_announced.clear()

    def cleanup(self) -> None:
        self._model = None

    # ------------------------------------------------------------------
    # Core processing  (called by detection_thread on a timer)
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        if self._model is None or frame is None:
            return None

        # Resize to 320×240 for fast inference on RPi 5
        small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)

        try:
            with torch.no_grad():
                results = self._model.predict(
                    small,
                    conf=config.OBJECT_CONFIDENCE,
                    verbose=False,
                    stream=False,
                )
        except Exception as exc:
            logger.warning("YOLO inference error: %s", exc)
            return None

        if not results:
            return None

        # Collect detections sorted by confidence desc
        detections: list[tuple[str, float]] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                name   = self._model.names.get(cls_id, "unknown")
                detections.append((name, conf))

        detections.sort(key=lambda x: x[1], reverse=True)

        now = time.time()
        force = self._force_announce
        self._force_announce = False

        # Pick top-3 unique classes that haven't been announced recently
        seen: set[str] = set()
        to_announce: list[str] = []
        for name, _conf in detections:
            if name in seen:
                continue
            seen.add(name)
            last = self._last_announced.get(name, 0.0)
            if force or (now - last) >= config.OBJECT_ANNOUNCE_COOLDOWN:
                to_announce.append(name)
                self._last_announced[name] = now
            if len(to_announce) == 3:
                break

        if not to_announce:
            return None

        bn_names = [self._labels_bn.get(n, _UNKNOWN_BN) for n in to_announce]
        announcement = "সামনে আছে: " + ", ".join(bn_names)
        logger.info("Objects: %s", announcement)
        return announcement

    def force_announce_now(self) -> None:
        """Called by ACTION button press — ignores cooldown timers."""
        self._force_announce = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not os.path.isfile(config.YOLO_MODEL_PATH):
            logger.warning(
                "YOLOv8n model not found at %s. "
                "Run: python3 -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\" "
                "then copy yolov8n.pt into models/",
                config.YOLO_MODEL_PATH,
            )
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(config.YOLO_MODEL_PATH)
            # Warmup — avoids long first-inference delay
            dummy = np.zeros((240, 320, 3), dtype=np.uint8)
            with torch.no_grad():
                self._model.predict(dummy, verbose=False)
            logger.info("YOLOv8n model loaded and warmed up")
        except Exception as exc:
            logger.error("Failed to load YOLO model: %s", exc)
            self._model = None

    def _load_labels(self) -> None:
        if not os.path.isfile(config.LABELS_BN_PATH):
            logger.warning("Bangla labels not found at %s", config.LABELS_BN_PATH)
            return
        try:
            with open(config.LABELS_BN_PATH, encoding="utf-8") as f:
                self._labels_bn = json.load(f)
            logger.info("Loaded %d Bangla labels", len(self._labels_bn))
        except Exception as exc:
            # An uncaught error here would propagate out of activate() —
            # crashing startup, or (via the mode-switch button callback)
            # leaving the app with the mode index advanced but the new
            # mode only half-activated. Fall back to English class names.
            logger.error("Failed to load Bangla labels: %s", exc)
            self._labels_bn = {}
