"""Dual-head detector: YOLOv8 denomination + authenticity head on shared features.

Head A — Ultralytics Detect (9 Taka classes), existing `models/best.pt`.
Head B — ROI-pooled SPPF features → 2-way genuine/counterfeit linear layer,
          ensembled with the CNN+ViT crop classifier and CLIP/DINO prototypes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

from .authenticity import AuthenticityClassifier
from .clip_zero_shot import PrototypeAuthenticator
from .config import DEVICE, DUAL_HEAD_WEIGHTS, YOLO_WEIGHTS


class AuthHead(nn.Module):
    """Lightweight authenticity head on pooled YOLO backbone features."""

    def __init__(self, in_ch: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


def _sppf_channels(yolo: YOLO) -> int:
    layer = yolo.model.model[9]
    # SPPF output channels
    conv = getattr(layer, "cv2", None) or getattr(layer, "cv1", None)
    if conv is None:
        return 512
    return int(conv.conv.out_channels)


class DualHeadYOLO:
    """Run denomination detection and authenticity together."""

    def __init__(
        self,
        yolo_path: Path | None = None,
        conf: float = 0.35,
        device: torch.device | None = None,
    ):
        self.device = device or DEVICE
        self.conf = conf
        self.detector = YOLO(str(yolo_path or YOLO_WEIGHTS))
        self.names = self.detector.names
        in_ch = _sppf_channels(self.detector)
        self.auth_head = AuthHead(in_ch=in_ch).to(self.device)
        self.auth_head_loaded = False
        if DUAL_HEAD_WEIGHTS.is_file():
            state = torch.load(DUAL_HEAD_WEIGHTS, map_location=self.device, weights_only=True)
            self.auth_head.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
            self.auth_head_loaded = True
        self.auth_head.eval()
        self.cnn_vit = AuthenticityClassifier(device=self.device)
        try:
            self.prototypes = PrototypeAuthenticator(device=self.device)
        except Exception:
            self.prototypes = None
        self._feat = None
        self._hook = self.detector.model.model[9].register_forward_hook(self._save_feat)

    def _save_feat(self, _module, _inp, out):
        self._feat = out

    def close(self):
        self._hook.remove()

    def _head_prob(self, box_xyxy: np.ndarray, frame_hw: tuple[int, int]) -> float | None:
        if self._feat is None or not self.auth_head_loaded:
            return None
        feat = self._feat
        if feat.ndim == 4:
            b, c, fh, fw = feat.shape
        else:
            return None
        h, w = frame_hw
        x1, y1, x2, y2 = box_xyxy
        rx1, ry1 = x1 / w * fw, y1 / h * fh
        rx2, ry2 = x2 / w * fw, y2 / h * fh
        x1i, x2i = int(max(0, rx1)), int(min(fw, max(rx1 + 1, rx2)))
        y1i, y2i = int(max(0, ry1)), int(min(fh, max(ry1 + 1, ry2)))
        if x2i <= x1i or y2i <= y1i:
            return None
        roi = feat[:, :, y1i:y2i, x1i:x2i]
        with torch.inference_mode():
            logits = self.auth_head(roi.to(self.device))
            prob = F.softmax(logits, dim=1)[0]
        return float(prob[1].item())

    def predict(self, frame_bgr: np.ndarray) -> list[dict]:
        h, w = frame_bgr.shape[:2]
        results = self.detector.predict(frame_bgr, conf=self.conf, verbose=False)
        r0 = results[0]
        detections: list[dict] = []
        if r0.boxes is None or len(r0.boxes) == 0:
            return detections
        for b in r0.boxes:
            xyxy = b.xyxy[0].detach().cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame_bgr[y1:y2, x1:x2]
            cls_id = int(b.cls.item())
            name = self.names[cls_id]
            conf = float(b.conf.item())
            votes = []
            head_p = self._head_prob(xyxy, (h, w))
            if head_p is not None:
                votes.append(head_p)
            if crop.size:
                cnn = self.cnn_vit.predict([crop])
                if cnn["loaded"]:
                    votes.append(cnn["genuine_prob"])
                if self.prototypes is not None:
                    proto = self.prototypes.predict(crop)
                    if proto.get("loaded"):
                        votes.append(proto["genuine_prob"])
            genuine = float(sum(votes) / len(votes)) if votes else 0.5
            detections.append(
                {
                    "cls_id": cls_id,
                    "name": name,
                    "conf": conf,
                    "xyxy": (x1, y1, x2, y2),
                    "crop": crop,
                    "genuine_prob": genuine,
                    "auth_label": "genuine" if genuine >= 0.55 else ("counterfeit" if genuine <= 0.45 else "unknown"),
                    "n_votes": len(votes),
                    "cnn_vit_loaded": self.cnn_vit.loaded,
                    "head_loaded": self.auth_head_loaded,
                }
            )
        return detections
