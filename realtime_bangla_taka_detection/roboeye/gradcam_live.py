"""Live Grad-CAM overlay on YOLOv8s (SPPF layer)."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

from .config import YOLO_WEIGHTS


class _Wrap(nn.Module):
    def __init__(self, yolo: YOLO):
        super().__init__()
        self.net = yolo.model
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(True)

    def forward(self, x):
        out = self.net(x)
        return out[0] if isinstance(out, (list, tuple)) else out


class _ClassScoreTarget:
    def __init__(self, cid: int):
        self.cid = cid

    def __call__(self, output):
        scores = output[0, 4 + self.cid]
        k = min(50, scores.numel())
        return torch.topk(scores, k).values.sum()


class LiveGradCAM:
    def __init__(self, yolo: YOLO | None = None):
        self.ok = False
        self._cam = None
        self._wrap = None
        try:
            from pytorch_grad_cam import GradCAM
        except Exception:
            return
        model = yolo or YOLO(str(YOLO_WEIGHTS))
        self._wrap = _Wrap(model)
        self._cam = GradCAM(self._wrap, [self._wrap.net.model[9]])
        self.ok = True

    def heatmap(self, frame_bgr: np.ndarray, class_id: int = 5) -> np.ndarray | None:
        if not self.ok:
            return None
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (640, 640))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        try:
            heat = self._cam(input_tensor=tensor, targets=[_ClassScoreTarget(class_id)])[0]
        except Exception:
            return None
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
        heat = cv2.GaussianBlur(heat, (0, 0), 5)
        return np.clip(heat / (heat.max() + 1e-8), 0.0, 1.0)

    def overlay(self, frame_bgr: np.ndarray, class_id: int = 5, alpha: float = 0.45) -> np.ndarray:
        heat = self.heatmap(frame_bgr, class_id=class_id)
        if heat is None:
            return frame_bgr
        color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(frame_bgr, 1.0 - alpha, color, alpha, 0)
