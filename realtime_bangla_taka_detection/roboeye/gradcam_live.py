"""Live Grad-CAM overlay on YOLOv8s (SPPF layer).

Self-contained: no pytorch-grad-cam dependency. Grad-CAM runs on a private copy of the
detector network, with hooks attached only for the duration of one call, so the model used
for live prediction is never modified (the earlier version hooked the shared model
permanently, turned on gradients for its weights, and did nothing when the optional
grad-cam package was missing).
"""

from __future__ import annotations

import copy
import logging

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import YOLO_WEIGHTS

logger = logging.getLogger(__name__)

SPPF_INDEX = 9  # YOLOv8 backbone: model.model[9] is SPPF


class LiveGradCAM:
    def __init__(self, yolo: YOLO | None = None, layer_index: int = SPPF_INDEX):
        self.ok = False
        self.error: str | None = None
        try:
            model = yolo or YOLO(str(YOLO_WEIGHTS))
            self._net = copy.deepcopy(model.model).float().eval()
            for p in self._net.parameters():
                p.requires_grad_(False)
            self._layer = self._net.model[layer_index]
            self._device = next(self._net.parameters()).device
            self.ok = True
        except Exception as exc:  # keep the live app running without the overlay
            self.error = f"{type(exc).__name__}: {exc}"
            logger.warning("Grad-CAM disabled: %s", self.error)

    def heatmap(self, frame_bgr: np.ndarray, class_id: int = 5) -> np.ndarray | None:
        if not self.ok:
            return None
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (640, 640)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).div(255.0).to(self._device)
        x.requires_grad_(True)  # weights stay frozen; the graph flows through the input
        store: dict[str, torch.Tensor] = {}

        def _keep(_module, _inp, out):
            out.retain_grad()
            store["act"] = out

        handle = self._layer.register_forward_hook(_keep)
        try:
            with torch.enable_grad():
                out = self._net(x)
                preds = out[0] if isinstance(out, (list, tuple)) else out  # (1, 4+nc, anchors)
                scores = preds[0, 4 + class_id]
                score = torch.topk(scores, min(50, scores.numel())).values.sum()
                score.backward()
            act = store["act"]
            weights = act.grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * act).sum(dim=1))[0].detach().cpu().numpy()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logger.warning("Grad-CAM failed: %s", self.error)
            return None
        finally:
            handle.remove()
        cam = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        cam = cv2.GaussianBlur(cam, (0, 0), 5)
        return np.clip(cam / (cam.max() + 1e-8), 0.0, 1.0)

    def overlay(self, frame_bgr: np.ndarray, class_id: int = 5, alpha: float = 0.45) -> np.ndarray:
        heat = self.heatmap(frame_bgr, class_id=class_id)
        if heat is None:
            return frame_bgr
        color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(frame_bgr, 1.0 - alpha, color, alpha, 0)
