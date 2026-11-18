"""EfficientNet-B0 REAL vs COUNTERFEIT checker (BD_Counterfeit + matched reals).

Classes: 0 = counterfeit, 1 = genuine.
High-confidence gate: only decide when max(prob) >= min_conf (default 0.90).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from .config import BD_COUNTERFEIT_AUTH_WEIGHTS, DEVICE

IMG = 224
CLASS_NAMES = {0: "counterfeit", 1: "genuine"}

_TF = transforms.Compose(
    [
        transforms.Resize((IMG, IMG)),
        transforms.CenterCrop(IMG),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class BDCounterfeitAuth:
    """Runtime wrapper around models/bd_counterfeit_auth.pt."""

    def __init__(
        self,
        weights: Path | None = None,
        device: torch.device | None = None,
        min_conf: float | None = None,
    ):
        self.device = device or DEVICE
        self.weights = Path(weights) if weights else BD_COUNTERFEIT_AUTH_WEIGHTS
        self.loaded = False
        self.min_conf = 0.95 if min_conf is None else float(min_conf)
        self.use_tta = True
        self.model = models.efficientnet_b0(weights=None)
        in_f = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_f, 2)
        # Prefer robust finetune if present
        robust = self.weights.with_name("bd_counterfeit_auth_robust.pt")
        load_path = robust if robust.is_file() else self.weights
        if load_path.is_file():
            blob = torch.load(load_path, map_location=self.device, weights_only=False)
            state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
            self.model.load_state_dict(state)
            self.weights = load_path
            if isinstance(blob, dict) and min_conf is None and "min_conf_default" in blob:
                self.min_conf = float(blob["min_conf_default"])
            self.loaded = True
        self.model.to(self.device).eval()

    def _tensor_from_bgr(self, crop_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return _TF(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

    def _tensor_from_path(self, path: Path) -> torch.Tensor:
        return _TF(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict_tensor(self, x: torch.Tensor) -> dict:
        if not self.loaded:
            return {
                "label": "unknown",
                "verdict": "CHECK_BY_HAND",
                "genuine_prob": 0.5,
                "counterfeit_prob": 0.5,
                "confidence": 0.0,
                "loaded": False,
            }
        logits = self.model(x)
        if self.use_tta:
            logits = (logits + self.model(torch.flip(x, dims=[3]))) / 2.0
        prob = F.softmax(logits, dim=1)[0]
        p_fake = float(prob[0].item())
        p_real = float(prob[1].item())
        conf = max(p_fake, p_real)
        pred = int(prob.argmax().item())
        model_class = CLASS_NAMES[pred]
        if conf < self.min_conf:
            verdict = "CHECK_BY_HAND"
            label = "unknown"
            strong = False
        else:
            verdict = "REAL" if pred == 1 else "COUNTERFEIT"
            label = "genuine" if pred == 1 else "counterfeit"
            strong = conf >= 0.95
        return {
            "label": label,
            "verdict": verdict,
            "model_class": model_class,
            "genuine_prob": p_real,
            "counterfeit_prob": p_fake,
            "confidence": conf,
            "strong": strong,
            "min_conf": self.min_conf,
            "tta": bool(self.use_tta),
            "loaded": True,
            # Glass-safe line: never speak the word counterfeit to the wearer.
            "glass_speak": "likely genuine" if (verdict == "REAL") else "check by hand",
        }

    def predict_bgr(self, crop_bgr: np.ndarray) -> dict:
        if crop_bgr is None or crop_bgr.size == 0:
            return self.predict_tensor(torch.zeros(1, 3, IMG, IMG, device=self.device))
        return self.predict_tensor(self._tensor_from_bgr(crop_bgr))

    def predict_path(self, path: Path | str) -> dict:
        return self.predict_tensor(self._tensor_from_path(Path(path)))

    # Alias used by currency_mode fallbacks that call .predict([crop])
    def predict(self, crops_bgr: list[np.ndarray]) -> dict:
        if not crops_bgr:
            return {
                "label": "unknown",
                "verdict": "CHECK_BY_HAND",
                "genuine_prob": 0.5,
                "counterfeit_prob": 0.5,
                "confidence": 0.0,
                "loaded": self.loaded,
            }
        return self.predict_bgr(crops_bgr[0])
