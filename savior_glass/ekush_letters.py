"""Ekush handwritten Bangla letter classifier (offline, 122 classes).

Dataset: data set/OCR bangla data set/Ekush Data set
Images are 28x28, dark background, lighter ink.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("smart_glass.ekush")

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "models" / "ekush_cnn.pt"
DEFAULT_LABELS = ROOT / "assets" / "ekush_labels.json"
N_CLASSES = 122


class EkushCNN(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def load_label_map(path: Path = DEFAULT_LABELS) -> dict[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data["id_to_char"].items()}


def labels_from_csv(csv_path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = int(str(row["Folder Name"]).strip())
            ch = str(row["Char Name"]).strip()
            out[cid] = ch
    return out


def preprocess_glyph(gray: np.ndarray, size: int = 28) -> np.ndarray:
    """Match Ekush polarity: light ink on dark background, 28x28 float32."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    if float(g.mean()) > 127:
        g = 255.0 - g
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    g = g / 255.0
    return g


class EkushRecognizer:
    def __init__(self, weights: Path = DEFAULT_WEIGHTS, labels: Path = DEFAULT_LABELS) -> None:
        self.ok = False
        self.device = torch.device("cpu")
        self.id_to_char = load_label_map(labels) if labels.is_file() else {}
        self.model = EkushCNN(len(self.id_to_char) or N_CLASSES)
        if not weights.is_file():
            logger.warning("Ekush weights missing: %s", weights)
            return
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()
        self.ok = True
        logger.info("Ekush CNN ready (%d classes)", len(self.id_to_char))

    @torch.inference_mode()
    def classify_crop(self, gray: np.ndarray) -> tuple[str, float]:
        if not self.ok:
            return "", 0.0
        x = preprocess_glyph(gray)
        t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
        logits = self.model(t)
        prob = torch.softmax(logits, dim=1)[0]
        idx = int(prob.argmax().item())
        return self.id_to_char.get(idx, ""), float(prob[idx].item())

    def read_frame(self, frame: np.ndarray, min_conf: float = 0.45) -> str:
        """Segment dark blobs and classify each as an Ekush glyph. Offline."""
        if not self.ok or frame is None:
            return ""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = gray.shape[:2]
        if w < 400:
            scale = 400 / max(w, 1)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        # binary: ink as white
        if float(clahe.mean()) > 127:
            bw = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        else:
            bw = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        area_img = bw.shape[0] * bw.shape[1]
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < 80 or area > 0.25 * area_img:
                continue
            ar = cw / max(ch, 1)
            if ar < 0.15 or ar > 4.0:
                continue
            boxes.append((x, y, cw, ch))
        boxes.sort(key=lambda b: (b[1] // max(bw.shape[0] // 8, 1), b[0]))
        chars: list[str] = []
        for x, y, cw, ch in boxes[:48]:
            pad = 2
            crop = clahe[max(0, y - pad): y + ch + pad, max(0, x - pad): x + cw + pad]
            if crop.size == 0:
                continue
            ch_s, conf = self.classify_crop(crop)
            if ch_s and conf >= min_conf:
                chars.append(ch_s)
        return "".join(chars)
