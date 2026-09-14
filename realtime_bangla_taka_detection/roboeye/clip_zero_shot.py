"""Zero-shot / prototype authenticity using CLIP, DINOv2, or ImageNet CNN.

Builds mean embedding vectors for genuine and counterfeit JaalTaka notes,
then classifies a crop by cosine similarity. CLIP/DINOv2 are used when the
optional packages are installed; otherwise MobileNetV3-Small ImageNet
features are the encoder (still a real frozen visual backbone).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from .authenticity import list_jaaltaka_notes
from .config import DEVICE, PROTO_WEIGHTS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _MobileNetEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = m.features
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def encode_pil(self, img: Image.Image) -> torch.Tensor:
        x = self.tf(img.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            z = self.pool(self.features(x)).flatten(1)
        return F.normalize(z, dim=1)[0]


def _try_clip(device: torch.device):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception:
        return None
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception:
        return None

    def encode(img: Image.Image) -> torch.Tensor:
        inputs = proc(images=img.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            z = model.get_image_features(**inputs)
            if not torch.is_tensor(z):
                z = getattr(z, "image_embeds", None) or getattr(z, "pooler_output")
            if z.dim() == 1:
                z = z.unsqueeze(0)
        return F.normalize(z, dim=-1)[0].cpu()

    return encode, "clip"


def _try_dinov2(device: torch.device):
    import os

    if os.environ.get("ROBOEYE_DINO") != "1":
        return None
    try:
        encoder = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            verbose=False,
            skip_validation=True,
            trust_repo=True,
        )
        encoder = encoder.to(device).eval()
    except Exception:
        return None
    tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    def encode(img: Image.Image) -> torch.Tensor:
        x = tf(img.convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            z = encoder(x)
        return F.normalize(z, dim=1)[0].cpu()

    return encode, "dinov2"


class PrototypeAuthenticator:
    """Cosine-prototype genuine/fake classifier (CLIP / DINOv2 / MobileNet)."""

    def __init__(self, device: torch.device | None = None):
        self.device = device or DEVICE
        self.backend = "mobilenet"
        self._encode_fn = None
        clip = _try_clip(self.device)
        dino = _try_dinov2(self.device) if clip is None else None
        if clip is not None:
            self._encode_fn, self.backend = clip
        elif dino is not None:
            self._encode_fn, self.backend = dino
        else:
            enc = _MobileNetEncoder().to("cpu").eval()
            self._encode_fn = enc.encode_pil
            self.backend = "mobilenet_imagenet"
        self.real_proto: torch.Tensor | None = None
        self.fake_proto: torch.Tensor | None = None
        self.loaded = self.load()

    def encode_bgr(self, crop_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return self._encode_fn(Image.fromarray(rgb)).detach().cpu().float()

    def build(self, max_notes_per_class: int = 80, views_per_note: int = 2) -> dict:
        notes = list_jaaltaka_notes()
        buckets = {0: [], 1: []}
        for note_dir, label in notes:
            if len(buckets[label]) >= max_notes_per_class:
                continue
            views = sorted(p for p in note_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            embs = []
            for p in views[:views_per_note]:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                embs.append(self.encode_bgr(img))
            if embs:
                buckets[label].append(torch.stack(embs).mean(0))
        if not buckets[0] or not buckets[1]:
            raise RuntimeError("JaalTaka real/fake notes missing — cannot build prototypes")
        self.fake_proto = F.normalize(torch.stack(buckets[0]).mean(0), dim=0)
        self.real_proto = F.normalize(torch.stack(buckets[1]).mean(0), dim=0)
        blob = {
            "backend": self.backend,
            "real": self.real_proto,
            "fake": self.fake_proto,
            "n_real": len(buckets[1]),
            "n_fake": len(buckets[0]),
        }
        PROTO_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, PROTO_WEIGHTS)
        self.loaded = True
        return blob

    def load(self) -> bool:
        if not PROTO_WEIGHTS.is_file():
            return False
        blob = torch.load(PROTO_WEIGHTS, map_location="cpu", weights_only=False)
        self.real_proto = blob["real"].float()
        self.fake_proto = blob["fake"].float()
        return True

    def predict(self, crop_bgr: np.ndarray) -> dict:
        if self.real_proto is None or self.fake_proto is None:
            return {"label": "unknown", "genuine_prob": 0.5, "backend": self.backend, "loaded": False}
        z = F.normalize(self.encode_bgr(crop_bgr), dim=0)
        sim_real = float(torch.dot(z, self.real_proto))
        sim_fake = float(torch.dot(z, self.fake_proto))
        # temperature softmax over the two prototype similarities
        logits = torch.tensor([sim_fake, sim_real]) * 8.0
        prob = torch.softmax(logits, dim=0)
        genuine = float(prob[1])
        return {
            "label": "genuine" if genuine >= 0.5 else "counterfeit",
            "genuine_prob": genuine,
            "sim_real": sim_real,
            "sim_fake": sim_fake,
            "backend": self.backend,
            "loaded": True,
        }
