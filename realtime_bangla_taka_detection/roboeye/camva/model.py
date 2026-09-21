"""CAMVA network: per-view CNN+ViT encoder, quality head, attention fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from ..authenticity import IMG_SIZE, TinyViT


class ViewEncoder(nn.Module):
    def __init__(self, freeze_cnn: bool = True):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.cnn = backbone.features
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        self.cnn_dim = 576
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False
        self.vit = TinyViT(img_size=IMG_SIZE)
        self.out_dim = self.cnn_dim + 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        cnn = self.cnn_pool(self.cnn(x)).flatten(1)
        vit = self.vit(x)
        return torch.cat([cnn, vit], dim=1)


class CAMVANet(nn.Module):
    """Variable-view quality-aware attention fusion + binary classifier.

    views: (B, V, C, H, W)
    mask:  (B, V) 1 = real view, 0 = pad
    """

    def __init__(self, freeze_cnn: bool = True, fusion: str = "quality_attention"):
        super().__init__()
        if fusion not in {"mean", "attention", "quality_attention"}:
            raise ValueError(fusion)
        self.fusion = fusion
        self.encoder = ViewEncoder(freeze_cnn=freeze_cnn)
        dim = self.encoder.out_dim
        self.quality_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        self.attn_head = nn.Sequential(
            nn.Linear(dim + 1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, 2),
        )

    def encode_views(self, views: torch.Tensor) -> torch.Tensor:
        b, v, c, h, w = views.shape
        return self.encoder(views.reshape(b * v, c, h, w)).reshape(b, v, -1)

    def fuse(self, z: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return F (B, D), weights (B, V), quality (B, V)."""
        q_logit = self.quality_head(z).squeeze(-1)
        q = torch.sigmoid(q_logit)
        if self.fusion == "mean":
            a = mask.to(z.dtype)
        elif self.fusion == "attention":
            a = self.attn_head(torch.cat([z, torch.zeros_like(q).unsqueeze(-1)], dim=-1)).squeeze(-1)
        else:
            a = self.attn_head(torch.cat([z, q.unsqueeze(-1)], dim=-1)).squeeze(-1)
        a = a.masked_fill(mask == 0, -1e9)
        w = torch.softmax(a, dim=1)
        w = w * mask.to(w.dtype)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        fused = (w.unsqueeze(-1) * z).sum(dim=1)
        return fused, w, q

    def forward(self, views: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if mask is None:
            mask = torch.ones(views.shape[:2], device=views.device, dtype=torch.long)
        z = self.encode_views(views)
        fused, w, q = self.fuse(z, mask)
        logits = self.classifier(fused)
        return {"logits": logits, "weights": w, "quality": q, "features": z, "fused": fused}

    def classify_features(self, fused: torch.Tensor) -> torch.Tensor:
        return self.classifier(fused)
