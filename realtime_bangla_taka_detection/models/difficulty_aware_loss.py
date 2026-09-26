"""Up-weight notes the current model still gets wrong. Online, not a second dataset."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.backbone import NovelAuthNet


class NoteDifficultyAwareLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = NovelAuthNet(algo="ndal")

    def forward(self, views, mask):
        return self.net(views, mask)

    def loss(self, out, labels):
        per = F.cross_entropy(out["logits"], labels, reduction="none")
        pt = torch.exp(-per.detach())
        focal = (1.0 - pt).pow(1.5) * per
        diff = focal.detach()
        weight = (diff / diff.mean().clamp_min(1e-6)).clamp(0.5, 2.0)
        return (focal * weight).mean()
