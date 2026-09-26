"""Fuse views with a learned gate of uncertainty, quality, and diversity."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.backbone import NovelAuthNet


class UncertaintyGatedFusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = NovelAuthNet(algo="ugf")

    def forward(self, views, mask):
        return self.net(views, mask)

    def loss(self, out, labels):
        return F.cross_entropy(out["logits"], labels)
