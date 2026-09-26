"""Prefix length is sampled by the trainer. This head also predicts that length.

The length target is the mask the trainer built. It is not a label from the test split.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.backbone import NovelAuthNet


class ViewCountDistributionTraining(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = NovelAuthNet(algo="vat")

    def forward(self, views: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.net(views, mask)

    def loss(self, out: dict[str, torch.Tensor], labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        auth = F.cross_entropy(out["logits"], labels)
        k = mask.sum(1).long().clamp(1, 6) - 1
        return auth + 0.2 * F.cross_entropy(out["k_logits"], k)
