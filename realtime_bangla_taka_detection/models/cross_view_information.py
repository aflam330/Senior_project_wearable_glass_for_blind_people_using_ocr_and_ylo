"""Authenticity plus a small information bottleneck on the fused vector.

The bottleneck is trained with a KL term. It is not a measured mutual-information estimator.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.backbone import NovelAuthNet


class CrossViewInformationSharing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = NovelAuthNet(algo="cris")

    def forward(self, views: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.net(views, mask)

    def loss(self, out: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(out["logits"], labels) + 0.01 * out["ib_kl"]
