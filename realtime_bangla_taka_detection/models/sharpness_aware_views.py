"""Same classifier as the other runs. The trainer applies one SAM step.

View choice stays the random prefix used by the shared trainer. SAM perturbs
the weights, not the pixels.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.backbone import NovelAuthNet


class SharpnessAwareViewSelection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = NovelAuthNet(algo="savs")

    def forward(self, views: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.net(views, mask)

    def loss(self, out: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(out["logits"], labels)
