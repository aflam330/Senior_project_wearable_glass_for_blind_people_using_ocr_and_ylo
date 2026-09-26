"""Train on the given view order and on one reversed order.

The reversed order is a fixed adversarial permutation of the same images.
Test-time forward uses the order it is given and does not search permutations.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.backbone import NovelAuthNet


class PrefixRobustAdversarial(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = NovelAuthNet(algo="pravt")

    def forward(self, views: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.net(views, mask)

    def loss(self, views: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor):
        out = self.forward(views, mask)
        auth = F.cross_entropy(out["logits"], labels)
        flipped = self.forward(views.flip(1), mask.flip(1))
        adv = F.cross_entropy(flipped["logits"], labels)
        return auth + adv, out
