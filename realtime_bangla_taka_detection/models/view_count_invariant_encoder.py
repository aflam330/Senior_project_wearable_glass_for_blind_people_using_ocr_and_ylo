"""View-count invariant encoder: shared view encoder plus a set transformer.

Self-attention over views has no positional encoding, so the pooled vector
is invariant to view order.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import NovelAuthNet, balanced_ce


class ViewCountInvariantEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = NovelAuthNet(algo="vcie")

    def forward(self, views: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.net(views, mask)

    def loss(self, out: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        return balanced_ce(out["logits"], labels)
