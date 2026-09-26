"""Curriculum over prefix length, plus higher loss weight on hard notes."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import NovelAuthNet


class AdaptivePrefixCurriculum(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = NovelAuthNet(algo="apc")
        self.max_k = 1

    def maybe_advance(self, batch_acc: float, epoch: int) -> None:
        """One extra view per epoch. An easy batch no longer skips ahead to 6 views."""
        del batch_acc
        self.max_k = min(6, max(1, int(epoch)))

    def forward(self, views: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.net(views, mask)

    def loss(self, out: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        per = F.cross_entropy(out["logits"], labels, reduction="none")
        # Online difficulty: detached error. Hard notes (wrong) get weight 2.
        wrong = (out["logits"].argmax(1) != labels).float().detach()
        weight = 1.0 + 0.25 * wrong
        return (per * weight).mean()
