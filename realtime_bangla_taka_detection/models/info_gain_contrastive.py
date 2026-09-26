"""Contrastive estimate of information in the next view. No entropy labels."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.backbone import NovelAuthNet


class ContrastiveInfoGain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = NovelAuthNet(algo="igcr")

    def forward(self, views: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.net(views, mask)

    def loss(self, out: dict[str, torch.Tensor], labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        auth = F.cross_entropy(out["logits"], labels)
        z = F.normalize(out["ig"], dim=-1)
        # First real view vs second real view. Positive pair is same note.
        counts = mask.sum(1)
        if int((counts >= 2).sum()) < 2:
            return auth
        a = z[:, 0]
        b = z[:, 1]
        logits = a @ b.T / 0.2
        target = torch.arange(a.size(0), device=a.device)
        return auth + 0.1 * F.cross_entropy(logits, target)
