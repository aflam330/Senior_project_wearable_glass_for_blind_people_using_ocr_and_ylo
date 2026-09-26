"""Classifier loss plus a support/query prototype loss inside the batch.

The query half is scored by distance to class prototypes from the support half.
Test-time prediction uses the classifier. It does not read query labels.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.backbone import NovelAuthNet


class MetaAdaptiveViewTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = NovelAuthNet(algo="mavt")

    def forward(self, views: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.net(views, mask)

    def loss(self, out: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(out["logits"], labels)
        fused = out["fused"]
        if labels.size(0) < 4:
            return ce
        mid = labels.size(0) // 2
        support, query = fused[:mid], fused[mid:]
        y_s, y_q = labels[:mid], labels[mid:]
        protos = []
        for cls in (0, 1):
            chosen = support[y_s == cls]
            protos.append(chosen.mean(0) if chosen.size(0) else support.mean(0))
        dist = torch.cdist(query, torch.stack(protos, 0))
        proto = F.nll_loss(F.log_softmax(-dist, dim=1), y_q)
        return ce + proto
