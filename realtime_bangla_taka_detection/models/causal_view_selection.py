"""Score each view by the change in the logit when that view is removed.

The intervention is computed inside the training batch. Test-time selection
uses the learned score and does not read labels.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.backbone import NovelAuthNet


class CausalViewSelection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = NovelAuthNet(algo="cvs")

    def forward(self, views, mask):
        return self.net(views, mask)

    def loss(self, views, mask, labels) -> torch.Tensor:
        out = self.forward(views, mask)
        auth = F.cross_entropy(out["logits"], labels)
        with torch.no_grad():
            base = out["logits"][:, 1] - out["logits"][:, 0]
            effects = []
            for k in range(views.size(1)):
                sub = mask.clone()
                sub[:, k] = 0
                # keep at least one view
                empty = sub.sum(1) == 0
                sub[empty, k] = mask[empty, k]
                alt = self.forward(views, sub)["logits"]
                effects.append((base - (alt[:, 1] - alt[:, 0])).abs())
            target = torch.stack(effects, dim=1)
            target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pred = out["causal"].masked_fill(mask == 0, 0)
        pred = pred / pred.sum(dim=1, keepdim=True).clamp_min(1e-6)
        align = F.kl_div(pred.clamp_min(1e-8).log(), target, reduction="batchmean")
        return auth + 0.05 * align
