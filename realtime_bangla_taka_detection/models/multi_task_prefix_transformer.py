"""Shared set encoder with an authenticity head and a proxy-quality head.

Denomination and emotion labels are not in the JaalTaka note records used
here, so those tasks are not trained and must be reported NOT_MEASURED.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.backbone import NovelAuthNet, balanced_ce
from models.security_feature_quality import image_security_proxies


class MultiTaskPrefixTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = NovelAuthNet(algo="mtpt")

    def forward(self, views, mask):
        out = self.net(views, mask)
        out["proxies"] = image_security_proxies(views, mask)
        return out

    def loss(self, out, labels):
        auth = balanced_ce(out["logits"], labels)
        weight = float(getattr(self, "aux_weight", 0.1))
        if weight == 0.0:
            return auth
        pred = out["sfaq_from_embed"]
        if bool(getattr(self, "aux_stopgrad", False)):
            pred = pred.detach()
            # Detached prediction has no gradient. Recompute the head on detached features.
            pred = self.net.aux_quality(out["fused"].detach())
        aux = F.mse_loss(torch.sigmoid(pred), out["proxies"].detach().clamp(0, 1))
        return auth + weight * aux
