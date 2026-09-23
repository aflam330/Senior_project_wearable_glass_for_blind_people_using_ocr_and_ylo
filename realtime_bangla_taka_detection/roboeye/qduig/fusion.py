"""Novelty 7 — Hypernetwork-Gated Evidence Fusion (HGEF).

Not averaging and not a flat concatenation classifier.
A tiny hypernetwork reads auxiliary evidence
(quality, entropy, residual volume, predicted IG, view count)
and produces a *feature-wise gate* plus a residual shift that
reweights the fused embedding before the authentication head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HypernetworkGatedEvidenceFusion(nn.Module):
    CTX_DIM = 5  # mean_q, entropy, residual_vol, pred_ig, n_views_norm

    def __init__(self, feat_dim: int, hidden: int = 96):
        super().__init__()
        self.hyper = nn.Sequential(
            nn.Linear(self.CTX_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.gate = nn.Linear(hidden, feat_dim)
        self.shift = nn.Linear(hidden, feat_dim)
        self.logit_bias = nn.Linear(hidden, 2)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, 2),
        )

    def context(
        self,
        mean_q: torch.Tensor,
        entropy: torch.Tensor,
        residual_vol: torch.Tensor,
        pred_ig: torch.Tensor,
        n_views_norm: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack([mean_q, entropy, residual_vol, pred_ig, n_views_norm], dim=-1)

    def forward(self, fused: torch.Tensor, ctx: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.hyper(ctx)
        gate = torch.sigmoid(self.gate(h))
        shift = torch.tanh(self.shift(h))
        fused_h = gate * fused + (1.0 - gate) * shift
        logits = self.classifier(fused_h) + 0.15 * self.logit_bias(h)
        return {
            "fused_h": fused_h,
            "logits": logits,
            "gate": gate,
            "shift": shift,
        }
