"""Novelty 4 — Prefix-Conditioned Residual Information-Gain Surrogate (PCR-IG).

Learned stand-in for expected binary-entropy reduction of adding one view.
Trained only on train/val rollouts (realized ΔH from the model's own
prefix→prefix+1 transitions). Test labels and test outcomes are never used
as targets.

Architecture: MLP over [fused_prefix, candidate_z, H, confidence,
usable, residual_energy, n_views_norm] → scalar predicted ΔH.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_entropy_t(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


class InfoGainSurrogate(nn.Module):
    """Predict expected information gain of a candidate view given the prefix state."""

    def __init__(self, feat_dim: int, hidden: int = 192):
        super().__init__()
        # fused + candidate + 5 scalars
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2 + 5, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        fused: torch.Tensor,
        candidate: torch.Tensor,
        entropy: torch.Tensor,
        confidence: torch.Tensor,
        usable: torch.Tensor,
        residual_energy: torch.Tensor,
        n_views_norm: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                fused,
                candidate,
                entropy.unsqueeze(-1),
                confidence.unsqueeze(-1),
                usable.unsqueeze(-1),
                residual_energy.unsqueeze(-1),
                n_views_norm.unsqueeze(-1),
            ],
            dim=-1,
        )
        # softplus keeps predicted IG ≥ 0 (information cannot increase entropy in expectation)
        return torch.nan_to_num(F.softplus(self.net(x).squeeze(-1)), nan=0.0)


def realized_information_gain(p_before: torch.Tensor, p_after: torch.Tensor) -> torch.Tensor:
    """H(p_before) − H(p_after). Detach before using as a regression target."""
    return torch.nan_to_num(binary_entropy_t(p_before) - binary_entropy_t(p_after)).detach()
