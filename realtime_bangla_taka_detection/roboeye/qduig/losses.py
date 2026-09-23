"""Novelty 8 — Composite Q-DUIG loss.

L = L_auth + λq L_quality + λu L_uncertainty + λd L_diversity
    + λg L_info_gain + λc L_cost

Each term is defined below. Coefficients are taken from config
(chosen on validation protocol, never on test).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .info_gain import binary_entropy_t


@dataclass
class LossWeights:
    auth: float = 1.0
    quality: float = 0.25
    uncertainty: float = 0.15
    diversity: float = 0.10
    info_gain: float = 0.20
    cost: float = 0.05


def authentication_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """L_auth: standard note-level cross-entropy."""
    return F.cross_entropy(logits, y)


def quality_consistency_loss(usable: torch.Tensor, mask: torch.Tensor, single_agree: torch.Tensor) -> torch.Tensor:
    """L_quality: usable score should track leave-one-out agreement with the fused decision.

    ``single_agree`` is detached (0/1) and computed on the current minibatch
    from train/val predictions only.
    """
    m = mask.to(usable.dtype)
    return (m * (usable - single_agree).abs()).sum() / m.sum().clamp_min(1.0)


def uncertainty_loss(entropy: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    """L_uncertainty: low entropy when correct, higher entropy when wrong.

    Encourages the model to be uncertain on its own mistakes (train/val).
    """
    ent = torch.nan_to_num(entropy, nan=0.693147)
    target = torch.where(correct.bool(), torch.zeros_like(ent), torch.full_like(ent, 0.693147))
    return F.mse_loss(ent, target)


def diversity_redundancy_loss(redundancy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L_diversity: penalise high pairwise redundancy among selected views."""
    m = mask.to(redundancy.dtype)
    return (m * redundancy).sum() / m.sum().clamp_min(1.0)


def info_gain_loss(pred_ig: torch.Tensor, realized_ig: torch.Tensor) -> torch.Tensor:
    """L_info_gain: MSE of surrogate vs realized ΔH (detached train/val rollouts)."""
    return F.mse_loss(pred_ig, realized_ig)


def acquisition_cost_loss(
    n_views: torch.Tensor,
    latency_proxy: torch.Tensor,
    energy_proxy: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    max_views: float = 6.0,
) -> torch.Tensor:
    """L_cost: C = α n_views + β latency + γ energy, normalised by max cost."""
    cost = alpha * n_views + beta * latency_proxy + gamma * energy_proxy
    denom = alpha * max_views + beta + gamma + 1e-8
    return (cost / denom).mean()


def composite_loss(terms: dict[str, torch.Tensor], weights: LossWeights) -> tuple[torch.Tensor, dict[str, float]]:
    total = (
        weights.auth * terms["auth"]
        + weights.quality * terms["quality"]
        + weights.uncertainty * terms["uncertainty"]
        + weights.diversity * terms["diversity"]
        + weights.info_gain * terms["info_gain"]
        + weights.cost * terms["cost"]
    )
    logged = {k: float(v.detach().item()) for k, v in terms.items()}
    logged["total"] = float(total.detach().item())
    return total, logged
