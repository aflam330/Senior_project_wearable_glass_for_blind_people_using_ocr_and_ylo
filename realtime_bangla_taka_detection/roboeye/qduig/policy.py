"""Novelty 5 — Cost-Regularized Information-Quality Policy (CRIQP).

Decides STOP or REQUEST NEXT VIEW without a raw confidence threshold.
Utility of a candidate:
  U = pred_IG · usable · (1 − redundancy) − λ · ΔC
where ΔC = α·1 + β·Δlatency + γ·Δenergy.

If max U ≤ 0, stop. Never force early stop when a useful view remains.
α, β, γ, λ are chosen on the validation Pareto frontier, never on test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

PolicyName = Literal[
    "random",
    "fixed_order",
    "quality_only",
    "confidence_only",
    "diversity_only",
    "quality_diversity",
    "uncertainty_quality",
    "uncertainty_diversity",
    "full_proposed",
]


@dataclass
class CostWeights:
    alpha: float = 1.0
    beta: float = 0.05
    gamma: float = 0.0  # energy typically NOT_MEASURED → 0
    lamb: float = 0.08


@dataclass
class AcquisitionDecision:
    action: str  # STOP | REQUEST
    next_index: int | None
    utility: float
    predicted_ig: float
    current_uncertainty: float
    expected_post_view_uncertainty: float
    n_views: int
    cost: float
    selected_indices: list[int] = field(default_factory=list)


def incremental_cost(alpha: float, beta: float, gamma: float, latency_proxy: float = 1.0, energy_proxy: float = 0.0) -> float:
    return alpha * 1.0 + beta * latency_proxy + gamma * energy_proxy


def criqp_utilities(
    pred_ig: torch.Tensor,
    usable: torch.Tensor,
    redundancy: torch.Tensor,
    available: torch.Tensor,
    cost_inc: float,
    lamb: float,
) -> torch.Tensor:
    """U (B, V) for remaining candidates."""
    u = pred_ig * usable * (1.0 - redundancy) - lamb * cost_inc
    return u.masked_fill(available == 0, -1e9)


def decide(
    utilities: torch.Tensor,
    pred_ig: torch.Tensor,
    entropy: torch.Tensor,
    selected: list[int],
    max_views: int,
) -> AcquisitionDecision:
    """utilities: (V,) one note."""
    n = len(selected)
    if n >= max_views:
        return AcquisitionDecision(
            action="STOP",
            next_index=None,
            utility=0.0,
            predicted_ig=0.0,
            current_uncertainty=float(entropy.item()),
            expected_post_view_uncertainty=float(entropy.item()),
            n_views=n,
            cost=float(n),
            selected_indices=list(selected),
        )
    best = int(torch.argmax(utilities).item())
    u = float(utilities[best].item())
    ig = float(pred_ig[best].item())
    h = float(entropy.item())
    if u <= 0.0:
        return AcquisitionDecision(
            action="STOP",
            next_index=None,
            utility=u,
            predicted_ig=ig,
            current_uncertainty=h,
            expected_post_view_uncertainty=max(h - ig, 0.0),
            n_views=n,
            cost=float(n),
            selected_indices=list(selected),
        )
    return AcquisitionDecision(
        action="REQUEST",
        next_index=best,
        utility=u,
        predicted_ig=ig,
        current_uncertainty=h,
        expected_post_view_uncertainty=max(h - ig, 0.0),
        n_views=n + 1,
        cost=float(n + 1),
        selected_indices=list(selected) + [best],
    )


def score_candidates(
    name: PolicyName,
    usable: torch.Tensor,
    confidence: torch.Tensor,
    diversity: torch.Tensor,
    entropy_per_view: torch.Tensor,
    utilities: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    """Return a (V,) score used to pick the next view for a named policy."""
    if name == "random":
        s = torch.rand_like(usable)
    elif name == "fixed_order":
        s = -torch.arange(usable.numel(), device=usable.device, dtype=usable.dtype)
    elif name == "quality_only":
        s = usable
    elif name == "confidence_only":
        s = confidence
    elif name == "diversity_only":
        s = diversity
    elif name == "quality_diversity":
        s = usable * diversity
    elif name == "uncertainty_quality":
        s = entropy_per_view * usable
    elif name == "uncertainty_diversity":
        s = entropy_per_view * diversity
    elif name == "full_proposed":
        s = utilities
    else:
        raise ValueError(name)
    return s.masked_fill(available == 0, -1e9)


def pareto_front(points: list[dict], acc_key: str = "accuracy", cost_key: str = "average_cost") -> list[dict]:
    """Undominated points: higher acc, lower cost. Validation only when selecting λ."""
    front = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q[acc_key] >= p[acc_key] and q[cost_key] <= p[cost_key] and (
                q[acc_key] > p[acc_key] or q[cost_key] < p[cost_key]
            ):
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda r: r[cost_key])
    return front


def select_operating_point(front: list[dict], min_val_acc: float) -> dict | None:
    """Choose the cheapest Pareto point that meets a validation accuracy floor."""
    feasible = [p for p in front if p["accuracy"] >= min_val_acc]
    if not feasible:
        return None
    return min(feasible, key=lambda p: p["average_cost"])
