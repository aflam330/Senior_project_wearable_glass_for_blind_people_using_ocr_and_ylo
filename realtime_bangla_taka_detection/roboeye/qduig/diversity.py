"""Novelty 2 — Chordal-Volume Redundancy (CVR) and Maximal Residual Volume Acquisition.

Difference from cosine-similarity diversity:
a candidate is scored by the *residual energy after orthogonal projection
onto the affine span of already selected embeddings*, times the
incremental log-determinant of the selected Gram matrix (parallelepiped
volume). Cosine only measures pairwise angle; CVR measures how much new
volume a view adds to the selected set.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _unit(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return z / z.norm(dim=-1, keepdim=True).clamp_min(eps)


def gram_schmidt_residual(selected: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Residual of ``candidate`` (B, D) after projection onto rows of ``selected`` (B, K, D)."""
    if selected.numel() == 0 or selected.size(1) == 0:
        return candidate
    basis = selected
    residual = candidate
    for k in range(basis.size(1)):
        b = basis[:, k]
        bn = b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        residual = residual - ((residual * b).sum(dim=-1, keepdim=True) / bn.pow(2)) * b
    return residual


def logdet_gram(z: torch.Tensor, mask: torch.Tensor | None = None, tau: float = 1e-2) -> torch.Tensor:
    """log det(G + τI) for unit-normalized embeddings. Always fp32. z: (B, V, D)."""
    z32 = z.float()
    u = _unit(z32)
    if mask is not None:
        u = u * mask.unsqueeze(-1).to(u.dtype)
    gram = torch.matmul(u, u.transpose(1, 2))
    v = z.size(1)
    eye = torch.eye(v, device=z.device, dtype=torch.float32).unsqueeze(0)
    _sign, logabs = torch.linalg.slogdet(gram + tau * eye)
    return logabs.clamp(-20.0, 20.0).to(z.dtype)


def chordal_volume_scores(
    z: torch.Tensor,
    selected_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    volume_tau: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Score each candidate view given the currently selected set.

    selected_mask, candidate_mask: (B, V) 0/1
    returns per-view residual energy, redundancy, diversity score (B, V)
    """
    b, v, d = z.shape
    residual_energy = torch.zeros(b, v, device=z.device, dtype=z.dtype)
    redundancy = torch.ones(b, v, device=z.device, dtype=z.dtype)
    diversity = torch.zeros(b, v, device=z.device, dtype=z.dtype)
    vol_before = logdet_gram(z, selected_mask)

    for j in range(v):
        cand = z[:, j]
        sel = []
        for k in range(v):
            if k == j:
                continue
            # gather selected others — done via mask multiply below
            sel.append(z[:, k : k + 1] * selected_mask[:, k].view(b, 1, 1).to(z.dtype))
        if sel:
            selected = torch.cat(sel, dim=1)
            # drop all-zero (unselected) columns by zeroing; Gram-Schmidt ignores zeros
            residual = gram_schmidt_residual(selected, cand)
        else:
            residual = cand
        energy = residual.norm(dim=-1)
        base = cand.norm(dim=-1).clamp_min(1e-8)
        residual_energy[:, j] = energy
        redundancy[:, j] = (1.0 - energy / base).clamp(0.0, 1.0)

        trial_mask = selected_mask.clone()
        trial_mask[:, j] = 1
        vol_after = logdet_gram(z, trial_mask)
        delta = (vol_after - vol_before).clamp(min=-5.0, max=5.0)
        diversity[:, j] = energy * (1.0 + volume_tau * torch.relu(delta))

    diversity = diversity * candidate_mask.to(diversity.dtype)
    redundancy = torch.where(candidate_mask.bool(), redundancy, torch.ones_like(redundancy))
    return {
        "residual_energy": residual_energy,
        "redundancy": redundancy,
        "diversity": diversity,
        "volume_selected": vol_before,
    }


def select_next_diverse(z: torch.Tensor, selected_mask: torch.Tensor, available_mask: torch.Tensor) -> torch.Tensor:
    """Return index (B,) of the candidate with maximal residual volume."""
    scores = chordal_volume_scores(z, selected_mask, available_mask)["diversity"]
    scores = scores.masked_fill(available_mask == 0, -1e9)
    return scores.argmax(dim=1)


def pairwise_mean_redundancy(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Scalar mean redundancy of the current set (B,)."""
    u = _unit(z)
    sim = torch.matmul(u, u.transpose(1, 2)).clamp(-1.0, 1.0)
    m = mask.unsqueeze(2) * mask.unsqueeze(1)
    eye = torch.eye(z.size(1), device=z.device, dtype=z.dtype).unsqueeze(0)
    m = m * (1.0 - eye)
    # residual-style: high |cos| => high redundancy
    red = (sim.abs() * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1.0)
    return red
