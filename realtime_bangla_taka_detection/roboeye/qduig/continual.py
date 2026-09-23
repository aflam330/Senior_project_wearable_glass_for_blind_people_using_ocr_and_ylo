"""Novelty 10 — Quality-Weighted Elastic Replay with View-Prototype Consolidation (QWER-VPC).

Beyond naive fine-tuning, reservoir replay, and EWC:
  * store a small set of *view prototypes* (mean embeddings of high-usable
    views per class), not raw images;
  * elastic penalty is Fisher-weighted *and* scaled by the prototype
    quality so parameters that served high-quality views are stiffer;
  * replay mixes prototype classification with a quality-weighted CE
    on a tiny stored buffer.

Compared to EWC: penalty is quality-modulated and applied in prototype
space as well as parameter space.
Compared to replay: buffer is prototype-first; raw replay is secondary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PrototypeBank:
    genuine: torch.Tensor  # (P, D)
    counterfeit: torch.Tensor
    genuine_q: torch.Tensor  # (P,)
    counterfeit_q: torch.Tensor


def update_prototypes(
    bank: PrototypeBank | None,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    usable: torch.Tensor,
    max_per_class: int = 32,
) -> PrototypeBank:
    """Keep the highest-usable embeddings per class as prototypes."""
    device = embeddings.device
    if bank is None:
        bank = PrototypeBank(
            genuine=embeddings.new_zeros(0, embeddings.size(-1)),
            counterfeit=embeddings.new_zeros(0, embeddings.size(-1)),
            genuine_q=embeddings.new_zeros(0),
            counterfeit_q=embeddings.new_zeros(0),
        )

    def _merge(old_z, old_q, z, q):
        if z.numel() == 0:
            return old_z, old_q
        z_all = torch.cat([old_z, z], dim=0)
        q_all = torch.cat([old_q, q], dim=0)
        k = min(max_per_class, z_all.size(0))
        top = torch.topk(q_all, k=k).indices
        return z_all[top], q_all[top]

    for lab, name in ((1, "genuine"), (0, "counterfeit")):
        m = labels == lab
        z, q = embeddings[m], usable[m]
        new_z, new_q = _merge(getattr(bank, name), getattr(bank, f"{name}_q"), z, q)
        setattr(bank, name, new_z.detach())
        setattr(bank, f"{name}_q", new_q.detach())
    return bank


def prototype_loss(fused: torch.Tensor, labels: torch.Tensor, bank: PrototypeBank) -> torch.Tensor:
    """Pull fused embeddings toward same-class quality-weighted prototypes."""
    if bank.genuine.numel() == 0 or bank.counterfeit.numel() == 0:
        return fused.new_zeros(())
    proto = []
    for i, y in enumerate(labels.tolist()):
        zs = bank.genuine if y == 1 else bank.counterfeit
        qs = bank.genuine_q if y == 1 else bank.counterfeit_q
        w = qs / qs.sum().clamp_min(1e-8)
        proto.append((w.unsqueeze(-1) * zs).sum(dim=0))
    target = torch.stack(proto, dim=0).to(fused.device)
    return F.mse_loss(fused, target.detach())


def quality_weighted_ewc(
    model: nn.Module,
    fisher: dict[str, torch.Tensor],
    star: dict[str, torch.Tensor],
    quality_scale: float,
) -> torch.Tensor:
    penalty = None
    for name, p in model.named_parameters():
        if name not in fisher or not p.requires_grad:
            continue
        term = (fisher[name].to(p.device) * (p - star[name].to(p.device)).pow(2)).sum()
        penalty = term if penalty is None else penalty + term
    if penalty is None:
        return next(model.parameters()).new_zeros(())
    return quality_scale * penalty


@torch.no_grad()
def fisher_diag(model: nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], device) -> dict[str, torch.Tensor]:
    """Diagonal Fisher on provided (views, mask, y) batches."""
    model.eval()
    fisher: dict[str, torch.Tensor] = {}
    n = 0
    for views, mask, y in batches:
        model.zero_grad(set_to_none=True)
        out = model(views.to(device), mask.to(device))
        loss = F.cross_entropy(out["logits"], y.to(device))
        loss.backward()
        n += 1
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g2 = p.grad.detach().pow(2)
            fisher[name] = g2 if name not in fisher else fisher[name] + g2
    for k in fisher:
        fisher[k] = fisher[k] / max(n, 1)
    return fisher


def snapshot_params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
