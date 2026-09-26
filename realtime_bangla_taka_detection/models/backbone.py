"""Shared frozen-CNN view encoder used by the novel algorithms."""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from roboeye.camva.model import ViewEncoder

SET_ALGOS = {"vcie", "mtpt", "sfpl"}


class SetBlock(nn.Module):
    """One permutation-equivariant self-attention block over views."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # Attention in fp32. fp16 multi-head attention was producing a constant predictor.
        z32 = z.float()
        h, _ = self.attn(z32, z32, z32, key_padding_mask=key_padding_mask, need_weights=False)
        h = h.to(dtype=z.dtype)
        z = self.norm1(z + h)
        return self.norm2(z + self.ff(z))


class NovelAuthNet(nn.Module):
    """Variable-view authenticator. ``algo`` selects the fusion or extra head.

    views: (B, V, C, H, W), mask: (B, V) with 1 = real view.
    Label 1 = genuine, 0 = counterfeit. No test labels enter this module.
    """

    def __init__(self, algo: str = "vcie", freeze_cnn: bool = True):
        super().__init__()
        self.algo = algo
        self.encoder = ViewEncoder(freeze_cnn=freeze_cnn)
        dim = self.encoder.out_dim
        self.dim = dim
        self.set_blocks = nn.ModuleList([SetBlock(dim) for _ in range(2)])
        self.pool_query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.pool_query, std=0.02)
        self.pool_attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.quality = nn.Linear(dim, 1)
        self.uncertainty = nn.Linear(dim, 1)
        self.gate = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
        self.policy = nn.Sequential(nn.Linear(dim + 2, 64), nn.GELU(), nn.Linear(64, 6))
        self.sfaq = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 1))
        self.ig_proj = nn.Linear(dim, 64)
        self.difficulty = nn.Linear(dim, 1)
        self.causal_score = nn.Linear(dim, 1)
        self.classifier = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))
        self.aux_quality = nn.Linear(dim, 6)
        self.pool_norm = nn.LayerNorm(dim)
        if algo == "cris":
            self.ib_mu = nn.Linear(dim, 128)
            self.ib_lv = nn.Linear(dim, 128)
            self.ib_dec = nn.Linear(128, dim)
        if algo == "vat":
            self.k_head = nn.Linear(dim, 6)

    def encode(self, views: torch.Tensor) -> torch.Tensor:
        b, v, c, h, w = views.shape
        return self.encoder(views.reshape(b * v, c, h, w)).reshape(b, v, -1)

    def _set_encode(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pad = mask == 0
        for block in self.set_blocks:
            z = block(z, pad)
        return z

    def _mean_pool(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask.to(z.dtype)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (w.unsqueeze(-1) * z).sum(dim=1)

    def pool(self, z: torch.Tensor, mask: torch.Tensor, gates: torch.Tensor | None = None) -> torch.Tensor:
        mean = self._mean_pool(z, mask)
        if self.algo == "vcie" and os.environ.get("NOVEL_SET_MODE", "residual") != "mean":
            q = self.pool_query.expand(z.size(0), -1, -1)
            pad = mask == 0
            attn, _ = self.pool_attn(q, z, z, key_padding_mask=pad, need_weights=False)
            return self.pool_norm(0.5 * (attn.squeeze(1) + mean))
        w = gates if gates is not None else mask.to(z.dtype)
        w = w.masked_fill(mask == 0, 0)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = (w.unsqueeze(-1) * z).sum(dim=1)
        # UGF, SFAQ, IGCR, and OGPD seed-42 checkpoints were trained without this norm.
        if self.algo in {"ugf", "sfaq", "igcr", "ogpd"}:
            return pooled
        return self.pool_norm(pooled)

    def forward(self, views: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        z = torch.nan_to_num(self.encode(views))
        if self.algo in SET_ALGOS and os.environ.get("NOVEL_SET_MODE", "residual") != "mean":
            # Residual keeps the CNN features that mean-pooling already classifies.
            z = z + self._set_encode(z, mask)
        q = torch.sigmoid(self.quality(z).squeeze(-1))
        u = torch.sigmoid(self.uncertainty(z).squeeze(-1))
        # pairwise |cos| as a cheap diversity proxy, no labels
        zn = torch.nn.functional.normalize(z, dim=-1)
        sim = torch.matmul(zn, zn.transpose(1, 2)).abs()
        eye = torch.eye(sim.size(-1), device=sim.device).unsqueeze(0)
        div = (1.0 - (sim * (1 - eye)).mean(dim=-1)).clamp(0, 1)
        g = torch.sigmoid(self.gate(torch.stack([u, q, div], dim=-1)).squeeze(-1))
        if self.algo == "ugf":
            fused = self.pool(z, mask, gates=g * mask.to(g.dtype))
        else:
            fused = self.pool(z, mask)
        ib_kl = fused.new_zeros(())
        if self.algo == "cris":
            mu = self.ib_mu(fused)
            lv = self.ib_lv(fused).clamp(-8, 8)
            if self.training:
                fused = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            else:
                fused = mu
            fused = self.ib_dec(fused)
            ib_kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        logits = self.classifier(fused)
        k_logits = self.k_head(fused) if self.algo == "vat" else logits.new_zeros(logits.size(0), 6)
        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=1)[:, 1],
            "z": z,
            "fused": fused,
            "quality": q,
            "uncertainty": u,
            "diversity": div,
            "gate": g,
            "policy_logits": self.policy(torch.cat([fused, q.mean(1, keepdim=True), u.mean(1, keepdim=True)], dim=-1)),
            "sfaq_from_embed": self.aux_quality(fused),
            "difficulty": torch.sigmoid(self.difficulty(fused).squeeze(-1)),
            "causal": torch.sigmoid(self.causal_score(z).squeeze(-1)),
            "ig": self.ig_proj(z),
            "ib_kl": ib_kl,
            "k_logits": k_logits,
        }


def balanced_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency cross entropy. Weights come from the batch, not the test split."""
    counts = torch.bincount(labels.detach(), minlength=2).float().clamp_min(1.0)
    weight = (counts.sum() / (2.0 * counts)).to(logits.device)
    ce = F.cross_entropy(logits, labels, reduction="none")
    if os.environ.get("NOVEL_FOCAL", "0") == "1":
        pt = torch.exp(-ce.detach())
        ce = (1.0 - pt).pow(2) * ce
    return (ce * weight[labels]).mean()
