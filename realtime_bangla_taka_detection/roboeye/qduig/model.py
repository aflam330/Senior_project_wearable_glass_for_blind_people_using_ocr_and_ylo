"""Q-DUIG-CAMVA network: encoder + 11-component sequential authentication."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..camva.model import ViewEncoder
from .diversity import chordal_volume_scores, logdet_gram, pairwise_mean_redundancy
from .fusion import HypernetworkGatedEvidenceFusion
from .info_gain import InfoGainSurrogate, binary_entropy_t
from .quality import QualityFactorHead, ResidualSpectralQualityAttention, image_quality_factors


@dataclass
class QDUIGConfig:
    freeze_cnn: bool = True
    use_quality: bool = True
    use_uncertainty: bool = True
    use_diversity: bool = True
    use_info_gain: bool = True
    use_hgef: bool = True
    use_cost: bool = True
    use_redundancy: bool = True
    # When true, the cost term includes mean predictive entropy, which depends
    # on the logits. The mask-count cost alone has no parameter gradient.
    cost_from_entropy: bool = False
    dropout: float = 0.15
    max_views: int = 6
    view_self_gate: bool = False
    view_count_bn: bool = False


class QDUIGNet(nn.Module):
    """View encoder with quality, uncertainty, diversity, IG, and gated fusion."""

    def __init__(self, cfg: QDUIGConfig | None = None):
        super().__init__()
        self.cfg = cfg or QDUIGConfig()
        self.encoder = ViewEncoder(freeze_cnn=self.cfg.freeze_cnn)
        dim = self.encoder.out_dim
        self.quality_head = QualityFactorHead(dim)
        self.rsqa = ResidualSpectralQualityAttention(dim)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.ig = InfoGainSurrogate(dim)
        self.hgef = HypernetworkGatedEvidenceFusion(dim)
        self.plain_classifier = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, 2),
        )
        self.single_classifier = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        # FIX A: per-view gate. Softmax-over-views is a no-op at V=1; this still acts.
        self.view_gate = nn.Linear(dim, dim)
        nn.init.zeros_(self.view_gate.bias)
        nn.init.zeros_(self.view_gate.weight)
        # FIX B: separate BN per view-count so 1-view features are not normalized as a 6-view vector.
        self.k_bn = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(7)])
        self.mean_fusion = False

    def encode_views(self, views: torch.Tensor) -> torch.Tensor:
        b, v, c, h, w = views.shape
        return self.encoder(views.reshape(b * v, c, h, w)).reshape(b, v, -1)

    def _apply_k_bn(self, fused: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        counts = mask.sum(dim=1).long().clamp(1, 6)
        out = fused
        for k in range(1, 7):
            idx = (counts == k).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            bn = self.k_bn[k]
            if self.training and idx.numel() < 2:
                was = bn.training
                bn.eval()
                out[idx] = bn(fused[idx])
                bn.train(was)
            else:
                out[idx] = bn(fused[idx])
        return torch.nan_to_num(out)

    def _quality(self, z: torch.Tensor, views: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            factors = image_quality_factors(views)
        q = self.quality_head(z, factors)
        if not self.cfg.use_quality:
            q = dict(q)
            q["usable"] = torch.ones_like(q["usable"])
        return q

    def fuse_prefix(
        self,
        z: torch.Tensor,
        q: dict[str, torch.Tensor],
        mask: torch.Tensor,
        pred_ig_mean: torch.Tensor | None = None,
        fast_diversity: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.cfg.use_quality:
            fused, w, z_mod = self.rsqa(z, q["embed"], q["usable"], mask)
        else:
            a = mask.to(z.dtype)
            w = a / a.sum(dim=1, keepdim=True).clamp_min(1e-8)
            z_mod = z
            fused = (w.unsqueeze(-1) * z).sum(dim=1)

        use_fast = True if fast_diversity is None else fast_diversity
        if self.cfg.use_redundancy and self.cfg.use_diversity:
            if use_fast:
                # Fast train path: pairwise |cos| redundancy. Full CVR is used at eval/policy time.
                red_scalar = pairwise_mean_redundancy(z_mod, mask)
                redundancy = red_scalar.unsqueeze(1).expand(-1, z.size(1))
                residual = z_mod.norm(dim=-1)
                diversity = residual * (1.0 - redundancy)
                volume = logdet_gram(z_mod, mask)
            else:
                cvr = chordal_volume_scores(z_mod, mask, mask)
                redundancy = cvr["redundancy"]
                diversity = cvr["diversity"]
                residual = cvr["residual_energy"]
                volume = cvr["volume_selected"]
        else:
            redundancy = torch.zeros_like(q["usable"])
            diversity = torch.ones_like(q["usable"])
            residual = torch.ones_like(q["usable"])
            volume = torch.zeros(z.size(0), device=z.device)

        mean_q = (w * q["usable"]).sum(dim=1)
        n_norm = mask.sum(dim=1).to(z.dtype) / float(self.cfg.max_views)
        fused = torch.nan_to_num(fused)
        if self.cfg.view_count_bn:
            fused = self._apply_k_bn(fused, mask)
        logits_plain = self.plain_classifier(fused)
        prob = F.softmax(logits_plain, dim=1)[:, 1]
        entropy = binary_entropy_t(prob)
        if pred_ig_mean is None:
            pred_ig_mean = torch.zeros_like(mean_q)

        if self.cfg.use_hgef:
            ctx = self.hgef.context(mean_q, entropy, volume, pred_ig_mean, n_norm)
            hgef = self.hgef(fused, ctx)
            logits = hgef["logits"]
            fused_h = hgef["fused_h"]
        else:
            logits = logits_plain
            fused_h = fused
            hgef = {"gate": None, "shift": None}

        log_var = self.uncertainty_head(fused_h).squeeze(-1)
        return {
            "fused": fused,
            "fused_h": fused_h,
            "weights": w,
            "z_mod": z_mod,
            "logits": logits,
            "logits_plain": logits_plain,
            "prob": F.softmax(logits, dim=1)[:, 1],
            "entropy": binary_entropy_t(F.softmax(logits, dim=1)[:, 1]),
            "log_var": log_var,
            "redundancy": redundancy,
            "diversity": diversity,
            "residual_energy": residual,
            "volume": volume,
            "mean_q": mean_q,
            "n_views_norm": n_norm,
            "hgef": hgef,
        }

    def candidate_ig(
        self,
        fused: torch.Tensor,
        z: torch.Tensor,
        entropy: torch.Tensor,
        confidence: torch.Tensor,
        usable: torch.Tensor,
        residual_energy: torch.Tensor,
        n_views_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Predicted IG for every view as if it were the next candidate. (B, V)."""
        if not self.cfg.use_info_gain:
            return torch.zeros_like(usable)
        b, v, d = z.shape
        fused_e = fused.unsqueeze(1).expand(b, v, d)
        ent_e = entropy.unsqueeze(1).expand(b, v)
        conf_e = confidence.unsqueeze(1).expand(b, v)
        n_e = n_views_norm.unsqueeze(1).expand(b, v)
        return self.ig(
            fused_e.reshape(b * v, d),
            z.reshape(b * v, d),
            ent_e.reshape(b * v),
            conf_e.reshape(b * v),
            usable.reshape(b * v),
            residual_energy.reshape(b * v),
            n_e.reshape(b * v),
        ).reshape(b, v)

    def forward(self, views: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if mask is None:
            mask = torch.ones(views.shape[:2], device=views.device, dtype=torch.long)
        z = torch.nan_to_num(self.encode_views(views))
        if self.cfg.view_self_gate:
            z = z * (1.0 + torch.tanh(self.view_gate(z)))
        q = self._quality(z, views)
        packed = self.fuse_prefix(z, q, mask)
        conf = packed["prob"]
        pred_ig = self.candidate_ig(
            packed["fused_h"],
            packed["z_mod"],
            packed["entropy"],
            conf,
            q["usable"],
            packed["residual_energy"],
            packed["n_views_norm"],
        )
        packed["logits"] = torch.nan_to_num(packed["logits"])
        packed["prob"] = F.softmax(packed["logits"], dim=1)[:, 1]
        packed["entropy"] = binary_entropy_t(packed["prob"])
        packed["pred_ig"] = torch.nan_to_num(pred_ig)
        packed["quality"] = q["usable"]
        packed["quality_embed"] = q["embed"]
        packed["quality_factors"] = q["factors"]
        packed["features"] = z
        single_logits = self.single_classifier(z)
        packed["single_logits"] = single_logits
        return packed

    def enable_mc_dropout(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
