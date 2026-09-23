"""Novelty 1 — Residual Spectral Quality Attention (RSQA).

Difference from quality-gated attention (CAMVA, TMC-style weights):
the quality head predicts a *12-factor embedding* from learned features
plus image descriptors, then applies a bilinear residual modulation
before attention. No descriptor is thresholded for accept/reject.
The usability score is calibrated on validation only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

QUALITY_FACTOR_NAMES = (
    "blur",
    "sharpness",
    "exposure",
    "brightness",
    "contrast",
    "saturation",
    "glare",
    "visible_note_fraction",
    "perspective",
    "occlusion",
    "detector_confidence",
    "crop_quality",
)

N_QUALITY_FACTORS = len(QUALITY_FACTOR_NAMES)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_quality_factors(views: torch.Tensor, detector_conf: torch.Tensor | None = None) -> torch.Tensor:
    """Compute 12 descriptor channels in [0, 1] from ImageNet-normalized views.

    These are *inputs* to a learned head, not decision thresholds.
    ``views``: (B, V, C, H, W)
    returns: (B, V, 12)
    """
    b, v, c, h, w = views.shape
    x = views.reshape(b * v, c, h, w)
    mean = views.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = views.new_tensor(IMAGENET_STD)[:, None, None]
    rgb = (x * std + mean).clamp(0.0, 1.0)
    y = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    cr = rgb[:, 0] - y
    cb = rgb[:, 2] - y

    mean_y = y.mean(dim=(1, 2))
    std_y = y.std(dim=(1, 2))
    chroma = torch.sqrt(cr.pow(2) + cb.pow(2) + 1e-8)
    sat = torch.tanh(chroma.std(dim=(1, 2)) * 4.0)
    glare = (rgb.amax(dim=1) > 0.90).float().mean(dim=(1, 2))
    dark = (y < 0.08).float().mean(dim=(1, 2))
    visible = (1.0 - dark).clamp(0.0, 1.0)
    exposure = 1.0 - (mean_y - 0.5).abs() * 2.0
    brightness = mean_y
    contrast = torch.tanh(std_y * 4.0)

    lap = y.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    y4 = y.unsqueeze(1)
    resp = F.conv2d(y4, lap, padding=1)
    lap_var = resp.var(dim=(1, 2, 3))
    sharpness = lap_var / (1.0 + lap_var)
    blur = 1.0 - sharpness

    top, bot = y[:, : h // 2].mean(dim=(1, 2)), y[:, h // 2 :].mean(dim=(1, 2))
    lef, rig = y[:, :, : w // 2].mean(dim=(1, 2)), y[:, :, w // 2 :].mean(dim=(1, 2))
    perspective = 1.0 - ((top - bot).abs() + (lef - rig).abs()).clamp(0.0, 1.0)

    pool = F.avg_pool2d(y4, kernel_size=max(h // 8, 2), stride=max(h // 8, 2))
    occlusion = (pool < 0.08).float().mean(dim=(1, 2, 3))

    border = torch.cat(
        [y[:, :2, :].reshape(b * v, -1), y[:, -2:, :].reshape(b * v, -1),
         y[:, :, :2].reshape(b * v, -1), y[:, :, -2:].reshape(b * v, -1)],
        dim=1,
    )
    crop_quality = 1.0 - (border < 0.08).float().mean(dim=1)

    if detector_conf is None:
        det = torch.ones(b * v, device=views.device, dtype=views.dtype)
    else:
        det = detector_conf.reshape(b * v)

    factors = torch.stack(
        [
            blur, sharpness, exposure, brightness, contrast, sat,
            glare, visible, perspective, occlusion, det, crop_quality,
        ],
        dim=-1,
    )
    factors = torch.nan_to_num(factors).clamp(0.0, 1.0)
    return factors.reshape(b, v, N_QUALITY_FACTORS)


class QualityFactorHead(nn.Module):
    """Learned usability predictor. No hand-crafted accept/reject threshold."""

    def __init__(self, feat_dim: int, factor_dim: int = N_QUALITY_FACTORS, hidden: int = 128):
        super().__init__()
        self.factor_proj = nn.Sequential(
            nn.Linear(factor_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
        )
        self.merge = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.usable = nn.Linear(hidden, 1)
        self.embed = nn.Linear(hidden, 32)

    def forward(self, z: torch.Tensor, factors: torch.Tensor) -> dict[str, torch.Tensor]:
        """z, factors: (B, V, *)"""
        h = self.merge(torch.cat([self.feat_proj(z), self.factor_proj(factors)], dim=-1))
        usable = torch.sigmoid(self.usable(h).squeeze(-1))
        embed = self.embed(h)
        return {"usable": usable, "embed": embed, "factors": factors, "hidden": h}


class ResidualSpectralQualityAttention(nn.Module):
    """Quality-weighted attention with bilinear residual modulation.

    Distinct from concatenating a scalar quality into an attention MLP:
    a learned quality embedding multiplicatively modulates features, then a
    bilinear interaction term enters the attention logits, plus a residual
    quality skip on the fused vector.
    """

    def __init__(self, feat_dim: int, q_dim: int = 32):
        super().__init__()
        self.mod = nn.Linear(q_dim, feat_dim)
        self.wz = nn.Linear(feat_dim, feat_dim)
        self.wq = nn.Linear(q_dim, feat_dim)
        self.wint = nn.Linear(feat_dim, feat_dim)
        self.score = nn.Linear(feat_dim, 1)
        self.residual = nn.Linear(q_dim, feat_dim)

    def forward(
        self,
        z: torch.Tensor,
        q_embed: torch.Tensor,
        usable: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fused (B, D), weights (B, V), modulated features (B, V, D)."""
        mod = 1.0 + torch.tanh(self.mod(q_embed))
        z_mod = z * mod
        inter = z_mod * torch.tanh(self.wq(q_embed))
        a = self.score(torch.tanh(self.wz(z_mod) + self.wq(q_embed) + self.wint(inter))).squeeze(-1)
        # usable is a soft gate on logits, not a hard threshold
        a = a + torch.log(usable.clamp(min=1e-4))
        a = a.masked_fill(mask == 0, -1e9)
        w = torch.softmax(a, dim=1)
        w = w * mask.to(w.dtype)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        fused = (w.unsqueeze(-1) * z_mod).sum(dim=1)
        q_bar = (w.unsqueeze(-1) * q_embed).sum(dim=1)
        fused = fused + 0.25 * torch.tanh(self.residual(q_bar))
        return fused, w, z_mod


def calibrate_quality_threshold(usable: torch.Tensor, helpful: torch.Tensor) -> float:
    """Pick a usability threshold on *validation* by Youden J on helpfulness.

    ``helpful`` is 1 if the view agreed with the full-set decision (val only).
    """
    u = usable.detach().float().cpu().numpy().ravel()
    y = helpful.detach().float().cpu().numpy().ravel()
    best_t, best_j = 0.5, -1.0
    for t in [i / 40.0 for i in range(2, 39)]:
        pred = (u >= t).astype("float64")
        tp = ((pred == 1) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t)
