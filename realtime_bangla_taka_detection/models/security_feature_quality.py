"""Security-feature quality proxies.

JaalTaka has no hologram, watermark, thread, microprint, or serial labels.
The six scores are image statistics used as a quality head, not measured
security-element detectors. Names in the config are the intended targets;
the code does not pretend those elements were annotated.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.backbone import NovelAuthNet


def image_security_proxies(views: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, 6) mean proxies over real views. No learned claim about holograms."""
    b, v, c, h, w = views.shape
    x = views.reshape(b * v, c, h, w)
    gray = x.mean(dim=1, keepdim=True)
    # local contrast, saturation spread, high-frequency energy, edge energy, brightness, chroma
    contrast = gray.std(dim=(2, 3)).flatten()
    mx = x.amax(dim=1)
    mn = x.amin(dim=1)
    sat = ((mx - mn) / mx.clamp_min(1e-4)).mean(dim=(1, 2))
    blur = torch.nn.functional.avg_pool2d(gray, 5, stride=1, padding=2)
    hf = (gray - blur).abs().mean(dim=(1, 2, 3))
    dx = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    bright = gray.mean(dim=(1, 2, 3))
    chroma = x.std(dim=1).mean(dim=(1, 2))
    feats = torch.stack([contrast, sat, hf, dx, bright, chroma], dim=-1)
    feats = feats.reshape(b, v, 6)
    m = mask.to(feats.dtype).unsqueeze(-1)
    return (feats * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class SecurityFeatureQualityHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = NovelAuthNet(algo="sfaq")

    def forward(self, views: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(views, mask)
        out["proxies"] = image_security_proxies(views, mask)
        return out

    def loss(self, out: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        auth = F.cross_entropy(out["logits"], labels)
        # Regress the six proxies from the fused embedding. Proxies are detached targets.
        reg = F.mse_loss(torch.sigmoid(out["sfaq_from_embed"]), out["proxies"].detach().clamp(0, 1))
        return auth + 0.2 * reg
