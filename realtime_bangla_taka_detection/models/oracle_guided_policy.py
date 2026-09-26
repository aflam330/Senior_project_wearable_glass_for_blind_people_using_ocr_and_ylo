"""Oracle-guided policy distillation.

The teacher is the shortest label-consistent prefix, computed only while
training. Test-time ``act`` never reads labels.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import NovelAuthNet


class OracleGuidedPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = NovelAuthNet(algo="ogpd")

    def forward(self, views: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.net(views, mask)

    @torch.no_grad()
    def oracle_stop(self, views: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Index 0..5 of the first prefix view-count whose prediction matches the label."""
        b, v = mask.shape
        target = torch.full((b,), v - 1, device=views.device, dtype=torch.long)
        for k in range(1, v + 1):
            sub = mask.clone()
            sub[:, k:] = 0
            logits = self.net(views, sub)["logits"]
            ok = logits.argmax(1) == labels
            unset = target == (v - 1)
            target = torch.where(ok & unset & (mask[:, k - 1] == 1), torch.full_like(target, k - 1), target)
        return target

    def loss(self, out: dict[str, torch.Tensor], labels: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        auth = F.cross_entropy(out["logits"], labels)
        temperature = 2.0
        student = F.log_softmax(out["policy_logits"] / temperature, dim=-1)
        teacher_soft = F.one_hot(teacher, 6).float()
        distill = F.kl_div(student, teacher_soft, reduction="batchmean") * (temperature ** 2)
        return auth + 0.25 * distill

    def act(self, views: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Test-time stop index. No labels."""
        out = self.net(views, mask)
        return out["policy_logits"].argmax(dim=-1)
