"""Simulated federated prefix learning. No network and no extra machines.

Clients are slices of the training split. Each client is assigned a max
view count so the server sees heterogeneous prefixes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.backbone import NovelAuthNet, balanced_ce


class SequentialFederatedPrefixLearning(torch.nn.Module):
    def __init__(self, n_clients: int = 10):
        super().__init__()
        self.net = NovelAuthNet(algo="sfpl")
        self.n_clients = n_clients

    def forward(self, views, mask):
        return self.net(views, mask)

    def client_of(self, note_ids: list[str]) -> torch.Tensor:
        idx = [sum(ord(ch) for ch in nid) % self.n_clients for nid in note_ids]
        return torch.tensor(idx, dtype=torch.long)

    def loss(self, out, labels):
        return balanced_ce(out["logits"], labels)
