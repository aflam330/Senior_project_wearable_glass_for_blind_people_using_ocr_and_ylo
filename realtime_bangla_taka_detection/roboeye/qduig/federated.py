"""Novelty 11 — Quality-and-Diversity Weighted Federated Aggregation (QDW-Fed).

SIMULATED federated learning only (no physical clients).
Beyond FedAvg (n_k weighting) and FedAvg+EWC:
  w_k = n_k · mean_usable_k · client_embedding_volume
so clients with redundant or low-quality views contribute less.

Non-IID client partitions are constructed from physical-note IDs.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def partition_notes_non_iid(
    note_ids: list[str],
    records: dict[str, dict],
    n_clients: int,
    seed: int,
    dirichlet_alpha: float = 0.4,
) -> list[list[str]]:
    """Dirichlet label-skew over physical notes. Returns client → note_id lists."""
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[str]] = {0: [], 1: []}
    for nid in note_ids:
        by_label[int(records[nid]["label"])].append(nid)
    clients: list[list[str]] = [[] for _ in range(n_clients)]
    for lab, ids in by_label.items():
        ids = list(ids)
        rng.shuffle(ids)
        mix = rng.dirichlet([dirichlet_alpha] * n_clients)
        cuts = (np.cumsum(mix) * len(ids)).astype(int)
        start = 0
        for c, end in enumerate(cuts):
            clients[c].extend(ids[start:end])
            start = end
        clients[-1].extend(ids[start:])
    return clients


def client_weight(n_k: int, mean_usable: float, volume: float, eps: float = 1e-6) -> float:
    return max(n_k, 1) * max(mean_usable, eps) * max(volume, eps)


def fedavg(states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    out = {}
    keys = states[0].keys()
    for k in keys:
        acc = None
        for st, wi in zip(states, w):
            term = st[k].float() * float(wi)
            acc = term if acc is None else acc + term
        out[k] = acc
    return out


def qdw_aggregate(
    states: list[dict[str, torch.Tensor]],
    n_k: list[int],
    mean_usable: list[float],
    volumes: list[float],
) -> dict[str, torch.Tensor]:
    weights = [client_weight(n, q, v) for n, q, v in zip(n_k, mean_usable, volumes)]
    return fedavg(states, weights)


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def communication_bytes(state: dict[str, torch.Tensor]) -> int:
    return int(sum(v.numel() * v.element_size() for v in state.values()))


def summarise_clients(partitions: list[list[str]], records: dict[str, dict]) -> list[dict[str, Any]]:
    rows = []
    for i, ids in enumerate(partitions):
        y = [int(records[n]["label"]) for n in ids]
        rows.append(
            {
                "client": i,
                "n_notes": len(ids),
                "n_genuine": int(sum(y)),
                "n_counterfeit": int(len(y) - sum(y)),
                "genuine_frac": float(np.mean(y)) if y else None,
            }
        )
    return rows
