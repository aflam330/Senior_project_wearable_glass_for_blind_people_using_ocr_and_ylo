"""Federated averaging (FedAvg) for the authenticity CNN+ViT fusion head.

Simulates K clients that each train on a non-IID shard of JaalTaka
(Dirichlet label split), then averages client weights on the server.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from .authenticity import JaalTakaMultiView, MultiViewCNNVIT, list_jaaltaka_notes, split_notes
from .config import DEVICE, MODELS_DIR

FED_WEIGHTS = MODELS_DIR / "authenticity_fedavg.pt"


def dirichlet_partition(
    samples: list,
    n_clients: int = 3,
    alpha: float = 0.5,
    seed: int = 42,
) -> list[list]:
    """Non-IID split: each class is drawn from Dirichlet(α) over clients."""
    if not samples:
        return [[] for _ in range(n_clients)]
    rng = np.random.default_rng(seed)
    labels = np.array([item[1] for item in samples], dtype=int)
    shards: list[list] = [[] for _ in range(n_clients)]
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        props = rng.dirichlet([alpha] * n_clients)
        counts = (props * len(idx)).astype(int)
        while counts.sum() < len(idx):
            counts[int(np.argmax(props))] += 1
        while counts.sum() > len(idx):
            j = int(np.argmax(counts))
            if counts[j] > 0:
                counts[j] -= 1
            else:
                break
        start = 0
        for client, n in enumerate(counts):
            shards[client].extend(samples[i] for i in idx[start : start + n])
            start += n
    for shard in shards:
        rng.shuffle(shard)
    return shards


def _average_state(states: list[dict]) -> dict:
    avg = copy.deepcopy(states[0])
    for key in avg:
        stacked = torch.stack([s[key].float() for s in states], dim=0)
        avg[key] = stacked.mean(dim=0)
    return avg


def _client_update(model: MultiViewCNNVIT, loader: DataLoader, epochs: int, lr: float) -> dict:
    device = DEVICE
    model = copy.deepcopy(model).to(device)
    model.train()
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for views, y in loader:
            views, y = views.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(views), y)
            loss.backward()
            opt.step()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def run_fedavg(
    n_clients: int = 3,
    rounds: int = 100,
    local_epochs: int = 1,
    batch_size: int = 8,
    max_notes: int | None = 180,
    lr: float = 1e-3,
    alpha: float = 0.5,
) -> dict:
    notes = list_jaaltaka_notes()
    if max_notes:
        notes = notes[:max_notes]
    splits = split_notes(notes)
    train = splits["train"]
    if len(train) < n_clients:
        raise RuntimeError("not enough JaalTaka notes for FedAvg clients")
    shards = dirichlet_partition(train, n_clients=n_clients, alpha=alpha)
    shard_sizes = [len(s) for s in shards]
    shard_pos = [sum(1 for _, y in s if y == 1) / max(len(s), 1) for s in shards]
    global_model = MultiViewCNNVIT(freeze_cnn=True)
    history = []
    for rnd in range(rounds):
        client_states = []
        for shard in shards:
            if not shard:
                continue
            ds = JaalTakaMultiView(shard, n_views=2, train=True)
            loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, num_workers=0)
            client_states.append(_client_update(global_model, loader, local_epochs, lr))
        if not client_states:
            raise RuntimeError("all Dirichlet shards were empty")
        avg = _average_state(client_states)
        global_model.load_state_dict(avg)
        if (rnd + 1) % max(1, rounds // 10) == 0 or rnd == 0:
            acc = _eval(global_model, splits["val"])
            history.append({"round": rnd + 1, "clients": n_clients, "val_acc": acc})
            print(f"FedAvg round {rnd + 1}/{rounds} val_acc={acc:.3f} n_clients={n_clients} alpha={alpha}")
        else:
            print(f"FedAvg round {rnd + 1}/{rounds} averaged {len(client_states)} clients")
            history.append({"round": rnd + 1, "clients": n_clients})
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": global_model.state_dict(), "history": history, "alpha": alpha}, FED_WEIGHTS)
    acc = _eval(global_model, splits["val"])
    return {
        "algorithm": "FedAvg",
        "rounds": rounds,
        "clients": n_clients,
        "alpha": alpha,
        "partition": "non-IID Dirichlet",
        "shard_sizes": shard_sizes,
        "shard_genuine_frac": shard_pos,
        "val_acc": acc,
        "history": history,
        "weights": str(FED_WEIGHTS),
    }


@torch.inference_mode()
def _eval(model: MultiViewCNNVIT, notes) -> float:
    if not notes:
        return 0.0
    model = model.to(DEVICE).eval()
    ds = JaalTakaMultiView(notes, n_views=2, train=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    correct = total = 0
    for views, y in loader:
        pred = model(views.to(DEVICE)).argmax(1).cpu()
        correct += int((pred == y).sum())
        total += len(y)
    return correct / max(total, 1)
