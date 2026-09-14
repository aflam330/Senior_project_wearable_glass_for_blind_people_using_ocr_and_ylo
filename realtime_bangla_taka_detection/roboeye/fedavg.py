"""Federated averaging (FedAvg) for the authenticity CNN+ViT fusion head.

Simulates K clients that each train on a shard of JaalTaka, then averages
their `fusion` layer weights on the server. The YOLO backbone stays local
and frozen — only the authenticity head is federated.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .authenticity import JaalTakaMultiView, MultiViewCNNVIT, list_jaaltaka_notes, split_notes
from .config import DEVICE, MODELS_DIR

FED_WEIGHTS = MODELS_DIR / "authenticity_fedavg.pt"


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
    rounds: int = 2,
    local_epochs: int = 1,
    batch_size: int = 8,
    max_notes: int | None = 180,
    lr: float = 1e-3,
) -> dict:
    notes = list_jaaltaka_notes()
    if max_notes:
        notes = notes[:max_notes]
    splits = split_notes(notes)
    train = splits["train"]
    if len(train) < n_clients:
        raise RuntimeError("not enough JaalTaka notes for FedAvg clients")
    shards = [train[i::n_clients] for i in range(n_clients)]
    global_model = MultiViewCNNVIT(freeze_cnn=True)
    history = []
    for rnd in range(rounds):
        client_states = []
        for shard in shards:
            ds = JaalTakaMultiView(shard, n_views=2, train=True)
            loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, num_workers=0)
            client_states.append(_client_update(global_model, loader, local_epochs, lr))
        avg = _average_state(client_states)
        global_model.load_state_dict(avg)
        history.append({"round": rnd + 1, "clients": n_clients})
        print(f"FedAvg round {rnd + 1}/{rounds} averaged {n_clients} clients")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": global_model.state_dict(), "history": history}, FED_WEIGHTS)
    acc = _eval(global_model, splits["val"])
    return {"rounds": rounds, "clients": n_clients, "val_acc": acc, "weights": str(FED_WEIGHTS)}


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
