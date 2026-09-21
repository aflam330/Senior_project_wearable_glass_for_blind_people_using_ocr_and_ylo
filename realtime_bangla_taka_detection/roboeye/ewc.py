"""Continual learning with Elastic Weight Consolidation (EWC) + replay.

Task 1 = first half of JaalTaka notes, task 2 = second half. After task 1
we store a Fisher diagonal and a replay buffer; task 2 minimises CE + EWC
+ replay so earlier notes are not forgotten.
"""

from __future__ import annotations

import copy
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .authenticity import JaalTakaMultiView, MultiViewCNNVIT, list_jaaltaka_notes
from .config import DEVICE, MODELS_DIR

EWC_WEIGHTS = MODELS_DIR / "authenticity_ewc.pt"


class _Replay(Dataset):
    def __init__(self, items: list[tuple[torch.Tensor, torch.Tensor]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _fisher(model: MultiViewCNNVIT, loader: DataLoader, max_batches: int = 20) -> dict:
    model = model.to(DEVICE).eval()
    fisher = {n: torch.zeros_like(p, device="cpu") for n, p in model.named_parameters() if p.requires_grad}
    n = 0
    for i, (views, y) in enumerate(loader):
        if i >= max_batches:
            break
        model.zero_grad()
        loss = F.cross_entropy(model(views.to(DEVICE)), y.to(DEVICE))
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[name] += p.grad.detach().cpu() ** 2
        n += 1
    n = max(n, 1)
    return {k: v / n for k, v in fisher.items()}


def _train_epoch(model, loader, opt, fisher=None, star=None, lam: float = 8.0, replay_loader=None):
    model.train()
    total = 0.0
    steps = 0
    replay_iter = iter(replay_loader) if replay_loader is not None else None
    for views, y in loader:
        views, y = views.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = F.cross_entropy(model(views), y)
        if replay_iter is not None:
            try:
                rv, ry = next(replay_iter)
            except StopIteration:
                replay_iter = iter(replay_loader)
                rv, ry = next(replay_iter)
            loss = loss + F.cross_entropy(model(rv.to(DEVICE)), ry.to(DEVICE))
        if fisher is not None and star is not None:
            ewc = 0.0
            for name, p in model.named_parameters():
                if name in fisher and p.requires_grad:
                    ewc = ewc + (fisher[name].to(DEVICE) * (p - star[name].to(DEVICE)).pow(2)).sum()
            loss = loss + lam * ewc
        loss.backward()
        opt.step()
        total += float(loss.item())
        steps += 1
    return total / max(steps, 1)


@torch.inference_mode()
def _acc(model, notes) -> float:
    if not notes:
        return 0.0
    ds = JaalTakaMultiView(notes, n_views=2, train=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    model.eval()
    correct = total = 0
    for views, y in loader:
        pred = model(views.to(DEVICE)).argmax(1).cpu()
        correct += int((pred == y).sum())
        total += len(y)
    return correct / max(total, 1)


def _snapshot_replay(notes, k: int = 40) -> _Replay:
    sample = notes if len(notes) <= k else random.sample(notes, k)
    ds = JaalTakaMultiView(sample, n_views=2, train=False)
    items = [ds[i] for i in range(len(ds))]
    return _Replay(items)


def run_ewc(
    epochs_per_task: int = 2,
    batch_size: int = 8,
    max_notes: int | None = 200,
    lam: float = 8.0,
    replay_k: int = 40,
) -> dict:
    notes = list_jaaltaka_notes()
    if max_notes:
        notes = notes[:max_notes]
    random.Random(42).shuffle(notes)
    mid = len(notes) // 2
    task1, task2 = notes[:mid], notes[mid:]
    model = MultiViewCNNVIT(freeze_cnn=True).to(DEVICE)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)

    loader1 = DataLoader(JaalTakaMultiView(task1, n_views=2, train=True), batch_size=batch_size, shuffle=True)
    for ep in range(epochs_per_task):
        loss = _train_epoch(model, loader1, opt)
        print(f"EWC task1 epoch {ep + 1} loss={loss:.4f}")
    acc1_before = _acc(model, task1)
    fisher = _fisher(model, loader1)
    star = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    replay = _snapshot_replay(task1, k=replay_k)
    replay_loader = DataLoader(replay, batch_size=min(8, len(replay)), shuffle=True)

    loader2 = DataLoader(JaalTakaMultiView(task2, n_views=2, train=True), batch_size=batch_size, shuffle=True)
    for ep in range(epochs_per_task):
        loss = _train_epoch(model, loader2, opt, fisher=fisher, star=star, lam=lam, replay_loader=replay_loader)
        print(f"EWC task2 epoch {ep + 1} loss={loss:.4f}")

    acc1_after = _acc(model, task1)
    acc2 = _acc(model, task2)
    forgetting = max(0.0, acc1_before - acc1_after)
    result = {
        "task1_acc_after_task1": acc1_before,
        "task1_acc_after_task2": acc1_after,
        "task2_acc": acc2,
        "forgetting": forgetting,
        "forgetting_pct": round(forgetting * 100.0, 2),
        "weights": str(EWC_WEIGHTS),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metrics": result}, EWC_WEIGHTS)
    print(result)
    return result
