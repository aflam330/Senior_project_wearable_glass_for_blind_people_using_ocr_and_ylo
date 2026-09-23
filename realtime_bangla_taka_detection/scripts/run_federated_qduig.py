"""SIMULATED federated learning: FedAvg, FedAvg+EWC, QDW-Fed."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from roboeye.camva.engine import make_loader
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.artifacts import init_run, save_json
from roboeye.qduig.config_io import load_config
from roboeye.qduig.continual import fisher_diag, quality_weighted_ewc, snapshot_params
from roboeye.qduig.diversity import logdet_gram
from roboeye.qduig.engine import load_qduig, predict_qduig, set_seed
from roboeye.qduig.federated import (
    clone_state,
    communication_bytes,
    fedavg,
    load_state,
    partition_notes_non_iid,
    qdw_aggregate,
    summarise_clients,
)
from roboeye.qduig.metrics_ext import full_binary_report


def local_train(model, ids, records, epochs: int, batch: int, lr: float, ewc=None):
    if not ids:
        return
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    model.train()
    for ep in range(epochs):
        loader = make_loader(ids, records, n_views=6, train=True, batch=batch, seed=3 + ep)
        for bd in loader:
            opt.zero_grad()
            out = model(bd["views"].to(DEVICE), bd["mask"].to(DEVICE))
            loss = F.cross_entropy(out["logits"], bd["label"].to(DEVICE))
            if ewc is not None:
                fisher, star = ewc
                loss = loss + 20.0 * quality_weighted_ewc(model, fisher, star, 1.0)
            loss.backward()
            opt.step()


@torch.no_grad()
def client_stats(model, ids, records) -> tuple[float, float]:
    if not ids:
        return 0.5, 1e-3
    loader = make_loader(ids[: min(16, len(ids))], records, n_views=6, train=False, batch=4)
    qs, vols = [], []
    for bd in loader:
        out = model(bd["views"].to(DEVICE), bd["mask"].to(DEVICE))
        qs.append(float(out["mean_q"].mean().item()))
        vols.append(float(out["volume"].mean().item()))
    return float(np.mean(qs) if qs else 0.5), float(np.mean(vols) if vols else 1e-3)


def eval_test(model, splits, records) -> dict:
    loader = make_loader(splits["test"], records, n_views=6, train=False, batch=8)
    pred = predict_qduig(model, loader, DEVICE)
    return full_binary_report(np.array(pred["y_true"]), np.array(pred["genuine_score"]))


def run_method(name, model_ctor, partitions, records, splits, rounds, local_epochs, batch, lr):
    global_model = model_ctor()
    comm = 0
    history = []
    fisher = star = None
    if name == "fedavg_ewc":
        # fisher from a tiny shared val slice of client 0 only (train/val, not test)
        loader = make_loader(partitions[0][:16], records, n_views=6, train=False, batch=4)
        batches = [(bd["views"], bd["mask"], bd["label"]) for i, bd in enumerate(loader) if i < 4]
        fisher = fisher_diag(global_model, batches, DEVICE)
        star = snapshot_params(global_model)

    for r in range(rounds):
        states, ns, qs, vs = [], [], [], []
        gstate = clone_state(global_model)
        for ids in partitions:
            client = model_ctor()
            load_state(client, gstate)
            ewc = (fisher, star) if name == "fedavg_ewc" else None
            local_train(client, ids, records, local_epochs, batch, lr, ewc=ewc)
            states.append(clone_state(client))
            ns.append(len(ids))
            q, vol = client_stats(client, ids, records)
            qs.append(q)
            vs.append(vol)
            comm += communication_bytes(states[-1])
        if name == "qdw_fed":
            new_state = qdw_aggregate(states, ns, qs, vs)
        else:
            new_state = fedavg(states, [float(n) for n in ns])
        load_state(global_model, new_state)
        comm += communication_bytes(new_state)
        met = eval_test(global_model, splits, records)
        history.append({"round": r + 1, "accuracy": met["accuracy"], "macro_f1": met["macro_f1"]})
        print(name, "round", r + 1, met["accuracy"], flush=True)
    final = eval_test(global_model, splits, records)
    size = communication_bytes(clone_state(global_model))
    return {
        "method": name,
        "simulated": True,
        "rounds": rounds,
        "history": history,
        "final": final,
        "communication_bytes": comm,
        "model_size_bytes": size,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "proposed.yaml"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clients", type=int, default=4)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    splits, records = load_splits()
    parts = partition_notes_non_iid(splits["train"], records, args.clients, args.seed)
    out = Path(args.output_dir) if args.output_dir else ROOT / "results" / "qduig" / "federated" / f"seed{args.seed}"
    init_run(out, cfg, args.seed, "SIMULATED federated QDW-Fed", "No physical clients.")
    save_json(out / "client_distributions.json", summarise_clients(parts, records))

    def ctor():
        set_seed(args.seed)
        return load_qduig(Path(args.checkpoint), cfg)

    results = {}
    for name in ("fedavg", "fedavg_ewc", "qdw_fed"):
        results[name] = run_method(name, ctor, parts, records, splits, args.rounds, args.local_epochs, 8, 5e-4)
    save_json(out / "federated.json", results)
    print("federated done", out)


if __name__ == "__main__":
    main()
