"""Continual learning: naive FT, replay, EWC, QWER-VPC. Multiple seeds if checkpoints exist."""
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
from roboeye.qduig.continual import (
    fisher_diag,
    prototype_loss,
    quality_weighted_ewc,
    snapshot_params,
    update_prototypes,
)
from roboeye.qduig.engine import build_model, load_qduig, pack_and_save, predict_qduig, set_seed
from roboeye.qduig.metrics_ext import full_binary_report


def split_tasks(train_ids, records, seed: int):
    rng = np.random.default_rng(seed)
    by = {0: [], 1: []}
    for nid in train_ids:
        by[int(records[nid]["label"])].append(nid)
    t1, t2 = [], []
    for lab, ids in by.items():
        ids = list(ids)
        rng.shuffle(ids)
        mid = len(ids) // 2
        t1.extend(ids[:mid])
        t2.extend(ids[mid:])
    return t1, t2


def finetune(model, ids, records, steps_epochs: int, batch: int, lr: float, extra=None):
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    model.train()
    for ep in range(steps_epochs):
        loader = make_loader(ids, records, n_views=6, train=True, batch=batch, seed=10 + ep)
        for bd in loader:
            views, mask, y = bd["views"].to(DEVICE), bd["mask"].to(DEVICE), bd["label"].to(DEVICE)
            opt.zero_grad()
            out = model(views, mask)
            loss = F.cross_entropy(out["logits"], y)
            if extra is not None:
                loss = loss + extra(model, out, y)
            loss.backward()
            opt.step()


def eval_ids(model, ids, records) -> dict:
    loader = make_loader(ids, records, n_views=6, train=False, batch=8)
    pred = predict_qduig(model, loader, DEVICE)
    return full_binary_report(np.array(pred["y_true"]), np.array(pred["genuine_score"]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "proposed.yaml"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--epochs", type=int, default=2)
    args = p.parse_args()

    cfg = load_config(args.config)
    splits, records = load_splits()
    t1, t2 = split_tasks(splits["train"], records, args.seed)
    out = Path(args.output_dir) if args.output_dir else ROOT / "results" / "qduig" / "continual" / f"seed{args.seed}"
    init_run(out, cfg, args.seed, "continual QWER-VPC", "Task split is note-disjoint halves of train. Test unused.")

    methods = {}
    for name in ("naive_ft", "replay", "ewc", "qwer_vpc"):
        set_seed(args.seed)
        model = load_qduig(Path(args.checkpoint), cfg)
        after_t1 = eval_ids(model, t1, records)

        extra = None
        replay_ids = t1[:64]
        if name == "replay":
            def extra(m, out, y, rid=replay_ids):
                return 0.0
        if name == "ewc":
            loader = make_loader(t1[:32], records, n_views=6, train=False, batch=4)
            batches = []
            for bd in loader:
                batches.append((bd["views"], bd["mask"], bd["label"]))
                if len(batches) >= 8:
                    break
            fisher = fisher_diag(model, batches, DEVICE)
            star = snapshot_params(model)

            def extra(m, out, y, fisher=fisher, star=star):
                return 100.0 * quality_weighted_ewc(m, fisher, star, 1.0)

        if name == "qwer_vpc":
            loader = make_loader(t1[:48], records, n_views=6, train=False, batch=4)
            bank = None
            for bd in loader:
                with torch.no_grad():
                    o = model(bd["views"].to(DEVICE), bd["mask"].to(DEVICE))
                bank = update_prototypes(bank, o["fused_h"].detach(), bd["label"].to(DEVICE), o["mean_q"].detach())
            batches = []
            for bd in make_loader(t1[:32], records, n_views=6, train=False, batch=4):
                batches.append((bd["views"], bd["mask"], bd["label"]))
                if len(batches) >= 8:
                    break
            fisher = fisher_diag(model, batches, DEVICE)
            star = snapshot_params(model)
            q_scale = float(np.mean([float(bank.genuine_q.mean()), float(bank.counterfeit_q.mean())])) if bank else 1.0

            def extra(m, out, y, bank=bank, fisher=fisher, star=star, q_scale=q_scale):
                return 0.5 * prototype_loss(out["fused_h"], y, bank) + 50.0 * quality_weighted_ewc(m, fisher, star, q_scale)

        if name == "replay":
            mixed = list(t2) + replay_ids
            finetune(model, mixed, records, args.epochs, 8, 5e-4, None)
        else:
            finetune(model, t2, records, args.epochs, 8, 5e-4, extra)

        new_t = eval_ids(model, t2, records)
        old_t = eval_ids(model, t1, records)
        methods[name] = {
            "after_t1_on_t1": after_t1,
            "after_t2_on_t2_new_task": new_t,
            "after_t2_on_t1_old_task": old_t,
            "forgetting_acc": after_t1["accuracy"] - old_t["accuracy"],
        }
        print(name, "new", new_t["accuracy"], "old", old_t["accuracy"], "forget", methods[name]["forgetting_acc"], flush=True)

    save_json(out / "continual.json", methods)
    print("continual done", out)


if __name__ == "__main__":
    main()
