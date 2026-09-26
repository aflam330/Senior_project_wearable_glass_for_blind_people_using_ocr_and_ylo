"""Train one novel algorithm. Seed is an argument. Test split is never loaded."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.adaptive_prefix_curriculum import AdaptivePrefixCurriculum
from models.causal_view_selection import CausalViewSelection
from models.cross_view_information import CrossViewInformationSharing
from models.difficulty_aware_loss import NoteDifficultyAwareLoss
from models.info_gain_contrastive import ContrastiveInfoGain
from models.meta_adaptive_view import MetaAdaptiveViewTransformer
from models.multi_task_prefix_transformer import MultiTaskPrefixTransformer
from models.oracle_guided_policy import OracleGuidedPolicy
from models.prefix_robust_adversarial import PrefixRobustAdversarial
from models.security_feature_quality import SecurityFeatureQualityHead
from models.sequential_federated import SequentialFederatedPrefixLearning
from models.sharpness_aware_views import SharpnessAwareViewSelection
from models.uncertainty_gated_fusion import UncertaintyGatedFusion
from models.view_count_distribution import ViewCountDistributionTraining
from models.view_count_invariant_encoder import ViewCountInvariantEncoder
from roboeye.camva.engine import make_loader
from roboeye.camva.notes import SPLIT_DIR, load_splits
from roboeye.config import DEVICE
from roboeye.qduig.artifacts import init_run, save_json
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import random_view_mask, set_seed
from roboeye.qduig.metrics_ext import full_binary_report

BUILDERS = {
    "ogpd": OracleGuidedPolicy,
    "vcie": ViewCountInvariantEncoder,
    "apc": AdaptivePrefixCurriculum,
    "sfaq": SecurityFeatureQualityHead,
    "igcr": ContrastiveInfoGain,
    "ugf": UncertaintyGatedFusion,
    "ndal": NoteDifficultyAwareLoss,
    "sfpl": SequentialFederatedPrefixLearning,
    "cvs": CausalViewSelection,
    "mtpt": MultiTaskPrefixTransformer,
    "pravt": PrefixRobustAdversarial,
    "cris": CrossViewInformationSharing,
    "savs": SharpnessAwareViewSelection,
    "mavt": MetaAdaptiveViewTransformer,
    "vat": ViewCountDistributionTraining,
}


def _module(model):
    net = model.net if hasattr(model, "net") else model
    return net._orig_mod if hasattr(net, "_orig_mod") else net


def _load_checkpoint(path: str):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def _autocast(enabled: bool):
    if enabled and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=True)
        except TypeError:
            pass
    if enabled:
        return torch.cuda.amp.autocast(enabled=True)
    return torch.autocast(device_type="cpu", enabled=False) if False else _NullContext()


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def _scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _loader_kwargs() -> dict:
    cuda = DEVICE.type == "cuda"
    workers = int(os.environ.get("NOVEL_WORKERS", "4" if cuda else "0"))
    return {"workers": max(0, workers), "pin_memory": cuda}


@torch.inference_mode()
def _accuracy(model, loader) -> dict:
    ys, ps = [], []
    model.eval()
    for batch in loader:
        views = batch["views"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        out = model(views, mask)
        ys.append(batch["label"].numpy())
        ps.append(out["prob"].detach().cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return full_binary_report(y, p)


def _loss(algo: str, model, views, mask, labels, note_ids: list[str]):
    if algo == "ogpd":
        teacher = model.oracle_stop(views, mask, labels)
        out = model(views, mask)
        return model.loss(out, labels, teacher), out
    if algo == "cvs":
        return model.loss(views, mask, labels), model(views, mask)
    if algo == "sfpl":
        clients = model.client_of(note_ids).to(views.device)
        capped = mask.clone()
        for i, c in enumerate(clients.tolist()):
            keep = (int(c) % 6) + 1
            capped[i, keep:] = 0
        out = model(views, capped)
        return model.loss(out, labels), out
    out = model(views, mask)
    if algo == "igcr":
        return model.loss(out, labels, mask), out
    if algo == "pravt":
        return model.loss(views, mask, labels)
    if algo == "vat":
        return model.loss(out, labels, mask), out
    if algo in {"sfaq", "mtpt", "apc", "ndal", "vcie", "ugf", "cris", "savs", "mavt"}:
        return model.loss(out, labels), out
    return F.cross_entropy(out["logits"], labels), out


def _sfpl_round(model, opt, scaler, use_amp: bool, splits, records, batch: int, seed: int, ep: int, load_kw: dict) -> float:
    """One FedAvg round. Each client keeps a different prefix length, then weights are averaged."""
    buckets: list[list[str]] = [[] for _ in range(model.n_clients)]
    for nid in splits["train"]:
        buckets[sum(ord(ch) for ch in nid) % model.n_clients].append(nid)
    base = {k: v.detach().clone() for k, v in _module(model).state_dict().items()}
    acc = {k: torch.zeros_like(v) for k, v in base.items()}
    total = 0.0
    running, n = 0.0, 0
    for cid, client_ids in enumerate(buckets):
        if len(client_ids) < 2:
            continue
        _module(model).load_state_dict(base)
        opt.state.clear()
        loader = make_loader(client_ids, records, n_views=6, train=True, batch=batch, seed=seed + ep * 17 + cid, **load_kw)
        keep = (cid % 6) + 1
        local_loss, local_n = 0.0, 0
        for bd in loader:
            views = bd["views"].to(DEVICE)
            mask = bd["mask"].to(DEVICE)
            labels = bd["label"].to(DEVICE)
            mask = mask.clone()
            mask[:, keep:] = 0
            opt.zero_grad(set_to_none=True)
            with _autocast(use_amp):
                out = model(views, mask)
                loss = model.loss(out, labels)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            local_loss += float(loss.detach().item()) * len(labels)
            local_n += len(labels)
        local = _module(model).state_dict()
        weight = float(len(client_ids))
        for key in acc:
            if torch.is_floating_point(local[key]):
                acc[key] += local[key].detach() * weight
            else:
                acc[key] = local[key].detach().clone()
        total += weight
        running += local_loss
        n += local_n
        print(f"sfpl client {cid} notes={len(client_ids)} keep={keep} loss={local_loss / max(local_n, 1):.4f}", flush=True)
    if total > 0:
        averaged = {}
        for key, value in acc.items():
            averaged[key] = value / total if torch.is_floating_point(value) else value
        _module(model).load_state_dict(averaged)
    return running / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--algo", required=True, choices=sorted(BUILDERS))
    p.add_argument("--batch", type=int, default=None)
    args = p.parse_args()
    if args.seed not in {42, 43, 44}:
        raise SystemExit("seeds for this experiment set are 42, 43, and 44")

    cfg = load_config(args.config)
    epochs = int(cfg.get("epochs", 4))
    batch = int(args.batch if args.batch is not None else cfg.get("batch", 4))
    lr = float(cfg.get("lr", 1e-3))
    patience = int(cfg.get("patience", 2))
    splits, records = load_splits()
    out = Path(args.output_dir)
    init_run(out, cfg, args.seed, f"{args.algo} seed {args.seed}", "Train and val only. Test split is not read.")
    manifest = SPLIT_DIR / "split_metadata.json"
    if manifest.is_file():
        (out / "dataset_manifest.json").write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
    set_seed(args.seed)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    model = BUILDERS[args.algo]().to(DEVICE)
    if os.environ.get("NOVEL_COMPILE", "1") != "0" and hasattr(torch, "compile") and DEVICE.type == "cuda":
        try:
            model.net = torch.compile(model.net)
            print(f"{args.algo} torch.compile enabled", flush=True)
        except Exception as exc:
            print(f"{args.algo} torch.compile skipped: {exc}", flush=True)
    else:
        print(f"{args.algo} torch.compile off", flush=True)
    if args.resume:
        blob = _load_checkpoint(args.resume)
        _module(model).load_state_dict(blob["model"])
    opt = torch.optim.Adam((t for t in model.parameters() if t.requires_grad), lr=lr)
    use_amp = DEVICE.type == "cuda" and args.algo not in {"vcie", "mtpt", "sfpl"}
    scaler = _scaler(use_amp)
    warmup = int(cfg.get("warmup_steps", 0))
    global_step = 0
    load_kw = _loader_kwargs()
    best = -1.0
    stall = 0
    history = []
    log_path = out / "train_log.csv"
    try:
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_mean_1_to_6"] + [f"val_acc_{k}" for k in range(1, 7)])
            writer.writeheader()
            for ep in range(1, epochs + 1):
                model.train()
                if args.algo == "apc":
                    model.maybe_advance(0.0, ep)
                print(f"{args.algo} epoch {ep}/{epochs}", flush=True)
                if args.algo == "sfpl":
                    train_loss = _sfpl_round(model, opt, scaler, use_amp, splits, records, batch, args.seed, ep, load_kw)
                    running, n = train_loss, 1
                    print(f"{args.algo} epoch {ep} fedavg loss={train_loss:.4f}", flush=True)
                else:
                    loader = make_loader(splits["train"], records, n_views=6, train=True, batch=batch, seed=args.seed + ep, **load_kw)
                    running, n = 0.0, 0
                opt.zero_grad(set_to_none=True)
                step_iter = [] if args.algo == "sfpl" else loader
                for step, bd in enumerate(step_iter, start=1):
                    views = bd["views"].to(DEVICE, non_blocking=use_amp)
                    mask = bd["mask"].to(DEVICE, non_blocking=use_amp)
                    labels = bd["label"].to(DEVICE, non_blocking=use_amp)
                    if args.algo == "apc":
                        mask = random_view_mask(mask, "prefix", max_k=model.max_k)
                    elif args.algo == "vat":
                        weights = torch.tensor([1.0 / k for k in range(1, 7)], device=mask.device)
                        chosen = torch.multinomial(weights, mask.size(0), replacement=True) + 1
                        for row, k in enumerate(chosen.tolist()):
                            mask[row, int(k):] = 0
                    elif args.algo != "sfpl":
                        mask = random_view_mask(mask, "mixed")
                    global_step += 1
                    if warmup:
                        scale = min(1.0, global_step / float(warmup))
                        for group in opt.param_groups:
                            group["lr"] = lr * scale
                    if args.algo == "savs":
                        loss, out_t = _loss(args.algo, model, views, mask, labels, bd["note_id"])
                        if not torch.isfinite(loss):
                            opt.zero_grad(set_to_none=True)
                            continue
                        loss.backward()
                        sq = [p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None]
                        gnorm = torch.sqrt(torch.stack(sq).sum()).clamp_min(1e-12)
                        saved = []
                        rho = 0.05
                        for p in model.parameters():
                            if p.grad is None:
                                saved.append(None)
                                continue
                            shift = p.grad.detach() * (rho / gnorm)
                            p.data.add_(shift)
                            saved.append(shift)
                        opt.zero_grad(set_to_none=True)
                        loss, out_t = _loss(args.algo, model, views, mask, labels, bd["note_id"])
                        loss.backward()
                        for p, shift in zip(model.parameters(), saved):
                            if shift is not None:
                                p.data.sub_(shift)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        running += float(loss.detach().item()) * len(labels)
                        n += len(labels)
                        if step % 50 == 0:
                            print(f"{args.algo} epoch {ep} step {step} loss={running / max(n, 1):.4f}", flush=True)
                        continue
                    with _autocast(use_amp):
                        loss, out_t = _loss(args.algo, model, views, mask, labels, bd["note_id"])
                    if not torch.isfinite(loss):
                        opt.zero_grad(set_to_none=True)
                        continue
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    running += float(loss.detach().item()) * len(labels)
                    n += len(labels)
                    if args.algo == "apc":
                        acc_b = float((out_t["logits"].argmax(1) == labels).float().mean().item())
                        model.maybe_advance(acc_b, ep)
                    if step % 50 == 0:
                        print(f"{args.algo} epoch {ep} step {step} loss={running / max(n, 1):.4f}", flush=True)
                per_k = {}
                for k in range(1, 7):
                    vloader = make_loader(splits["val"], records, n_views=k, train=False, batch=batch, **load_kw)
                    per_k[k] = _accuracy(model, vloader)["accuracy"]
                mean_k = float(sum(per_k.values()) / 6.0)
                row = {"epoch": ep, "train_loss": running / max(n, 1), "val_mean_1_to_6": mean_k, **{f"val_acc_{k}": per_k[k] for k in per_k}}
                writer.writerow(row)
                handle.flush()
                history.append(row)
                print(f"{args.algo} epoch {ep} loss={row['train_loss']:.4f} mean={mean_k:.4f} {per_k}", flush=True)
                if mean_k > best + 1e-6:
                    best = mean_k
                    stall = 0
                    torch.save({"model": _module(model).state_dict(), "epoch": ep, "val_mean_1_to_6": mean_k, "algo": args.algo, "seed": args.seed, "config": cfg}, out / "checkpoint.pt")
                else:
                    stall += 1
                    if stall >= patience:
                        print(f"{args.algo} early stop at epoch {ep} patience={patience}", flush=True)
                        break
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            print("CUDA_OOM", flush=True)
            raise SystemExit(2) from exc
        raise
    save_json(out / "val_metrics.json", {"best_val_mean_1_to_6": best, "history": history, "split": "val", "patience": patience})
    print(json.dumps({"algo": args.algo, "best_val_mean_1_to_6": best}))


if __name__ == "__main__":
    main()
