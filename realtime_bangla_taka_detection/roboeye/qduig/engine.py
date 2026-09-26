"""Train / sequential-evaluate Q-DUIG-CAMVA on the existing note-disjoint split."""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..authenticity import MultiViewCNNVIT
from ..camva.data import NoteViewDataset, apply_corruption, collate_notes
from ..camva.engine import load_baseline, make_loader, predict_baseline
from ..camva.notes import load_splits
from ..config import DEVICE
from .artifacts import init_run, save_json
from .calibration import HERCalibrator, binary_entropy, fit_temperature
from .diversity import chordal_volume_scores
from .info_gain import realized_information_gain
from .losses import (
    LossWeights,
    acquisition_cost_loss,
    authentication_loss,
    composite_loss,
    diversity_redundancy_loss,
    info_gain_loss,
    quality_consistency_loss,
    uncertainty_loss,
)
from .metrics_ext import full_binary_report
from .model import QDUIGConfig, QDUIGNet
from .oracle import oracle_for_note, summarise_oracle
from .policy import (
    CostWeights,
    PolicyName,
    criqp_utilities,
    decide,
    incremental_cost,
    score_candidates,
)


def random_view_mask(mask: torch.Tensor, mode: str, max_k: int | None = None, full_prob: float = 0.0) -> torch.Tensor:
    """FIX C/G: keep a random prefix or subset so the classifier sees 1..K views.

    ``max_k`` caps the largest subset (curriculum). ``full_prob`` keeps every
    view on that fraction of batches. Never uses labels.
    """
    if not mode or mode in {"none", "false", "False"}:
        return mask
    if full_prob > 0 and float(torch.rand(1).item()) < full_prob:
        return mask
    b, v = mask.shape
    out = torch.zeros_like(mask)
    cap = v if max_k is None else max(1, min(v, int(max_k)))
    for i in range(b):
        valid = (mask[i] == 1).nonzero(as_tuple=False).view(-1)
        if valid.numel() == 0:
            continue
        hi = min(int(valid.numel()), cap)
        k = int(torch.randint(1, hi + 1, (1,)).item())
        use_prefix = mode == "prefix" or (mode == "mixed" and torch.rand(1).item() < 0.5)
        if use_prefix:
            chosen = valid[:k]
        else:
            perm = valid[torch.randperm(valid.numel(), device=mask.device)]
            chosen = perm[:k]
        out[i, chosen] = 1
    return out


def contrastive_view_loss(z: torch.Tensor, mask: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """FIX E: a single kept view should match its note and not other notes. Train/val only."""
    b = z.size(0)
    if b < 2:
        return z.new_zeros(())
    u = F.normalize(torch.nan_to_num(z), dim=-1)
    m = mask.unsqueeze(-1).to(u.dtype)
    proto = F.normalize((u * m).sum(1) / m.sum(1).clamp_min(1.0), dim=-1)
    anchors = []
    for i in range(b):
        valid = (mask[i] == 1).nonzero(as_tuple=False).view(-1)
        if valid.numel() == 0:
            anchors.append(u[i, 0])
        else:
            j = valid[int(torch.randint(0, valid.numel(), (1,)).item())]
            anchors.append(u[i, j])
    anchor = torch.stack(anchors, dim=0)
    logits = anchor @ proto.T / temperature
    labels = torch.arange(b, device=z.device)
    return F.cross_entropy(logits, labels)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cfg_from_dict(d: dict) -> QDUIGConfig:
    keys = {f.name for f in QDUIGConfig.__dataclass_fields__.values()}
    return QDUIGConfig(**{k: d[k] for k in d if k in keys})


def weights_from_dict(d: dict) -> LossWeights:
    return LossWeights(
        auth=float(d.get("lambda_auth", 1.0)),
        quality=float(d.get("lambda_q", 0.25)),
        uncertainty=float(d.get("lambda_u", 0.15)),
        diversity=float(d.get("lambda_d", 0.10)),
        info_gain=float(d.get("lambda_g", 0.20)),
        cost=float(d.get("lambda_c", 0.05)),
    )


def cost_from_dict(d: dict) -> CostWeights:
    return CostWeights(
        alpha=float(d.get("alpha", 1.0)),
        beta=float(d.get("beta", 0.05)),
        gamma=float(d.get("gamma", 0.0)),
        lamb=float(d.get("lambda_cost_policy", 0.08)),
    )


def build_model(config: dict) -> QDUIGNet:
    return QDUIGNet(cfg_from_dict(config.get("model", config)))


def _amp_enabled() -> bool:
    return torch.cuda.is_available()


def train_qduig(
    model: QDUIGNet,
    train_ids: list[str],
    val_ids: list[str],
    records: dict,
    *,
    config: dict,
    seed: int,
    output_dir: Path,
    resume: Path | None = None,
) -> dict:
    set_seed(seed)
    device = DEVICE
    model = model.to(device)
    epochs = int(config.get("epochs", 6))
    batch = int(config.get("batch", 8))
    lr = float(config.get("lr", 1e-3))
    n_views = int(config.get("views", 6))
    view_dropout = str(config.get("view_dropout", "none"))
    full_prob = float(config.get("full_view_prob", 0.0))
    curriculum = bool(config.get("curriculum", False))
    multitask_w = float(config.get("multitask_single", 0.0))
    contrastive_w = float(config.get("contrastive", 0.0))
    lw = weights_from_dict(config.get("loss", config))
    cw = cost_from_dict(config.get("cost", config))

    if resume and Path(resume).is_file():
        try:
            blob = torch.load(resume, map_location=device, weights_only=False)
        except TypeError:
            blob = torch.load(resume, map_location=device)
        model.load_state_dict(blob["model"])

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    use_amp = False  # AMP + slogdet/quality descriptors produced NaN on seed 42 epoch 1
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best = -1.0
    history: list[dict] = []
    ckpt = output_dir / "checkpoint.pt"
    log_path = output_dir / "train_log.csv"

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_acc", "val_mean_1_to_6", "auth", "quality", "uncertainty", "diversity", "info_gain", "cost"]
            + [f"val_acc_{k}" for k in range(1, 7)],
        )
        writer.writeheader()

        for ep in range(1, epochs + 1):
            model.train()
            print(f"qduig seed={seed} start epoch {ep}/{epochs}", flush=True)
            loader = make_loader(train_ids, records, n_views=n_views, train=True, batch=batch, seed=seed + ep)
            running = 0.0
            n = 0
            term_acc = {k: 0.0 for k in ("auth", "quality", "uncertainty", "diversity", "info_gain", "cost")}
            for bd in loader:
                views = bd["views"].to(device)
                mask = bd["mask"].to(device)
                y = bd["label"].to(device)
                if curriculum:
                    mask = random_view_mask(mask, "prefix", max_k=min(ep, n_views))
                elif view_dropout not in {"none", "false", "False", ""}:
                    mask = random_view_mask(mask, view_dropout, full_prob=full_prob)
                opt.zero_grad(set_to_none=True)
                try:
                    amp_ctx = torch.amp.autocast("cuda", enabled=use_amp)
                except (AttributeError, TypeError):
                    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp)
                with amp_ctx:
                    out = model(views, mask)
                    if not torch.isfinite(out["logits"]).all():
                        print("non-finite logits, skip batch", flush=True)
                        continue
                    terms = _loss_terms(
                        model, out, views, mask, y, cw, n_views,
                        multitask_w=multitask_w, contrastive_w=contrastive_w,
                    )
                    loss, logged = composite_loss(terms, lw)
                if not torch.isfinite(loss):
                    print("non-finite loss, skip batch", {k: logged[k] for k in logged}, flush=True)
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = torch.nan_to_num(p.grad)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                with torch.no_grad():
                    for p in model.parameters():
                        if not torch.isfinite(p).all():
                            p.nan_to_num_(0.0)
                running += float(loss.item()) * len(y)
                n += len(y)
                for k in term_acc:
                    term_acc[k] += logged[k] * len(y)

            per_k = {}
            for k in range(1, n_views + 1):
                val_loader = make_loader(val_ids, records, n_views=k, train=False, batch=batch)
                pred = predict_qduig(model, val_loader, device)
                per_k[k] = full_binary_report(np.array(pred["y_true"]), np.array(pred["genuine_score"]))["accuracy"]
            acc = per_k[n_views]
            mean_k = float(sum(per_k.values()) / len(per_k))
            row = {
                "epoch": ep,
                "train_loss": running / max(n, 1),
                "val_acc": acc,
                "val_mean_1_to_6": mean_k,
                **{k: term_acc[k] for k in term_acc},
                **{f"val_acc_{k}": per_k[k] for k in per_k},
            }
            # term_acc was a sum; convert
            for k in term_acc:
                row[k] = term_acc[k] / max(n, 1)
            history.append(row)
            writer.writerow(row)
            f.flush()
            ks = " ".join(f"{k}:{per_k[k]:.3f}" for k in per_k)
            print(
                f"qduig seed={seed} epoch {ep}/{epochs} loss={row['train_loss']:.4f} "
                f"val6={acc:.4f} mean1to6={mean_k:.4f} [{ks}]",
                flush=True,
            )
            if mean_k >= best:
                best = mean_k
                torch.save(
                    {
                        "model": model.state_dict(),
                        "val_acc": acc,
                        "val_mean_1_to_6": mean_k,
                        "val_by_k": per_k,
                        "epoch": ep,
                        "seed": seed,
                        "config": config,
                        "kind": "qduig",
                    },
                    ckpt,
                )

    save_json(output_dir / "val_metrics.json", {"best_val_acc": best, "history": history})
    return {"best_val_acc": best, "history": history, "ckpt": str(ckpt)}


def _loss_terms(
    model: QDUIGNet,
    out: dict[str, torch.Tensor],
    views: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    cw: CostWeights,
    max_views: int,
    multitask_w: float = 0.0,
    contrastive_w: float = 0.0,
) -> dict[str, torch.Tensor]:
    logits = out["logits"]
    l_auth = authentication_loss(logits, y)
    if multitask_w > 0:
        m = mask.bool()
        if int(m.sum()) > 0:
            y_rep = y.unsqueeze(1).expand_as(mask)[m]
            l_auth = l_auth + multitask_w * F.cross_entropy(out["single_logits"][m], y_rep)
    if contrastive_w > 0:
        l_auth = l_auth + contrastive_w * contrastive_view_loss(out["features"], mask)
    pred = logits.argmax(dim=1)
    correct = (pred == y).float()
    l_unc = uncertainty_loss(out["entropy"], correct)

    # leave-one-out agreement for quality (detached)
    with torch.no_grad():
        single_p = F.softmax(out["single_logits"], dim=-1)[..., 1]
        fused_pred = (out["prob"] >= 0.5).float().unsqueeze(1)
        agree = (single_p >= 0.5).float()
        agree = (agree == fused_pred).float()
    l_q = quality_consistency_loss(out["quality"], mask, agree.detach())
    l_d = diversity_redundancy_loss(out["redundancy"], mask)

    # realized IG: random prefix vs prefix+1 on train/val only
    l_g = _rollout_ig_loss(model, out, views, mask)
    n_used = mask.sum(dim=1).to(out["logits"].dtype)
    latency_proxy = n_used / float(max_views)
    energy_proxy = torch.zeros_like(n_used)
    l_c = acquisition_cost_loss(n_used, latency_proxy, energy_proxy, cw.alpha, cw.beta, cw.gamma, max_views)
    if not model.cfg.use_cost:
        l_c = l_c.detach() * 0.0
    if not model.cfg.use_info_gain:
        l_g = l_g.detach() * 0.0
    if not model.cfg.use_diversity:
        l_d = l_d.detach() * 0.0
    if not model.cfg.use_quality:
        l_q = l_q.detach() * 0.0
    if not model.cfg.use_uncertainty:
        l_unc = l_unc.detach() * 0.0
    def _finite(t: torch.Tensor) -> torch.Tensor:
        return t if torch.isfinite(t).all() else t.new_zeros(())

    return {
        "auth": l_auth,
        "quality": _finite(l_q),
        "uncertainty": _finite(l_unc),
        "diversity": _finite(l_d),
        "info_gain": _finite(l_g),
        "cost": _finite(l_c),
    }


def _rollout_ig_loss(model: QDUIGNet, out: dict, views: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compare prefix of k views vs k+1 on a random cut. Targets detached."""
    b, v = mask.shape
    if v < 2:
        return out["logits"].new_zeros(())
    # Random cut on the *kept* views so the surrogate sees 1→2 as well as longer prefixes.
    k = int(torch.randint(1, v, (1,)).item())
    prefix = torch.zeros_like(mask)
    extra = torch.zeros_like(mask)
    prefix[:, :k] = mask[:, :k]
    extra[:, : k + 1] = mask[:, : k + 1]
    if int(prefix.sum()) == 0 or int(extra.sum()) == int(prefix.sum()):
        return out["logits"].new_zeros(())
    was_training = model.training
    model.eval()
    with torch.no_grad():
        z = out["features"]
        q = {
            "usable": out["quality"],
            "embed": out["quality_embed"],
            "factors": out["quality_factors"],
        }
        before = model.fuse_prefix(z, q, prefix, fast_diversity=True)
        after = model.fuse_prefix(z, q, extra, fast_diversity=True)
        realized = realized_information_gain(before["prob"], after["prob"])
    if was_training:
        model.train()
    # candidate is view k (0-indexed)
    cand_idx = min(k, v - 1)
    pred = out["pred_ig"][:, cand_idx]
    return info_gain_loss(pred, realized)


@torch.inference_mode()
def predict_qduig(model: QDUIGNet, loader: DataLoader, device) -> dict:
    model.eval()
    ys, scores, logits_l, ids, nviews, times, quality, entropy, ig = [], [], [], [], [], [], [], [], []
    selected = []
    for batch in loader:
        views = batch["views"].to(device)
        mask = batch["mask"].to(device)
        t0 = time.perf_counter()
        out = model(views, mask)
        times.append((time.perf_counter() - t0) * 1000.0 / max(len(batch["label"]), 1))
        ys.append(batch["label"].numpy())
        scores.append(out["prob"].cpu().numpy())
        logits_l.append(out["logits"].cpu().numpy())
        ids.extend(batch["note_id"])
        nviews.extend(mask.sum(1).cpu().tolist())
        quality.extend(out["mean_q"].cpu().tolist())
        entropy.extend(out["entropy"].cpu().tolist())
        ig.extend(out["pred_ig"].mean(dim=1).cpu().tolist())
        for i in range(mask.size(0)):
            selected.append([j for j in range(mask.size(1)) if int(mask[i, j]) == 1])
    return {
        "note_id": ids,
        "y_true": np.concatenate(ys).tolist(),
        "genuine_score": np.concatenate(scores).tolist(),
        "logits": np.concatenate(logits_l).tolist(),
        "n_views": nviews,
        "latency_ms_per_note": times,
        "quality": quality,
        "entropy": entropy,
        "pred_ig": ig,
        "selected_indices": selected,
    }


@torch.inference_mode()
def sequential_predict(
    model: QDUIGNet,
    note_ids: list[str],
    records: dict,
    *,
    policy: PolicyName,
    cost: CostWeights,
    max_views: int = 6,
    force_k: int | None = None,
    seed: int = 42,
) -> dict:
    """Causal sequential acquisition. force_k uses the policy only for *order*, not stop."""
    model.eval()
    device = DEVICE
    ds = NoteViewDataset(note_ids, records, n_views=None, train=False, view_order="fixed")
    rng = np.random.default_rng(seed)
    rows = {
        "note_id": [],
        "y_true": [],
        "genuine_score": [],
        "logits": [],
        "n_views": [],
        "selected_indices": [],
        "stopped_early": [],
        "latency_ms_per_note": [],
        "quality": [],
        "entropy": [],
        "pred_ig": [],
        "decisions": [],
    }
    inc = incremental_cost(cost.alpha, cost.beta, cost.gamma)
    for i, nid in enumerate(note_ids):
        item = ds[i]
        views = item["views"].to(device)
        y = int(item["label"])
        vcount = int(views.size(0))
        t0 = time.perf_counter()
        z = model.encode_views(views.unsqueeze(0)).squeeze(0)
        q = model._quality(z.unsqueeze(0), views.unsqueeze(0))
        usable = q["usable"].squeeze(0)
        available = torch.ones(vcount, device=device, dtype=torch.long)
        selected: list[int] = []
        last = None
        decisions = []
        while len(selected) < min(vcount, max_views):
            if not selected:
                # first view: policy score with empty prefix
                residual = z.norm(dim=-1)
                diversity = residual
                entropy_v = torch.ones(vcount, device=device) * 0.693
                conf_v = usable
                dummy_u = criqp_utilities(usable, usable, torch.zeros_like(usable), available, inc, cost.lamb)
                scores = score_candidates(policy, usable, conf_v, diversity, entropy_v, dummy_u, available)
                if policy == "random":
                    scores = torch.tensor(rng.random(vcount), device=device, dtype=usable.dtype)
                    scores = scores.masked_fill(available == 0, -1e9)
                nxt = int(scores.argmax().item())
                selected.append(nxt)
                available[nxt] = 0
                decisions.append({"step": 1, "action": "REQUEST", "index": nxt, "reason": "first_view"})
            else:
                mask = torch.zeros(1, vcount, device=device, dtype=torch.long)
                mask[0, selected] = 1
                packed = model.fuse_prefix(z.unsqueeze(0), q, mask)
                last = packed
                pred_ig = model.candidate_ig(
                    packed["fused_h"],
                    packed["z_mod"],
                    packed["entropy"],
                    packed["prob"],
                    q["usable"],
                    packed["residual_energy"],
                    packed["n_views_norm"],
                ).squeeze(0)
                utilities = criqp_utilities(
                    pred_ig, q["usable"].squeeze(0), packed["redundancy"].squeeze(0), available, inc, cost.lamb
                )
                single_p = F.softmax(model.single_classifier(z), dim=-1)[:, 1]
                scores = score_candidates(
                    policy,
                    q["usable"].squeeze(0),
                    single_p,
                    packed["diversity"].squeeze(0),
                    binary_entropy_t_np_like(single_p),
                    utilities,
                    available,
                )
                if force_k is not None:
                    if len(selected) >= force_k:
                        break
                    nxt = int(scores.argmax().item())
                    selected.append(nxt)
                    available[nxt] = 0
                    decisions.append({"step": len(selected), "action": "REQUEST", "index": nxt, "forced_k": force_k})
                    continue
                if policy != "full_proposed":
                    # non-proposed policies: take next by score until force_k or all views
                    if len(selected) >= min(vcount, max_views):
                        break
                    nxt = int(scores.argmax().item())
                    selected.append(nxt)
                    available[nxt] = 0
                    decisions.append({"step": len(selected), "action": "REQUEST", "index": nxt, "policy": policy})
                    continue
                dec = decide(utilities, pred_ig, packed["entropy"].squeeze(0), selected, min(vcount, max_views))
                decisions.append(
                    {
                        "step": len(selected),
                        "action": dec.action,
                        "index": dec.next_index,
                        "utility": dec.utility,
                        "predicted_ig": dec.predicted_ig,
                        "current_uncertainty": dec.current_uncertainty,
                        "expected_post_view_uncertainty": dec.expected_post_view_uncertainty,
                    }
                )
                if dec.action == "STOP" or dec.next_index is None:
                    break
                selected.append(dec.next_index)
                available[dec.next_index] = 0

        mask = torch.zeros(1, vcount, device=device, dtype=torch.long)
        mask[0, selected] = 1
        packed = model.fuse_prefix(z.unsqueeze(0), q, mask)
        dt = (time.perf_counter() - t0) * 1000.0
        rows["note_id"].append(nid)
        rows["y_true"].append(y)
        rows["genuine_score"].append(float(packed["prob"].item()))
        rows["logits"].append(packed["logits"].cpu()[0].tolist())
        rows["n_views"].append(len(selected))
        rows["selected_indices"].append(list(selected))
        rows["stopped_early"].append(len(selected) < min(vcount, max_views))
        rows["latency_ms_per_note"].append(dt)
        rows["quality"].append(float(packed["mean_q"].item()))
        rows["entropy"].append(float(packed["entropy"].item()))
        rows["pred_ig"].append(float(packed.get("pred_ig", packed["entropy"]).mean().item()) if "pred_ig" in packed else 0.0)
        rows["decisions"].append(decisions)
    rows["policy"] = policy
    rows["average_views"] = float(np.mean(rows["n_views"])) if rows["n_views"] else 0.0
    rows["average_cost"] = float(
        cost.alpha * rows["average_views"] + cost.beta * float(np.mean(rows["latency_ms_per_note"]))
    )
    return rows


def binary_entropy_t_np_like(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-8, 1.0 - 1e-8)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def load_qduig(ckpt: Path, config: dict | None = None) -> QDUIGNet:
    try:
        blob = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    except TypeError:
        blob = torch.load(ckpt, map_location=DEVICE)
    cfg = config or blob.get("config") or {}
    net = build_model(cfg).to(DEVICE)
    net.load_state_dict(blob["model"])
    net.eval()
    return net


@torch.inference_mode()
def mc_dropout_probs(model: QDUIGNet, loader: DataLoader, device, n_samples: int = 8) -> np.ndarray:
    acc = []
    for _ in range(n_samples):
        model.enable_mc_dropout()
        scores = []
        for batch in loader:
            out = model(batch["views"].to(device), batch["mask"].to(device))
            scores.append(out["prob"].cpu().numpy())
        acc.append(np.concatenate(scores))
    model.eval()
    return np.mean(acc, axis=0)


@torch.inference_mode()
def run_oracle(model: QDUIGNet, note_ids: list[str], records: dict, max_notes: int | None = None) -> dict:
    model.eval()
    device = DEVICE
    ds = NoteViewDataset(note_ids, records, n_views=None, train=False, view_order="fixed")
    per = []
    learned_k, learned_ok = [], []
    ids = note_ids if max_notes is None else note_ids[:max_notes]
    for i, nid in enumerate(ids):
        item = ds[i]
        views = item["views"].to(device)
        y = int(item["label"])
        vcount = int(views.size(0))
        z = model.encode_views(views.unsqueeze(0))
        q = model._quality(z, views.unsqueeze(0))

        def predict_subset(subset: tuple[int, ...], z=z, q=q, y=y, vcount=vcount):
            mask = torch.zeros(1, vcount, device=device, dtype=torch.long)
            mask[0, list(subset)] = 1
            packed = model.fuse_prefix(z, q, mask)
            p = float(packed["prob"].item())
            return int(p >= 0.5), p

        rec = oracle_for_note(predict_subset, y, n_views=vcount)
        rec["note_id"] = nid
        rec["y_true"] = y
        per.append(rec)
        # 6-view learned
        full = predict_subset(tuple(range(vcount)))
        learned_k.append(vcount)
        learned_ok.append(int(full[0] == y))
    summary = summarise_oracle(per, learned_k, learned_ok)
    return {"summary": summary, "per_note": per}


def pack_and_save(pred: dict, name: str, out_dir: Path) -> dict:
    y = np.array(pred["y_true"])
    p = np.array(pred["genuine_score"])
    m = full_binary_report(y, p)
    m["name"] = name
    m["average_views"] = float(np.mean(pred["n_views"])) if pred.get("n_views") else None
    lat = np.array(pred.get("latency_ms_per_note") or [0.0])
    m["latency"] = {
        "median_ms": float(np.median(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "mean_ms": float(lat.mean()),
        "n": int(len(lat)),
    }
    if "average_cost" in pred:
        m["average_cost"] = pred["average_cost"]
    save_json(out_dir / "test_predictions.json", pred)
    save_json(out_dir / "test_metrics.json", m)
    save_json(out_dir / "confusion_matrix.json", {"matrix": m["confusion_matrix"]})
    return m
