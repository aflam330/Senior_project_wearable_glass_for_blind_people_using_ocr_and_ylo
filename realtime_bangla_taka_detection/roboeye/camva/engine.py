"""Train / evaluate / calibrate CAMVA and the CNN+ViT baseline on the same split."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..authenticity import MultiViewCNNVIT
from ..config import DEVICE
from .data import NoteViewDataset, collate_notes
from .metrics import (
    binary_metrics,
    bootstrap_ci,
    pr_curve_data,
    reliability_diagram,
    roc_curve_data,
)
from .model import CAMVANet
from .notes import CAMVA_ROOT

CKPT_DIR = CAMVA_ROOT / "checkpoints"
METRIC_DIR = CAMVA_ROOT / "metrics"
PRED_DIR = CAMVA_ROOT / "predictions"
CM_DIR = CAMVA_ROOT / "confusion_matrices"
CAL_DIR = CAMVA_ROOT / "calibration"
PLOT_DIR = CAMVA_ROOT / "plots"
LAT_DIR = CAMVA_ROOT / "latency"
CFG_DIR = CAMVA_ROOT / "configs"


def _dirs() -> None:
    for d in (CKPT_DIR, METRIC_DIR, PRED_DIR, CM_DIR, CAL_DIR, PLOT_DIR, LAT_DIR, CFG_DIR, CAMVA_ROOT / "splits"):
        d.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def temperature_scale(logits: np.ndarray, y: np.ndarray) -> float:
    """Fit T on val logits (N, 2)."""
    t = torch.nn.Parameter(torch.ones(1))
    y_t = torch.tensor(y, dtype=torch.long)
    logits_t = torch.tensor(logits, dtype=torch.float32)
    opt = torch.optim.LBFGS([t], lr=0.25, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits_t / t.clamp_min(1e-3), y_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.detach().clamp(min=0.05, max=20.0).item())


def apply_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    x = logits / max(t, 1e-6)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


@torch.inference_mode()
def predict_camva(model: CAMVANet, loader: DataLoader, device) -> dict:
    model.eval()
    ys, scores, logits_l, ids, nviews, times = [], [], [], [], [], []
    for batch in loader:
        views = batch["views"].to(device)
        mask = batch["mask"].to(device)
        t0 = time.perf_counter()
        out = model(views, mask)
        times.append((time.perf_counter() - t0) * 1000.0 / max(len(batch["label"]), 1))
        logit = out["logits"].cpu().numpy()
        prob = F.softmax(out["logits"], dim=1)[:, 1].cpu().numpy()
        ys.append(batch["label"].numpy())
        scores.append(prob)
        logits_l.append(logit)
        ids.extend(batch["note_id"])
        nviews.extend(mask.sum(1).cpu().tolist())
    y = np.concatenate(ys)
    s = np.concatenate(scores)
    lg = np.concatenate(logits_l)
    return {
        "note_id": ids,
        "y_true": y.tolist(),
        "genuine_score": s.tolist(),
        "logits": lg.tolist(),
        "n_views": nviews,
        "latency_ms_per_note": times,
    }


@torch.inference_mode()
def predict_baseline(model: MultiViewCNNVIT, loader: DataLoader, device) -> dict:
    model.eval()
    ys, scores, logits_l, ids, nviews, times = [], [], [], [], [], []
    for batch in loader:
        views = batch["views"].to(device)
        mask = batch["mask"]
        t0 = time.perf_counter()
        logit = model(views)
        times.append((time.perf_counter() - t0) * 1000.0 / max(len(batch["label"]), 1))
        prob = F.softmax(logit, dim=1)[:, 1].cpu().numpy()
        ys.append(batch["label"].numpy())
        scores.append(prob)
        logits_l.append(logit.cpu().numpy())
        ids.extend(batch["note_id"])
        nviews.extend(mask.sum(1).tolist())
    y = np.concatenate(ys)
    s = np.concatenate(scores)
    return {
        "note_id": ids,
        "y_true": y.tolist(),
        "genuine_score": s.tolist(),
        "logits": np.concatenate(logits_l).tolist(),
        "n_views": nviews,
        "latency_ms_per_note": times,
    }


def pack_eval(pred: dict, name: str, temperature: float | None = None) -> dict:
    y = np.array(pred["y_true"])
    logits = np.array(pred["logits"])
    if temperature is not None:
        prob = apply_temperature(logits, temperature)[:, 1]
        pred = dict(pred)
        pred["genuine_score"] = prob.tolist()
        pred["temperature"] = temperature
    else:
        prob = np.array(pred["genuine_score"])
    m = binary_metrics(y, prob)
    m["bootstrap_accuracy"] = bootstrap_ci(y, prob, "accuracy")
    m["bootstrap_f1"] = bootstrap_ci(y, prob, "f1")
    m["roc_curve"] = roc_curve_data(y, prob)
    m["pr_curve"] = pr_curve_data(y, prob)
    m["reliability"] = reliability_diagram(y, prob)
    lat = np.array(pred.get("latency_ms_per_note") or [0.0])
    m["latency"] = {
        "median_ms": float(np.median(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "mean_ms": float(lat.mean()),
        "n": int(len(lat)),
    }
    m["average_views"] = float(np.mean(pred["n_views"])) if pred.get("n_views") else None
    m["name"] = name
    return m, pred


def make_loader(ids, records, n_views, train, batch, view_order="fixed", corruption=None, severity=0.0, seed=42):
    ds = NoteViewDataset(
        ids, records, n_views=n_views, train=train, view_order=view_order,
        corruption=corruption, corruption_severity=severity, seed=seed,
    )
    return DataLoader(ds, batch_size=batch, shuffle=train, num_workers=0, collate_fn=collate_notes)


def train_one(
    model: torch.nn.Module,
    train_ids,
    val_ids,
    records,
    *,
    epochs: int,
    batch: int,
    lr: float,
    n_views: int,
    ckpt: Path,
    kind: str,
    seed: int,
    fusion: str | None = None,
) -> dict:
    _dirs()
    device = DEVICE
    model = model.to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    best = -1.0
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        loader = make_loader(train_ids, records, n_views=n_views, train=True, batch=batch, seed=seed + ep)
        running = 0.0
        n = 0
        for batch_data in loader:
            views = batch_data["views"].to(device)
            y = batch_data["label"].to(device)
            opt.zero_grad()
            if kind == "camva":
                mask = batch_data["mask"].to(device)
                logits = model(views, mask)["logits"]
            else:
                logits = model(views)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(y)
            n += len(y)
        val_loader = make_loader(val_ids, records, n_views=n_views, train=False, batch=batch)
        if kind == "camva":
            pred = predict_camva(model, val_loader, device)
        else:
            pred = predict_baseline(model, val_loader, device)
        acc = binary_metrics(np.array(pred["y_true"]), np.array(pred["genuine_score"]))["accuracy"]
        row = {"epoch": ep, "train_loss": running / max(n, 1), "val_acc": acc}
        history.append(row)
        print(f"{kind} epoch {ep}/{epochs} loss={row['train_loss']:.4f} val_acc={acc:.4f}", flush=True)
        if acc >= best:
            best = acc
            payload = {
                "model": model.state_dict(),
                "val_acc": acc,
                "epoch": ep,
                "kind": kind,
                "fusion": fusion,
                "seed": seed,
                "n_views_train": n_views,
            }
            torch.save(payload, ckpt)
    return {"best_val_acc": best, "history": history, "ckpt": str(ckpt)}


def load_camva(ckpt: Path, fusion: str) -> CAMVANet:
    net = CAMVANet(freeze_cnn=True, fusion=fusion).to(DEVICE)
    blob = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    net.load_state_dict(blob["model"])
    net.eval()
    return net


def load_baseline(ckpt: Path) -> MultiViewCNNVIT:
    net = MultiViewCNNVIT(freeze_cnn=True).to(DEVICE)
    blob = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    net.load_state_dict(blob["model"])
    net.eval()
    return net


@torch.inference_mode()
def adaptive_predict(
    model: CAMVANet,
    note_ids: list[str],
    records: dict,
    *,
    threshold: float,
    order: str,
    temperature: float = 1.0,
    max_views: int = 6,
) -> dict:
    """Sequential fusion. Stop uses only views already consumed.

    order=confidence ranks *stored* views by quality_head (offline dataset).
    That uses each view image to score reliability before fusion order is
    chosen, which is valid only because JaalTaka already captured all views.
    Causal live capture must use fixed or random order instead.
    """
    model.eval()
    ds = NoteViewDataset(note_ids, records, n_views=None, train=False, view_order="fixed")
    ys, scores, n_used, ids, stopped, times, logits_out = [], [], [], [], [], [], []
    for i, nid in enumerate(note_ids):
        item = ds[i]
        views = item["views"].to(DEVICE)
        y = int(item["label"])
        vcount = int(views.size(0))
        order_idx = list(range(vcount))
        if order == "random":
            rng = np.random.default_rng(abs(hash((nid, "rand"))) % (2**32))
            rng.shuffle(order_idx)
        elif order == "confidence":
            z_all = model.encoder(views)
            q = torch.sigmoid(model.quality_head(z_all).squeeze(-1))
            order_idx = torch.argsort(q, descending=True).cpu().tolist()
        last_logit = None
        used = 0
        t0 = time.perf_counter()
        for step in range(min(vcount, max_views)):
            prefix = [order_idx[j] for j in range(step + 1)]
            z = model.encoder(views[prefix]).unsqueeze(0)
            mask = torch.ones(1, z.size(1), device=DEVICE, dtype=torch.long)
            fused, _, _ = model.fuse(z, mask)
            last_logit = model.classifier(fused)
            used = step + 1
            conf = float(F.softmax(last_logit / temperature, dim=1)[0].max().item())
            if conf >= threshold:
                break
        dt = (time.perf_counter() - t0) * 1000.0
        p = float(F.softmax(last_logit / temperature, dim=1)[0, 1].item())
        ys.append(y)
        scores.append(p)
        logits_out.append(last_logit.cpu()[0].tolist())
        n_used.append(used)
        ids.append(nid)
        stopped.append(used < min(vcount, max_views))
        times.append(dt)
    return {
        "note_id": ids,
        "y_true": ys,
        "genuine_score": scores,
        "logits": logits_out,
        "n_views": n_used,
        "stopped_early": stopped,
        "latency_ms_per_note": times,
        "threshold": threshold,
        "order": order,
        "pct_used_all_six": float(np.mean([n >= 6 for n in n_used])),
        "average_views": float(np.mean(n_used)),
    }


def save_plots_views(rows: list[dict], path: Path, title: str, ykey: str, ylabel: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    xs = [r["n_views"] for r in rows]
    ys = [r[ykey] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Number of views")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_adaptive_plot(rows: list[dict], path: Path, ykey: str, ylabel: str, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([r["average_views"] for r in rows], [r[ykey] for r in rows], marker="o")
    for r in rows:
        ax.annotate(f"t={r['threshold']}", (r["average_views"], r[ykey]), fontsize=8)
    ax.set_xlabel("Average number of views")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
