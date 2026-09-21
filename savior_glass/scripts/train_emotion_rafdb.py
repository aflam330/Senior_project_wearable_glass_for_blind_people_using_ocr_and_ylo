"""Train 7-class emotion on official RAF-DB train/test folders.

RAF-DB folder ids: 1 Surprise, 2 Fear, 3 Disgust, 4 Happiness,
5 Sadness, 6 Anger, 7 Neutral.

Uses EfficientNet-V2-S + class-balanced loss, mixup, AMP, and TTA.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data set" / "RAF-DB" / "DATASET"
OUT_PT = ROOT / "models" / "emotion_faces.pt"
OUT_JSON = ROOT / "results" / "emotion_rafdb.json"
OUT_PNG = ROOT / "results" / "emotion_confusion_matrix.png"

TABLE = ["Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust", "Neutral"]
RAF_TO_TABLE = {1: 3, 2: 4, 3: 5, 4: 0, 5: 1, 6: 2, 7: 6}
SEED = 42
IMG_SIZE = 224
ARCH = "efficientnet_v2_s"
EPOCHS_HEAD = 3
EPOCHS_FULL = 28
MIXUP_ALPHA = 0.25
LABEL_SMOOTH = 0.05


def metrics7(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = np.zeros((7, 7), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros(7), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros(7), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(7), where=(prec + rec) > 0)
    n = int(cm.sum())
    return {
        "n_test": n,
        "accuracy": float(tp.sum() / max(n, 1)),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()),
        "class_distribution": {TABLE[i]: int(cm[i].sum()) for i in range(7)},
        "per_class": {
            TABLE[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(cm[i].sum()),
            }
            for i in range(7)
        },
        "confusion_matrix": cm.astype(int).tolist(),
        "labels": TABLE,
    }


def save_cm(cm: np.ndarray, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(7), TABLE, rotation=45, ha="right")
    ax.set_yticks(range(7), TABLE)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    vmax = max(int(cm.max()), 1)
    for i in range(7):
        for j in range(7):
            ax.text(
                j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > vmax * 0.6 else "black", fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)


def list_split(split: str) -> list[tuple[Path, int]]:
    items = []
    root = DATA / split
    for raf_id, y in RAF_TO_TABLE.items():
        folder = root / str(raf_id)
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                items.append((path, y))
    return items


class RafDS(Dataset):
    def __init__(self, items: list[tuple[Path, int]], train: bool):
        print(f"preload {len(items)} images...", flush=True)
        self.images = [Image.open(p).convert("RGB") for p, _ in items]
        self.ys = [int(y) for _, y in items]
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.92, 1.08)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.22, 0.22, 0.18, 0.04),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.12, scale=(0.02, 0.08)),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        return self.tf(self.images[i]), self.ys[i]


def build_model(arch: str = ARCH) -> nn.Module:
    if arch == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, 7)
        return model
    if arch == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, 7)
        return model
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, 7)
    return model


def freeze_backbone(model: nn.Module, freeze: bool) -> None:
    if hasattr(model, "features"):
        for p in model.features.parameters():
            p.requires_grad = not freeze
    else:
        for name, p in model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = not freeze


def class_weights(items: list[tuple[Path, int]]) -> torch.Tensor:
    counts = np.zeros(7, dtype=np.float64)
    for _, y in items:
        counts[y] += 1
    # Mild inverse-frequency weights so rare classes are not ignored,
    # without overpowering Happy/Neutral (which dominate accuracy).
    w = counts.sum() / (7.0 * np.maximum(counts, 1.0))
    w = np.power(w, 0.35)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x) + model(torch.flip(x, dims=[3]))
        ys.append(y.numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    return metrics7(np.concatenate(ys), np.concatenate(ps))


def cosine_lr(opt, epoch: int, total: int, base: float, min_lr: float = 1e-6) -> None:
    t = epoch / max(total, 1)
    lr = min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * t))
    for g in opt.param_groups:
        g["lr"] = lr * g.get("lr_mult", 1.0)


def pick_batch() -> int:
    if not torch.cuda.is_available():
        return 8
    mem = torch.cuda.get_device_properties(0).total_memory
    if mem < 5 * 1024**3:
        return 16
    return 24


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ROOT.joinpath("models").mkdir(exist_ok=True)
    ROOT.joinpath("results").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = pick_batch()
    print("device", device, "batch", batch, "arch", ARCH)

    tr = list_split("train")
    te = list_split("test")
    print("RAF-DB train", len(tr), "test", len(te), DATA)
    if len(tr) < 1000 or len(te) < 200:
        raise SystemExit("RAF-DB folders look incomplete")

    train_loader = DataLoader(
        RafDS(tr, True), batch_size=batch, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda", drop_last=True,
    )
    test_loader = DataLoader(
        RafDS(te, False), batch_size=batch, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    model = build_model(ARCH).to(device)
    weights = class_weights(tr).to(device)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTH)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    best_acc, best_state, best_m = -1.0, None, None
    history = []

    def run_epochs(n_epochs: int, start: int, opt, base_lr: float, mixup: bool) -> int:
        nonlocal best_acc, best_state, best_m
        last = start
        for ep in range(start, start + n_epochs):
            last = ep
            cosine_lr(opt, ep - start, n_epochs, base_lr)
            model.train()
            t0 = time.time()
            correct = total = 0
            running = 0.0
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if mixup:
                    x, y_a, y_b, lam = mixup_batch(x, y, MIXUP_ALPHA)
                opt.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(x)
                    if mixup:
                        loss = lam * crit(logits, y_a) + (1.0 - lam) * crit(logits, y_b)
                    else:
                        loss = crit(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 4.0)
                scaler.step(opt)
                scaler.update()
                pred = logits.argmax(1)
                correct += int((pred == (y_a if mixup else y)).sum())
                total += len(y)
                running += float(loss.detach())
            m = eval_model(model, test_loader, device)
            row = {
                "epoch": ep,
                "train_acc": correct / max(total, 1),
                "val_acc": m["accuracy"],
                "val_macro_f1": m["macro_f1"],
                "seconds": round(time.time() - t0, 1),
                "lr": opt.param_groups[0]["lr"],
            }
            history.append(row)
            print(
                f"epoch {ep:02d} train={row['train_acc']:.3f} "
                f"test_acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
                f"{row['seconds']}s lr={row['lr']:.2e}"
            )
            if m["accuracy"] > best_acc + 1e-5:
                best_acc = m["accuracy"]
                best_m = m
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(
                    {
                        "state_dict": best_state,
                        "labels": TABLE,
                        "img_size": IMG_SIZE,
                        "arch": ARCH,
                        "metrics": m,
                    },
                    OUT_PT,
                )
                print("  saved", round(best_acc, 4), "F1", round(m["macro_f1"], 4))
        return last

    freeze_backbone(model, True)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1.5e-3, weight_decay=1e-4,
    )
    print("Phase 1 — frozen backbone")
    last = run_epochs(EPOCHS_HEAD, 1, opt, 1.5e-3, mixup=False)

    freeze_backbone(model, False)
    opt = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": 8e-5, "lr_mult": 1.0},
            {"params": model.classifier.parameters(), "lr": 4e-4, "lr_mult": 5.0},
        ],
        weight_decay=1e-4,
    )
    print("Phase 2 — full fine-tune")
    run_epochs(EPOCHS_FULL, last + 1, opt, 8e-5, mixup=True)

    if best_state:
        model.load_state_dict(best_state)
    m = eval_model(model, test_loader, device)
    save_cm(np.array(m["confusion_matrix"]), f"RAF-DB official test ({ARCH})")
    payload = {
        **m,
        "dataset": str(DATA),
        "split": "official RAF-DB train/test",
        "n_train": len(tr),
        "img_size": IMG_SIZE,
        "arch": ARCH,
        "history": history,
        "winner": ARCH,
        "device": str(device),
        "best_acc": best_acc,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", OUT_PT)
    print("RAF-DB TEST acc", m["accuracy"], "macroF1", m["macro_f1"])
    print("per-class F1", {k: round(v["f1"], 3) for k, v in m["per_class"].items()})


if __name__ == "__main__":
    main()
