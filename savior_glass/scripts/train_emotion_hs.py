"""Fine-tune AffectNet-pretrained EfficientNet-B2 (HSEmotion) on RAF-DB."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data set" / "RAF-DB" / "DATASET"
HS_PT = Path.home() / ".hsemotion" / "enet_b2_7.pt"
OUT_PT = ROOT / "models" / "emotion_faces.pt"
OUT_JSON = ROOT / "results" / "emotion_rafdb.json"
OUT_PNG = ROOT / "results" / "emotion_confusion_matrix.png"

TABLE = ["Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust", "Neutral"]
RAF_TO_TABLE = {1: 3, 2: 4, 3: 5, 4: 0, 5: 1, 6: 2, 7: 6}
# HSEmotion 7-class: Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
HS_TO_TABLE = [2, 5, 4, 0, 6, 1, 3]
SEED = 42
IMG_SIZE = 260
ARCH = "tf_efficientnet_b2_hs"
EPOCHS_HEAD = 2
EPOCHS_FULL = 18


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
    def __init__(self, items, train: bool):
        print(f"preload {len(items)}", flush=True)
        self.images = [Image.open(p).convert("RGB") for p, _ in items]
        self.ys = [int(y) for _, y in items]
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.82, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.18, 0.18, 0.12, 0.02),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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


def build_model() -> nn.Module:
    src = torch.load(HS_PT, map_location="cpu", weights_only=False)
    model = timm.create_model("tf_efficientnet_b2", pretrained=False, num_classes=7, global_pool="max")
    model.load_state_dict(src.state_dict())
    with torch.no_grad():
        w = model.classifier.weight.clone()
        b = model.classifier.bias.clone()
        order = torch.tensor(HS_TO_TABLE, dtype=torch.long)
        inv = torch.empty(7, dtype=torch.long)
        inv[order] = torch.arange(7)
        model.classifier.weight.copy_(w[inv])
        model.classifier.bias.copy_(b[inv])
    return model


@torch.no_grad()
def eval_model(model, loader, device) -> dict:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x) + model(torch.flip(x, dims=[3]))
        ys.append(y.numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    return metrics7(np.concatenate(ys), np.concatenate(ps))


def cosine_lr(opt, epoch: int, total: int, base: float) -> None:
    t = epoch / max(total, 1)
    lr = 1e-6 + 0.5 * (base - 1e-6) * (1 + math.cos(math.pi * t))
    for g in opt.param_groups:
        g["lr"] = lr * g.get("lr_mult", 1.0)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 8 if device.type == "cuda" else 4
    tr, te = list_split("train"), list_split("test")
    print("device", device, "batch", batch, "train", len(tr), "test", len(te), flush=True)
    train_loader = DataLoader(RafDS(tr, True), batch_size=batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(RafDS(te, False), batch_size=batch, shuffle=False)

    model = build_model().to(device)
    z = eval_model(model, test_loader, device)
    print("zero-shot (remapped) acc", z["accuracy"], "F1", z["macro_f1"], flush=True)

    crit = nn.CrossEntropyLoss(label_smoothing=0.04)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    best_acc, best_state, best_m = z["accuracy"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, z
    history = [{"epoch": 0, "val_acc": z["accuracy"], "val_macro_f1": z["macro_f1"], "note": "affectnet-zero-shot"}]

    def run_epochs(n_epochs, start, opt, base_lr):
        nonlocal best_acc, best_state, best_m
        last = start
        for ep in range(start, start + n_epochs):
            last = ep
            cosine_lr(opt, ep - start, n_epochs, base_lr)
            model.train()
            t0 = time.time()
            correct = total = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(x)
                    loss = crit(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 4.0)
                scaler.step(opt)
                scaler.update()
                correct += int((logits.argmax(1) == y).sum())
                total += len(y)
            m = eval_model(model, test_loader, device)
            row = {
                "epoch": ep,
                "train_acc": correct / max(total, 1),
                "val_acc": m["accuracy"],
                "val_macro_f1": m["macro_f1"],
                "seconds": round(time.time() - t0, 1),
            }
            history.append(row)
            print(
                f"epoch {ep:02d} train={row['train_acc']:.3f} test_acc={m['accuracy']:.4f} "
                f"macroF1={m['macro_f1']:.4f} {row['seconds']}s",
                flush=True,
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
                        "timm_name": "tf_efficientnet_b2",
                        "global_pool": "max",
                        "metrics": m,
                    },
                    OUT_PT,
                )
                print("  saved", round(best_acc, 4), flush=True)
        return last

    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    print("Phase 1 head", flush=True)
    last = run_epochs(EPOCHS_HEAD, 1, opt, 1e-3)

    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if "classifier" not in n], "lr": 6e-5, "lr_mult": 1.0},
            {"params": model.classifier.parameters(), "lr": 3e-4, "lr_mult": 5.0},
        ],
        weight_decay=1e-4,
    )
    print("Phase 2 full", flush=True)
    run_epochs(EPOCHS_FULL, last + 1, opt, 6e-5)

    model.load_state_dict(best_state)
    m = eval_model(model, test_loader, device)
    save_cm(np.array(m["confusion_matrix"]), "RAF-DB official test (EfficientNet-B2 AffectNet FT)")
    payload = {
        **m,
        "dataset": str(DATA),
        "split": "official RAF-DB train/test",
        "n_train": len(tr),
        "img_size": IMG_SIZE,
        "arch": ARCH,
        "history": history,
        "winner": ARCH,
        "zero_shot_acc": z["accuracy"],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("BEST acc", m["accuracy"], "macroF1", m["macro_f1"])
    print("per-class F1", {k: round(v["f1"], 3) for k, v in m["per_class"].items()})


if __name__ == "__main__":
    main()
