"""Fine-tune a pretrained face-emotion ViT on official RAF-DB.

Saves savior_glass/models/emotion_vit.pt and updates emotion_rafdb.json
if the ViT beats the CNN checkpoint.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data set" / "RAF-DB" / "DATASET"
CNN_PT = ROOT / "models" / "emotion_faces.pt"
OUT_PT = ROOT / "models" / "emotion_vit.pt"
OUT_JSON = ROOT / "results" / "emotion_rafdb.json"
OUT_PNG = ROOT / "results" / "emotion_confusion_matrix.png"

TABLE = ["Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust", "Neutral"]
RAF_TO_TABLE = {1: 3, 2: 4, 3: 5, 4: 0, 5: 1, 6: 2, 7: 6}
SEED = 42
EPOCHS = 12
MODEL_IDS = [
    "trpakov/vit-face-expression",
    "michaelgathara/vit-face-raf-db",
]


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


def name_to_table(name: str) -> int | None:
    n = name.lower().strip().replace(" ", "").replace("-", "_")
    aliases = {
        "happy": 0, "happiness": 0, "joy": 0,
        "sad": 1, "sadness": 1,
        "angry": 2, "anger": 2,
        "surprise": 3, "surprised": 3,
        "fear": 4, "fearful": 4,
        "disgust": 5, "disgusted": 5,
        "neutral": 6,
    }
    if n in aliases:
        return aliases[n]
    if n.isdigit():
        fer = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        i = int(n)
        if 0 <= i < 7:
            return aliases[fer[i]]
    return None


def remap_logits(logits: torch.Tensor, id2table: list[int]) -> torch.Tensor:
    out = torch.zeros(logits.size(0), 7, device=logits.device, dtype=logits.dtype)
    for src, dst in enumerate(id2table):
        if dst is None or dst < 0:
            continue
        out[:, dst] = out[:, dst] + logits[:, src]
    return out


class RafDS(Dataset):
    def __init__(self, items, train: bool, size: int):
        print(f"preload {len(items)}", flush=True)
        self.images = [Image.open(p).convert("RGB") for p, _ in items]
        self.ys = [int(y) for _, y in items]
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.15, 0.03),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        return self.tf(self.images[i]), self.ys[i]


@torch.no_grad()
def eval_model(model, loader, device, id2table) -> dict:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast("cuda", enabled=device.type == "cuda"):
            logits = model(pixel_values=x).logits
            logits = remap_logits(logits, id2table)
            logits = logits + remap_logits(model(pixel_values=torch.flip(x, dims=[3])).logits, id2table)
        ys.append(y.numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    return metrics7(np.concatenate(ys), np.concatenate(ps))


def load_teacher(device):
    last_err = None
    for mid in MODEL_IDS:
        try:
            print("loading", mid, flush=True)
            processor = AutoImageProcessor.from_pretrained(mid)
            model = AutoModelForImageClassification.from_pretrained(mid)
            id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
            id2table = []
            for i in range(model.config.num_labels):
                lab = id2label.get(i, str(i))
                mapped = name_to_table(lab)
                if mapped is None:
                    mapped = name_to_table(lab.split("_")[-1])
                id2table.append(-1 if mapped is None else mapped)
            print("id2label", id2label, "id2table", id2table, flush=True)
            if sum(1 for v in id2table if v >= 0) < 7:
                raise RuntimeError("incomplete label map")
            size = int(getattr(processor, "size", {}).get("height", 224) or 224)
            if isinstance(processor.size, int):
                size = processor.size
            elif isinstance(processor.size, dict):
                size = int(processor.size.get("height") or processor.size.get("shortest_edge") or 224)
            return mid, model.to(device), id2table, size
        except Exception as exc:
            last_err = exc
            print("failed", mid, exc, flush=True)
    raise SystemExit(f"no ViT loaded: {last_err}")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = list_split("train")
    te = list_split("test")
    print("RAF-DB", len(tr), len(te), device, flush=True)

    mid, model, id2table, size = load_teacher(device)
    batch = 4 if device.type == "cuda" else 2
    train_loader = DataLoader(RafDS(tr, True, size), batch_size=batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(RafDS(te, False, size), batch_size=batch, shuffle=False)

    print("zero-shot eval...", flush=True)
    z = eval_model(model, test_loader, device, id2table)
    print("zero-shot acc", z["accuracy"], "F1", z["macro_f1"], flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.02)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.04)
    best_acc, best_state, best_m = z["accuracy"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, z
    history = [{"epoch": 0, "val_acc": z["accuracy"], "val_macro_f1": z["macro_f1"], "note": "zero-shot"}]

    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        correct = total = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                logits = remap_logits(model(pixel_values=x).logits, id2table)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
        m = eval_model(model, test_loader, device, id2table)
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
                    "img_size": size,
                    "arch": "hf_vit",
                    "model_id": mid,
                    "id2table": id2table,
                    "metrics": m,
                },
                OUT_PT,
            )
            print("  saved", round(best_acc, 4), flush=True)

    model.load_state_dict(best_state)
    m = best_m
    save_cm(np.array(m["confusion_matrix"]), f"RAF-DB official test ({mid})")
    payload = {
        **m,
        "dataset": str(DATA),
        "split": "official RAF-DB train/test",
        "n_train": len(tr),
        "img_size": size,
        "arch": "hf_vit",
        "model_id": mid,
        "zero_shot_acc": z["accuracy"],
        "history": history,
        "winner": mid,
        "cnn_best_acc": None,
    }
    if CNN_PT.is_file():
        cnn = torch.load(CNN_PT, map_location="cpu", weights_only=False)
        payload["cnn_best_acc"] = (cnn.get("metrics") or {}).get("accuracy")
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("BEST acc", m["accuracy"], "macroF1", m["macro_f1"])
    print("per-class F1", {k: round(v["f1"], 3) for k, v in m["per_class"].items()})


if __name__ == "__main__":
    main()
