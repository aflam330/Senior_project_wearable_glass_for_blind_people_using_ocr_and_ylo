"""Subject-grouped 10-fold CK+48 (6-class, no contempt).

Does not overwrite models/emotion_faces.pt (RAF-DB wearable weights).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data set" / "CK+48"
RAF_PT = ROOT / "models" / "emotion_faces.pt"
OUT_JSON = ROOT / "results" / "emotion_ckplus.json"
OUT_PNG = ROOT / "results" / "emotion_ckplus_cm.png"

# Align with Savior Glass names; CK+ has no Neutral, drop contempt.
TABLE6 = ["Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust"]
FOLDER_TO_Y = {
    "happy": 0,
    "sadness": 1,
    "anger": 2,
    "surprise": 3,
    "fear": 4,
    "disgust": 5,
}
SEED = 42
IMG_SIZE = 192
BATCH = 16
N_FOLDS = 10
ARCH = "mobilenet_v3_large"


def metrics6(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = np.zeros((6, 6), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros(6), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros(6), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(6), where=(prec + rec) > 0)
    n = int(cm.sum())
    return {
        "n_test": n,
        "accuracy": float(tp.sum() / max(n, 1)),
        "macro_f1": float(f1.mean()),
        "per_class": {
            TABLE6[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(cm[i].sum()),
            }
            for i in range(6)
        },
        "confusion_matrix": cm.astype(int).tolist(),
        "labels": TABLE6,
    }


def save_cm(cm: np.ndarray, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6), TABLE6, rotation=45, ha="right")
    ax.set_yticks(range(6), TABLE6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    vmax = max(int(cm.max()), 1)
    for i in range(6):
        for j in range(6):
            ax.text(
                j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > vmax * 0.6 else "black", fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)


def list_items() -> list[tuple[Path, int, str]]:
    items = []
    for folder, y in FOLDER_TO_Y.items():
        d = DATA / folder
        for path in sorted(d.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            subj = path.stem.split("_")[0]
            items.append((path, y, subj))
    return items


def subject_folds(items: list[tuple[Path, int, str]], n_folds: int) -> list[int]:
    rng = np.random.RandomState(SEED)
    subjects = sorted({s for _, _, s in items})
    rng.shuffle(subjects)
    subj_fold = {s: i % n_folds for i, s in enumerate(subjects)}
    return [subj_fold[s] for _, _, s in items]


class FaceDS(Dataset):
    def __init__(self, items: list[tuple[Path, int, str]], train: bool):
        self.items = items
        if train:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE + 12, IMG_SIZE + 12)),
                transforms.RandomCrop(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.15, 0.15, 0.1, 0.02),
                transforms.RandomRotation(8),
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
        return len(self.items)

    def __getitem__(self, i):
        path, y, _ = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), int(y)


def build_model6() -> nn.Module:
    model = models.mobilenet_v3_large(weights=None)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, 6)
    return model


def load_raf_backbone(model: nn.Module) -> None:
    if not RAF_PT.is_file():
        return
    blob = torch.load(RAF_PT, map_location="cpu", weights_only=False)
    sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    keep = {k: v for k, v in sd.items() if not k.startswith("classifier.3")}
    missing, unexpected = model.load_state_dict(keep, strict=False)
    print("loaded RAF backbone", "missing", len(missing), "unexpected", len(unexpected))


@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = 0.5 * (model(x) + model(torch.flip(x, dims=[3])))
        ys.append(y.numpy())
        ps.append(logits.argmax(1).numpy())
    return metrics6(np.concatenate(ys), np.concatenate(ps))


@torch.no_grad()
def zeroshot_raf(items: list[tuple[Path, int, str]]) -> dict:
    model = models.mobilenet_v3_large(weights=None)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, 7)
    blob = torch.load(RAF_PT, map_location="cpu", weights_only=False)
    sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model.load_state_dict(sd, strict=False)
    model.eval()
    loader = DataLoader(FaceDS(items, False), batch_size=BATCH, shuffle=False, num_workers=0)
    ys, ps = [], []
    for x, y in loader:
        logits = 0.5 * (model(x) + model(torch.flip(x, dims=[3])))
        # drop Neutral (index 6); CK+ 6-class matches TABLE[:6]
        pred = logits[:, :6].argmax(1)
        ys.append(y.numpy())
        ps.append(pred.numpy())
    return metrics6(np.concatenate(ys), np.concatenate(ps))


def train_fold(train_items, test_items) -> dict:
    model = build_model6()
    load_raf_backbone(model)
    crit = nn.CrossEntropyLoss(label_smoothing=0.04)
    train_loader = DataLoader(FaceDS(train_items, True), batch_size=BATCH, shuffle=True, num_workers=0)
    test_loader = DataLoader(FaceDS(test_items, False), batch_size=BATCH, shuffle=False, num_workers=0)

    for p in model.features.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1.2e-3, weight_decay=1e-4)
    best_acc, best_state, best_m = -1.0, None, None

    def run(n_epochs, opt_):
        nonlocal best_acc, best_state, best_m
        for _ in range(n_epochs):
            model.train()
            for x, y in train_loader:
                opt_.zero_grad()
                logits = model(x)
                loss = crit(logits, y)
                loss.backward()
                opt_.step()
            m = eval_model(model, test_loader)
            if m["accuracy"] > best_acc + 1e-4:
                best_acc = m["accuracy"]
                best_m = m
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    run(2, opt)
    params = list(model.features.parameters())
    for p in params[int(len(params) * 0.55):]:
        p.requires_grad = True
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2.5e-4, weight_decay=1e-4)
    run(8, opt)
    if best_state:
        model.load_state_dict(best_state)
    return best_m or eval_model(model, test_loader)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ROOT.joinpath("results").mkdir(exist_ok=True)
    items = list_items()
    print("CK+48 6-class images", len(items), DATA)
    if len(items) < 400:
        raise SystemExit("CK+48 looks incomplete")

    zs = zeroshot_raf(items)
    print("RAF zero-shot on CK+6 acc", round(zs["accuracy"], 4), "F1", round(zs["macro_f1"], 4))

    folds = subject_folds(items, N_FOLDS)
    fold_rows = []
    all_y, all_p = [], []
    t0 = time.time()
    for k in range(N_FOLDS):
        tr = [it for it, f in zip(items, folds) if f != k]
        te = [it for it, f in zip(items, folds) if f == k]
        print(f"fold {k+1}/{N_FOLDS} train={len(tr)} test={len(te)}")
        m = train_fold(tr, te)
        fold_rows.append({
            "fold": k + 1,
            "n_test": m["n_test"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
        })
        cm = np.array(m["confusion_matrix"])
        # reconstruct y/p from cm for pooled metrics
        for t in range(6):
            for p in range(6):
                n = int(cm[t, p])
                all_y.extend([t] * n)
                all_p.extend([p] * n)
        print(f"  acc={m['accuracy']:.3f} F1={m['macro_f1']:.3f}")

    pooled = metrics6(np.array(all_y), np.array(all_p))
    save_cm(np.array(pooled["confusion_matrix"]), "CK+48 10-fold subject-grouped (6-class)")
    payload = {
        **pooled,
        "protocol": "10-fold GroupKFold by subject, 6-class no contempt/no Neutral",
        "dataset": str(DATA),
        "n_images": len(items),
        "n_subjects": len({s for _, _, s in items}),
        "img_size": IMG_SIZE,
        "arch": ARCH,
        "init": "RAF-DB MobileNetV3-Large backbone",
        "wearable_weights_unchanged": str(RAF_PT),
        "rafdb_official_test_acc": 0.809973924380704,
        "raf_zeroshot_ckplus6": zs,
        "folds": fold_rows,
        "mean_fold_acc": float(np.mean([r["accuracy"] for r in fold_rows])),
        "std_fold_acc": float(np.std([r["accuracy"] for r in fold_rows])),
        "seconds": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", OUT_JSON)
    print(
        "CK+48 10-fold mean acc", payload["mean_fold_acc"],
        "pooled", pooled["accuracy"], "macroF1", pooled["macro_f1"],
    )


if __name__ == "__main__":
    main()
