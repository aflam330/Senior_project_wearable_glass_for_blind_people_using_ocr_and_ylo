"""Validation-ranked watermark architecture search, then one hybrid refit.

Ten candidates, three seeds (42, 43, 44), eight epochs, AdamW 3e-4. The kept epoch is the
best validation AUC. The cited architecture is the one with the highest mean validation
AUC. Test accuracy is recorded after that choice and is not used to pick the winner.

Candidates:
  resnet18, resnet34, efficientnet_b0, mobilenet_v3_large, mobilenet_v3_small,
  densenet121, convnext_tiny, swin_t, mobilenet_v2 with CLAHE, and a small
  CNN-transformer trained from scratch.
EfficientFormer, LeViT, ViT-Small and ConvNeXt-Nano are not in this torchvision build
and are not scored.

The hybrid refits a logistic regression on validation only:
  [prefix-network logit at k views, mean watermark probability of the winning
  architecture, missing-crop flag].
It is compared with the saved full-resolution ResNet-50 fine-tune on the same notes.

Output: results/watermark_arch/summary.json
Published watermark and fine-tune files are not overwritten.
"""
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from roboeye.camva.notes import load_splits  # noqa: E402
from hybrid_serial_split import RUNS, mcnemar, preds  # noqa: E402

WM = ROOT / "results" / "watermark"
OUT = ROOT / "results" / "watermark_arch"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (42, 43, 44)
EPOCHS = 8
ARCHS = (
    "resnet18", "resnet34", "efficientnet_b0", "mobilenet_v3_large", "mobilenet_v3_small",
    "densenet121", "convnext_tiny", "swin_t", "mobilenet_v2_clahe", "cnn_transformer",
)
NORM = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
TR = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomRotation(6),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.03),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    transforms.ToTensor(), NORM,
])
EV = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def metrics(prob, y, threshold=0.5):
    pred = prob >= threshold
    genuine = y == 1
    return {
        "accuracy": float((pred == genuine).mean()),
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else None,
        "false_counterfeit_on_genuine": int((~pred & genuine).sum()),
        "genuine_n": int(genuine.sum()),
        "counterfeit_missed": int((pred & ~genuine).sum()),
        "counterfeit_n": int((~genuine).sum()),
    }


class CNNTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=4), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.pos = nn.Parameter(torch.zeros(1, 196, 64))
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        tok = self.stem(x).flatten(2).transpose(1, 2) + self.pos
        return self.fc(self.enc(tok).mean(1))


def build(name: str) -> nn.Module:
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    elif name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
    elif name == "swin_t":
        m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        m.head = nn.Linear(m.head.in_features, 2)
    elif name == "mobilenet_v2_clahe":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    elif name == "cnn_transformer":
        m = CNNTransformer()
    else:
        raise KeyError(name)
    return m


def clahe(img: Image.Image) -> Image.Image:
    rgb = np.array(img)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = CLAHE.apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


class Crops(Dataset):
    def __init__(self, items, tf, use_clahe: bool):
        self.items, self.tf, self.use_clahe = items, tf, use_clahe

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        nid, y = self.items[i]
        img = Image.open(WM / "crops" / (nid.replace(":", "_") + ".png")).convert("RGB")
        if self.use_clahe:
            img = clahe(img)
        return self.tf(img), y


@torch.inference_mode()
def predict(model, loader):
    model.eval()
    prob, y = [], []
    for x, t in loader:
        logit = model(x.to(DEV))
        prob.append(torch.softmax(logit, 1)[:, 1].cpu().numpy())
        y.append(t.numpy())
    return np.concatenate(prob), np.concatenate(y)


def train_one(name, seed, items):
    last_error = None
    for batch in (16, 8, 4):
        try:
            return _train_one(name, seed, items, batch)
        except RuntimeError as exc:
            last_error = exc
            torch.cuda.empty_cache()
            if "out of memory" not in str(exc).lower():
                raise
            print(name, seed, "OOM at batch", batch, "retrying", flush=True)
    raise last_error


def _train_one(name, seed, items, batch):
    torch.manual_seed(seed)
    np.random.seed(seed)
    use_clahe = name == "mobilenet_v2_clahe"
    loaders = {
        split: DataLoader(
            Crops(items[split], TR if split == "train" else EV, use_clahe),
            batch_size=batch if split == "train" else max(batch, 8), shuffle=(split == "train"),
        )
        for split in ("train", "val", "test")
    }
    model = build(name).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3 if name == "cnn_transformer" else 3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    best, hist = (-1.0, None), []
    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in loaders["train"]:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = nn.functional.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        pv, yv = predict(model, loaders["val"])
        auc = float(roc_auc_score(yv, pv))
        hist.append({"epoch": ep, "val_auc": auc, "val_acc": float(((pv >= 0.5) == (yv == 1)).mean())})
        print(name, seed, "batch", batch, hist[-1], flush=True)
        if auc > best[0]:
            best = (auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    pv, yv = predict(model, loaders["val"])
    pt, yt = predict(model, loaders["test"])
    nparam = sum(p.numel() for p in model.parameters())
    del model
    torch.cuda.empty_cache()
    return {
        "arch": name, "seed": seed, "best_val_auc": best[0], "n_params": nparam, "history": hist,
        "val_note_ids": [n for n, _ in items["val"]], "val_probs": np.round(pv, 6).tolist(),
        "test_note_ids": [n for n, _ in items["test"]], "test_probs": np.round(pt, 6).tolist(),
        "val": metrics(pv, yv), "test": metrics(pt, yt),
    }


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def align(ids, table, default=0.5):
    return np.array([table.get(n, default) for n in ids], dtype=np.float64)


def hybrid_and_baseline(winner_runs, records):
    ids = {}
    splits, _ = load_splits(ROOT / "results" / "serial_split")
    ids = {s: list(splits[s]) for s in ("val", "test")}
    y = {s: np.array([int(records[n]["label"]) for n in ids[s]]) for s in ids}
    ok = {r["note_id"] for r in json.loads((WM / "features.json").read_text(encoding="utf-8")) if r["ok"]}
    has = {s: np.array([n in ok for n in ids[s]]) for s in ids}
    tables = {}
    for split in ("val", "test"):
        mats = []
        for run in winner_runs:
            mats.append(align(ids[split], dict(zip(run[f"{split}_note_ids"], run[f"{split}_probs"]))))
        tables[split] = np.mean(mats, axis=0)
    per_seed = {}
    for seed, run in RUNS.items():
        block = {}
        for k in (1, 6):
            net = {s: np.array([preds(run, f"{s}_views", k)[n] for n in ids[s]]) for s in ids}
            feat = {s: np.column_stack([logit(net[s]), logit(tables[s]), (~has[s]).astype(float)]) for s in ids}
            hy = LogisticRegression(max_iter=2000).fit(feat["val"], y["val"])
            ph = hy.predict_proba(feat["test"])[:, 1]
            ft = json.loads((ROOT / "results" / "serial_split" / "ft_resnet50_fullres" / f"seed{seed}.json").read_text(encoding="utf-8"))
            ft_ids = ft["test_note_ids"]
            ft_p = np.array([float(np.mean(row[:k])) for row in ft["test_probs"]])
            ft_y = np.array([int(records[n]["label"]) for n in ft_ids])
            ft_acc = float(((ft_p >= 0.5) == (ft_y == 1)).mean())
            if abs(ft_acc - ft["test"][str(k)]) > 1e-9:
                raise RuntimeError(f"fine-tune probability index mismatch seed {seed} k {k}: {ft_acc} vs {ft['test'][str(k)]}")
            order = [ft_ids.index(n) for n in ids["test"]]
            ft_aligned = ft_p[order]
            right_h = (ph >= 0.5) == (y["test"] == 1)
            right_ft = (ft_aligned >= 0.5) == (y["test"] == 1)
            n01, n10, p = mcnemar(right_h, right_ft)
            block[f"k{k}"] = {
                "hybrid": metrics(ph, y["test"]),
                "network": metrics(net["test"], y["test"]),
                "ft_resnet50_fullres": metrics(ft_aligned, y["test"]),
                "mcnemar_hybrid_vs_ft": {"n01_hybrid_only_correct": n01, "n10_ft_only_correct": n10, "p": p},
            }
            print("hybrid", seed, k, block[f"k{k}"]["hybrid"]["accuracy"], "ft", block[f"k{k}"]["ft_resnet50_fullres"]["accuracy"], "p", p, flush=True)
        per_seed[str(seed)] = block
    return per_seed


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    splits, records = load_splits(ROOT / "results" / "serial_split")
    ok = {r["note_id"] for r in json.loads((WM / "features.json").read_text(encoding="utf-8")) if r["ok"]}
    items = {s: [(n, int(records[n]["label"])) for n in splits[s] if n in ok] for s in ("train", "val", "test")}
    runs = []
    for name in ARCHS:
        for seed in SEEDS:
            path = OUT / f"{name}_seed{seed}.json"
            if path.exists():
                runs.append(json.loads(path.read_text(encoding="utf-8")))
                print("loaded", path.name, "val_auc", runs[-1].get("best_val_auc"), flush=True)
                continue
            try:
                row = train_one(name, seed, items)
            except RuntimeError as exc:
                torch.cuda.empty_cache()
                row = {"arch": name, "seed": seed, "error": str(exc), "trace": traceback.format_exc()[-1500:]}
                print("FAILED", name, seed, exc, flush=True)
            path.write_text(json.dumps(row), encoding="utf-8")
            runs.append(row)
    grouped = {}
    for row in runs:
        grouped.setdefault(row["arch"], []).append(row)
    ranking = []
    for name, group in grouped.items():
        good = [r for r in group if "best_val_auc" in r]
        if len(good) != len(SEEDS):
            ranking.append({"arch": name, "mean_val_auc": None, "error": "incomplete"})
            continue
        ranking.append({
            "arch": name,
            "mean_val_auc": float(np.mean([r["best_val_auc"] for r in good])),
            "std_val_auc": float(np.std([r["best_val_auc"] for r in good], ddof=1)),
            "mean_test_acc": float(np.mean([r["test"]["accuracy"] for r in good])),
            "std_test_acc": float(np.std([r["test"]["accuracy"] for r in good], ddof=1)),
            "mean_test_auc": float(np.mean([r["test"]["auc"] for r in good])),
            "n_params": good[0]["n_params"],
        })
    ranked = sorted([r for r in ranking if r["mean_val_auc"] is not None], key=lambda r: r["mean_val_auc"], reverse=True)
    winner = ranked[0]["arch"]
    winner_runs = [r for r in grouped[winner] if "best_val_auc" in r]
    ids = winner_runs[0]["test_note_ids"]
    mean_test = np.mean([align(ids, dict(zip(r["test_note_ids"], r["test_probs"]))) for r in winner_runs], axis=0)
    y_test = np.array([int(records[n]["label"]) for n in ids])
    per_seed = hybrid_and_baseline(winner_runs, records)
    summary = {
        "selection": "highest mean validation AUC across seeds 42, 43, 44; test not used",
        "epochs": EPOCHS,
        "unavailable_not_scored": ["EfficientFormer", "LeViT", "ViT-Small", "ConvNeXt-Nano", "ResNet-50", "EfficientNet-B1", "EfficientNet-B2"],
        "ranking_by_val_auc": ranked,
        "winner": winner,
        "winner_mean_test": metrics(mean_test, y_test),
        "hybrid_per_network_seed": per_seed,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({"winner": winner, "ranking": ranked, "winner_test": summary["winner_mean_test"]}, indent=1))


if __name__ == "__main__":
    main()
