"""Short hard-augmentation fine-tune of bd_counterfeit_auth.pt for stronger field robustness.

Keeps the same print-locked splits. Saves:
  models/bd_counterfeit_auth_hard.pt
  and overwrites models/bd_counterfeit_auth.pt if --promote
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.config import DATA_SET, DEVICE, MODELS_DIR
from scripts.train.train_bd_counterfeit_auth import AuthDataset, build_model, eval_loader

MANIFEST = DATA_SET / "BD_Auth_Combined" / "manifest.csv"
IN_WEIGHTS = MODELS_DIR / "bd_counterfeit_auth.pt"
OUT_HARD = MODELS_DIR / "bd_counterfeit_auth_hard.pt"
OUT_METRICS = ROOT / "results" / "strengthen" / "bd_auth_harden_metrics.json"
IMG = 224
SEED = 42


class HardAuthDataset(Dataset):
    def __init__(self, rows: list[dict], train: bool):
        self.rows = rows
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((IMG + 32, IMG + 32)),
                    transforms.RandomResizedCrop(IMG, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(degrees=25, translate=(0.08, 0.08), scale=(0.9, 1.1)),
                    transforms.ColorJitter(0.35, 0.35, 0.35, 0.08),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((IMG, IMG)),
                    transforms.CenterCrop(IMG),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tf(img), int(r["label"]), r["note_id"], r["denom"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--promote", action="store_true", help="Also copy best hard weights to bd_counterfeit_auth.pt")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    splits = {k: [r for r in rows if r["split"] == k] for k in ("train", "val", "test")}
    print({k: len(v) for k, v in splits.items()})

    model = build_model().to(DEVICE)
    if IN_WEIGHTS.is_file():
        blob = torch.load(IN_WEIGHTS, map_location=DEVICE, weights_only=False)
        model.load_state_dict(blob["model"])
        print(f"Loaded {IN_WEIGHTS}")

    train_loader = DataLoader(
        HardAuthDataset(splits["train"], True),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(HardAuthDataset(splits["val"], False), batch_size=args.batch)
    test_loader = DataLoader(HardAuthDataset(splits["test"], False), batch_size=args.batch)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best = {"val_note_acc": -1.0}
    bad = 0
    OUT_HARD.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y, _, _ in tqdm(train_loader, desc=f"harden {epoch}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(y)
            n += len(y)
        sched.step()
        val_m = eval_loader(model, val_loader, DEVICE)
        print(
            f"epoch {epoch}/{args.epochs} loss={running / max(n, 1):.4f} "
            f"val_note={val_m['note_acc']:.3f} img={val_m['image_acc']:.3f}"
        )
        score = val_m["note_acc"]
        if score > best["val_note_acc"]:
            best = {"val_note_acc": score, **val_m, "epoch": epoch}
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": "efficientnet_b0",
                    "classes": ["counterfeit", "genuine"],
                    "img_size": IMG,
                    "val": val_m,
                    "epoch": epoch,
                    "min_conf_default": 0.90,
                    "hardened": True,
                },
                OUT_HARD,
            )
            bad = 0
        else:
            if best["val_note_acc"] < 0:
                best = {"val_note_acc": score, **val_m, "epoch": epoch}
                torch.save({"model": model.state_dict(), "arch": "efficientnet_b0", "hardened": True}, OUT_HARD)
            bad += 1
            if bad >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

    blob = torch.load(OUT_HARD, map_location=DEVICE, weights_only=False)
    model.load_state_dict(blob["model"])
    test_m = eval_loader(model, test_loader, DEVICE)
    metrics = {"best_val": best, "test": test_m, "weights": str(OUT_HARD)}
    OUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    if args.promote:
        import shutil

        shutil.copy2(OUT_HARD, IN_WEIGHTS)
        print(f"Promoted -> {IN_WEIGHTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
