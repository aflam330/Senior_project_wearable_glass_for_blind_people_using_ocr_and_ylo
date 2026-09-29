"""Close-up denomination classifier for hand-held, partial Taka photos.

Trains MobileNetV3-Small (ImageNet init, all layers) on NSTU-BDTAKA Recognition/train.
Validation is 10% of NSTU train *original photos* (Roboflow copies of one photo stay on one
side). Epoch selection uses validation accuracy only. NSTU test and Bangla Money are not
read here. Output: models/closeup_mobilenet.pt  {"state_dict", "classes", "val_acc", "epoch"}
(classes in numeric order, same checkpoint format CurrencyMode already reads).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
REC = ROOT.parent / "data set for comparison/NSTU-BDTAKA dataset Mendeley/data/Recognition"
CLASSES = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_TF = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.6, 1.0)), transforms.RandomRotation(15),
                               transforms.ColorJitter(0.3, 0.3, 0.2, 0.03), transforms.ToTensor(), NORM])
EVAL_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])


def oid(p: Path) -> str:
    return re.split(r"_(?:jpg|jpeg|png)\.rf\.", p.name)[0]


class Files(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def split_train(seed):
    groups = {}
    for folder in sorted(x for x in (REC / "train").iterdir() if x.is_dir()):
        y = CLASSES.index(int(folder.name.split("_")[0]))
        for p in folder.glob("*"):
            groups.setdefault((folder.name, oid(p)), []).append((p, y))
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_val = len(keys) // 10
    val = [groups[k][0] for k in keys[:n_val]]  # one image per validation original
    train = [x for k in keys[n_val:] for x in groups[k]]
    return train, val


@torch.inference_mode()
def accuracy(model, loader, dev):
    model.eval()
    right = n = 0
    for x, y in loader:
        right += int((model(x.to(dev)).argmax(1).cpu() == y).sum())
        n += len(y)
    return right / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "models/closeup_mobilenet.pt"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val = split_train(args.seed)
    tl = DataLoader(Files(train, TRAIN_TF), batch_size=args.batch, shuffle=True, num_workers=6, persistent_workers=True)
    vl = DataLoader(Files(val, EVAL_TF), batch_size=128, num_workers=4)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, history = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            loss = nn.functional.cross_entropy(model(x), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(y)
            n += len(y)
        sched.step()
        va = accuracy(model, vl, dev)
        history.append({"epoch": ep, "train_loss": loss_sum / n, "val_acc": va, "seconds": round(time.time() - t0)})
        print(json.dumps(history[-1]), flush=True)
        if va > best:
            best = va
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "classes": [str(c) for c in CLASSES], "val_acc": va, "epoch": ep,
                        "arch": "mobilenet_v3_small", "trained_on": "NSTU-BDTAKA Recognition/train (90% of originals)",
                        "args": vars(args)}, args.out)
    Path(args.out).with_suffix(".history.json").write_text(json.dumps(
        {"train_images": len(train), "val_originals": len(val), "best_val_acc": best, "history": history}, indent=2))
    print("BEST", best, flush=True)


if __name__ == "__main__":
    main()
