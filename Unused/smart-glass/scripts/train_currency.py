"""
Currency Model Training Script
================================
Trains a MobileNetV3-Small classifier to recognize Bangladeshi Taka notes.

Dataset structure required (one folder per denomination, folder name = Taka value):
  data-dir/
    2/  5/  10/  20/  50/  100/  200/  500/  1000/

Any subset of these folders works; the classes actually found are saved in the
checkpoint, so CurrencyMode maps output indices back to the right denomination.

Capture tips:
  • Shoot under different lighting conditions (indoor, outdoor, fluorescent)
  • Include worn, folded, and slightly crumpled notes
  • Vary distance (20–50 cm) and angle (0–30°)
  • Aim for ≥500 images per class; 1000+ gives best results

Usage:
  python3 scripts/train_currency.py --data-dir /path/to/taka_dataset
  python3 scripts/train_currency.py --data-dir /path/to/taka_dataset --epochs 30 --batch-size 16
  python3 scripts/train_currency.py --data-dir /path/to/taka_dataset --output models/my_model.pt

Output:
  models/currency_mobilenet.pt   (loaded automatically by CurrencyMode)
  Checkpoint format: {"state_dict", "classes", "val_acc", "arch"}
  An existing checkpoint is never overwritten unless --overwrite is given.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

DENOMINATIONS = [2, 5, 10, 20, 50, 100, 200, 500, 1000]

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_OUT  = os.path.join(BASE_DIR, "models", "currency_mobilenet.pt")


class DenominationFolder(ImageFolder):
    """ImageFolder whose class indices follow numeric Taka order (2, 5, 10, …),
    not string order (10, 100, 1000, 2, …). Non-numeric folders are rejected."""

    def find_classes(self, directory):
        names = [e.name for e in os.scandir(directory) if e.is_dir()]
        bad = [n for n in names if not n.isdigit()]
        if bad:
            raise ValueError(f"Folder names must be Taka values (e.g. 10, 500); got: {bad}")
        classes = sorted(names, key=int)
        return classes, {c: i for i, c in enumerate(classes)}


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # Freeze backbone — only train the classifier head
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def train(data_dir: str, epochs: int, batch_size: int, lr: float, output: str, workers: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Dataset: {data_dir}")
    print(f"Epochs: {epochs}  |  Batch size: {batch_size}  |  LR: {lr}")

    train_tf, val_tf = build_transforms()

    # Two views of the same folder: identical file order, different transforms
    train_full = DenominationFolder(data_dir, transform=train_tf)
    val_full   = DenominationFolder(data_dir, transform=val_tf)
    classes = train_full.classes
    print(f"\nClasses found: {classes}")
    print(f"Total images:  {len(train_full)}")
    if len(classes) < 2:
        print("ERROR: need at least two denomination folders")
        sys.exit(1)

    for d in DENOMINATIONS:
        if str(d) not in classes:
            print(f"  WARNING: Class '{d}' not found in dataset. Add images to {data_dir}/{d}/")
    unknown = [c for c in classes if int(c) not in DENOMINATIONS]
    if unknown:
        print(f"  WARNING: {unknown} are not Bangladeshi Taka denominations")

    # 80/20 train-val split
    n_val   = max(1, int(len(train_full) * 0.2))
    perm    = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(42)).tolist()
    train_ds = Subset(train_full, perm[n_val:])
    val_ds   = Subset(val_full,   perm[:n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=False)

    model     = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        model.features.eval()   # frozen backbone: keep its BatchNorm statistics fixed
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            pred           = out.argmax(dim=1)
            train_correct += (pred == labels).sum().item()
            train_total   += imgs.size(0)

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out   = model(imgs)
                pred  = out.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total   += imgs.size(0)

        train_acc = train_correct / train_total * 100
        val_acc   = val_correct   / val_total   * 100
        elapsed   = time.time() - t0

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Loss: {train_loss/train_total:.4f} | "
              f"Train acc: {train_acc:.1f}% | "
              f"Val acc: {val_acc:.1f}% | "
              f"{elapsed:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"state_dict": state, "classes": classes,
                        "val_acc": val_acc / 100, "arch": "mobilenet_v3_small"}, output)
            print(f"  Best model saved -> {output}  (val acc: {val_acc:.1f}%)")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BDT currency classifier")
    parser.add_argument("--data-dir",   required=True,
                        help="Path to dataset directory (subdirs: 2, 5, 10, …, 1000)")
    parser.add_argument("--epochs",     type=int,   default=25)
    parser.add_argument("--batch-size", type=int,   default=8,
                        help="Smaller batch (8) fits in RPi 5 RAM")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--workers",    type=int,   default=2)
    parser.add_argument("--output",     default=MODEL_OUT,
                        help=f"Checkpoint path (default: {MODEL_OUT})")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Allow replacing an existing checkpoint at --output")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: data-dir not found: {args.data_dir}")
        sys.exit(1)
    if os.path.exists(args.output) and not args.overwrite:
        print(f"ERROR: {args.output} already exists. Pass --overwrite to replace it, "
              f"or --output to write somewhere else.")
        sys.exit(1)

    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.output, args.workers)


if __name__ == "__main__":
    main()
