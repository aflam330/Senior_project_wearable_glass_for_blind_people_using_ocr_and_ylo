"""
Currency Model Training Script
================================
Trains a MobileNetV3-Small classifier to recognize Bangladeshi Taka notes.

Dataset structure required:
  data-dir/
    10/       ← ~500+ images of 10 Taka notes
    20/       ← ~500+ images of 20 Taka notes
    50/
    100/
    200/
    500/
    1000/

Capture tips:
  • Shoot under different lighting conditions (indoor, outdoor, fluorescent)
  • Include worn, folded, and slightly crumpled notes
  • Vary distance (20–50 cm) and angle (0–30°)
  • Aim for ≥500 images per class; 1000+ gives best results

Usage:
  python3 scripts/train_currency.py --data-dir /path/to/taka_dataset
  python3 scripts/train_currency.py --data-dir /path/to/taka_dataset --epochs 30 --batch-size 16

Output:
  models/currency_mobilenet.pt   (loaded automatically by CurrencyMode)
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

DENOMINATIONS = [10, 20, 50, 100, 200, 500, 1000]
NUM_CLASSES   = len(DENOMINATIONS)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_OUT  = os.path.join(BASE_DIR, "models", "currency_mobilenet.pt")


def build_model() -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # Freeze backbone — only train the classifier head
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
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


def train(data_dir: str, epochs: int, batch_size: int, lr: float) -> None:
    device = torch.device("cpu")   # RPi 5 has no CUDA GPU
    print(f"Training on: {device}")
    print(f"Dataset: {data_dir}")
    print(f"Epochs: {epochs}  |  Batch size: {batch_size}  |  LR: {lr}")

    train_tf, val_tf = build_transforms()

    # Load full dataset with training transforms first to get class mapping
    full_ds = ImageFolder(data_dir, transform=train_tf)
    print(f"\nClasses found: {full_ds.classes}")
    print(f"Total images:  {len(full_ds)}")

    # Warn about missing classes
    expected = [str(d) for d in DENOMINATIONS]
    for cls in expected:
        if cls not in full_ds.classes:
            print(f"  WARNING: Class '{cls}' not found in dataset. Add images to {data_dir}/{cls}/")

    # 80/20 train-val split
    n_val   = max(1, int(len(full_ds) * 0.2))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # Apply val transforms to validation subset
    val_ds.dataset = ImageFolder(data_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=False)

    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
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
            os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  ✓ Best model saved → {MODEL_OUT}  (val acc: {val_acc:.1f}%)")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved: {MODEL_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BDT currency classifier")
    parser.add_argument("--data-dir",   required=True,
                        help="Path to dataset directory (subdirs: 10, 20, 50, …, 1000)")
    parser.add_argument("--epochs",     type=int,   default=25)
    parser.add_argument("--batch-size", type=int,   default=8,
                        help="Smaller batch (8) fits in RPi 5 RAM")
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: data-dir not found: {args.data_dir}")
        sys.exit(1)

    train(args.data_dir, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
