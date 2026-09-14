"""Train the JaalTaka CNN+ViT authenticity classifier (and optional dual head)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from roboeye.authenticity import (
    AUTH_WEIGHTS,
    JaalTakaMultiView,
    MultiViewCNNVIT,
    list_jaaltaka_notes,
    split_notes,
)
from roboeye.clip_zero_shot import PrototypeAuthenticator
from roboeye.config import DEVICE, DUAL_HEAD_WEIGHTS, YOLO_WEIGHTS
from roboeye.dual_head import AuthHead, _sppf_channels


@torch.inference_mode()
def evaluate(model, loader) -> float:
    model.eval()
    correct = total = 0
    for views, y in loader:
        pred = model(views.to(DEVICE)).argmax(1).cpu()
        correct += int((pred == y).sum())
        total += len(y)
    return correct / max(total, 1)


def _cap_notes(notes, max_notes: int):
    if not max_notes:
        return notes
    by = {0: [], 1: []}
    for item in notes:
        by[item[1]].append(item)
    half = max(1, max_notes // 2)
    return by[0][:half] + by[1][:half]


def train_cnn_vit(args) -> dict:
    notes = _cap_notes(list_jaaltaka_notes(), args.max_notes)
    if len(notes) < 20:
        raise SystemExit(f"Need JaalTaka notes, found {len(notes)}")
    splits = split_notes(notes)
    train_ds = JaalTakaMultiView(splits["train"], n_views=args.views, train=True)
    val_ds = JaalTakaMultiView(splits["val"], n_views=args.views, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    model = MultiViewCNNVIT(freeze_cnn=True).to(DEVICE)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc = 0.0
    AUTH_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    print(f"Device={DEVICE}  train={len(train_ds)}  val={len(val_ds)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for views, y in train_loader:
            views, y = views.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(views), y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(y)
            n += len(y)
        acc = evaluate(model, val_loader)
        print(f"epoch {epoch}/{args.epochs}  loss={running / max(n, 1):.4f}  val_acc={acc:.3f}")
        if acc >= best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "val_acc": acc, "epoch": epoch}, AUTH_WEIGHTS)
    test_loader = DataLoader(
        JaalTakaMultiView(splits["test"], n_views=args.views, train=False),
        batch_size=args.batch,
        shuffle=False,
    )
    # reload best
    blob = torch.load(AUTH_WEIGHTS, map_location=DEVICE, weights_only=True)
    model.load_state_dict(blob["model"])
    test_acc = evaluate(model, test_loader)
    print(f"best val_acc={best_acc:.3f}  test_acc={test_acc:.3f}  saved {AUTH_WEIGHTS}")
    return {"val_acc": best_acc, "test_acc": test_acc}


def train_dual_head(args) -> dict:
    """Train YOLO SPPF authenticity head on JaalTaka single views."""
    from ultralytics import YOLO
    import cv2
    from roboeye.authenticity import _view_paths

    notes = _cap_notes(list_jaaltaka_notes(), args.max_notes)
    splits = split_notes(notes)
    yolo = YOLO(str(YOLO_WEIGHTS))
    in_ch = _sppf_channels(yolo)
    head = AuthHead(in_ch=in_ch).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    feat_box = {}

    def hook(_m, _i, out):
        feat_box["f"] = out

    h = yolo.model.model[9].register_forward_hook(hook)

    def batch_iter(pairs, train: bool):
        for note_dir, label in pairs:
            paths = _view_paths(note_dir)
            if not paths:
                continue
            img = cv2.imread(str(paths[0]))
            if img is None:
                continue
            yolo.predict(img, conf=0.15, verbose=False)
            feat = feat_box.get("f")
            if feat is None:
                continue
            yield feat.detach(), torch.tensor([label], dtype=torch.long)

    best = 0.0
    for epoch in range(1, max(1, args.epochs) + 1):
        head.train()
        total = n = 0
        for feat, y in batch_iter(splits["train"], True):
            opt.zero_grad()
            logits = head(feat.to(DEVICE))
            loss = loss_fn(logits, y.to(DEVICE))
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
            if args.quick and n >= 40:
                break
        head.eval()
        correct = tot = 0
        with torch.inference_mode():
            for feat, y in batch_iter(splits["val"], False):
                pred = head(feat.to(DEVICE)).argmax(1).cpu()
                correct += int((pred == y).sum())
                tot += 1
                if args.quick and tot >= 20:
                    break
        acc = correct / max(tot, 1)
        print(f"dual-head epoch {epoch} loss={total / max(n, 1):.4f} val_acc={acc:.3f}")
        if acc >= best:
            best = acc
            torch.save({"model": head.state_dict(), "val_acc": acc}, DUAL_HEAD_WEIGHTS)
    h.remove()
    print(f"dual-head best val_acc={best:.3f} saved {DUAL_HEAD_WEIGHTS}")
    return {"val_acc": best}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--views", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-notes", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-prototypes", action="store_true")
    p.add_argument("--skip-dual-head", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 1)
        args.max_notes = args.max_notes or 80
        args.batch = min(args.batch, 8)
    metrics = train_cnn_vit(args)
    if not args.skip_prototypes:
        proto = PrototypeAuthenticator()
        blob = proto.build(max_notes_per_class=40 if args.quick else 80, views_per_note=2)
        print(f"prototypes backend={blob['backend']} n_real={blob['n_real']} n_fake={blob['n_fake']}")
    if not args.skip_dual_head and YOLO_WEIGHTS.is_file():
        metrics["dual_head"] = train_dual_head(args)
    print(metrics)


if __name__ == "__main__":
    main()
