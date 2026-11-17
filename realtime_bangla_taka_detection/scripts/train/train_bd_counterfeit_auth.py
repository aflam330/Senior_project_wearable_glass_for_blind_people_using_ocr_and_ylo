"""
Train REAL vs COUNTERFEIT authenticity on:
  - Fakes:  data set/BD_Counterfeit  (your new fake-only set)
  - Reals:  Bangladeshi_Paper_Currency_Raw (matched 80/denom + same angle augs)

Note-level / print-group locked splits. Saves:
  models/bd_counterfeit_auth.pt
  results/bd_counterfeit_auth_metrics.json

Safe verdict policy (for the glass):
  only say REAL or FAKE when confidence >= --min-conf (default 0.90);
  otherwise UNCERTAIN / check by hand.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
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

from roboeye.config import BANGATAKA_DIR, DATA_SET, DEVICE, MODELS_DIR

FAKE_ROOT = DATA_SET / "BD_Counterfeit"
OUT_WEIGHTS = MODELS_DIR / "bd_counterfeit_auth.pt"
OUT_METRICS = ROOT / "results" / "bd_counterfeit_auth_metrics.json"
MANIFEST = DATA_SET / "BD_Auth_Combined" / "manifest.csv"
ANGLES = (-30, -15, 15, 30)
SEED = 42
IMG = 224


def set_seed(s: int = SEED) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def rotate_bgr(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def match_l_mean_std(bgr: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    m, s = float(L.mean()), float(L.std() + 1e-6)
    lab[:, :, 0] = np.clip((L - m) * (target_std / s) + target_mean, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def fake_target_light() -> tuple[float, float]:
    """Estimate target L stats from a sample of fake normals."""
    means, stds = [], []
    for denom in ("200", "500", "1000"):
        paths = sorted((FAKE_ROOT / denom).glob("F*_normal.jpg"))[:20]
        for p in paths:
            im = cv2.imread(str(p))
            if im is None:
                continue
            L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
            means.append(float(L.mean()))
            stds.append(float(L.std() + 1e-6))
    if not means:
        return 150.0, 45.0
    return float(np.median(means)), float(np.clip(np.median(stds), 30, 60))


def build_manifest(force: bool = False) -> list[dict]:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    real_cache = MANIFEST.parent / "real_aug"
    if MANIFEST.exists() and not force:
        with MANIFEST.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    # --- FAKE rows from BD_Counterfeit labels.csv (keep its splits) ---
    fake_csv = FAKE_ROOT / "labels.csv"
    rows: list[dict] = []
    with fake_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "path": str((FAKE_ROOT / r["file"]).resolve()),
                    "note_id": r["note_id"],
                    "denom": r["denomination"],
                    "label": "0",  # counterfeit
                    "split": r["split"],
                    "view": r["view"],
                    "print_group": r.get("print_group", ""),
                    "source": "BD_Counterfeit",
                }
            )

    # --- REAL: 80 per denom from Paper Currency Raw + same angle augs ---
    t_mean, t_std = fake_target_light()
    print(f"Matching real-note light to fake target L mean={t_mean:.1f} std={t_std:.1f}")
    real_cache.mkdir(parents=True, exist_ok=True)

    def real_split(note_id: str) -> str:
        h = int(hashlib.sha256(f"real:{SEED}:{note_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if h < 0.70:
            return "train"
        if h < 0.85:
            return "val"
        return "test"

    for denom in ("200", "500", "1000"):
        src_dir = BANGATAKA_DIR / denom
        imgs = sorted(src_dir.glob("*.jpg"))[:80]
        if len(imgs) < 80:
            print(f"WARNING: only {len(imgs)} real {denom} images")
        for i, src in enumerate(imgs, start=1):
            note_id = f"R{denom}_{i:04d}"
            split = real_split(note_id)
            bgr = cv2.imread(str(src))
            if bgr is None:
                continue
            bgr = match_l_mean_std(bgr, t_mean, t_std)
            variants = [("normal", bgr)] + [(f"rot{a:+d}".replace("+", "p").replace("-", "m"), rotate_bgr(bgr, float(a))) for a in ANGLES]
            for view, im in variants:
                out = real_cache / denom / f"{note_id}_{view}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    cv2.imwrite(str(out), im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                rows.append(
                    {
                        "path": str(out.resolve()),
                        "note_id": note_id,
                        "denom": denom,
                        "label": "1",  # genuine/real
                        "split": split,
                        "view": view,
                        "print_group": "NA",
                        "source": "PaperCurrencyRaw",
                    }
                )

    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["path", "note_id", "denom", "label", "split", "view", "print_group", "source"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {MANIFEST} ({len(rows)} rows)")
    return rows


class AuthDataset(Dataset):
    def __init__(self, rows: list[dict], train: bool):
        self.rows = rows
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((IMG, IMG)),
                    transforms.RandomResizedCrop(IMG, scale=(0.75, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
                    transforms.ToTensor(),
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

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")
        x = self.tf(img)
        y = int(r["label"])
        return x, y, r["note_id"], r["denom"]


def build_model() -> nn.Module:
    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        m = models.efficientnet_b0(weights=weights)
    except Exception:
        m = models.efficientnet_b0(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, 2)
    return m


@torch.inference_mode()
def eval_loader(model, loader, device) -> dict:
    model.eval()
    ys, ps, confs, notes = [], [], [], []
    for x, y, note_id, _denom in loader:
        logits = model(x.to(device))
        prob = F.softmax(logits, dim=1)
        pred = prob.argmax(1).cpu()
        conf = prob.max(1).values.cpu()
        ys.extend(y.tolist())
        ps.extend(pred.tolist())
        confs.extend(conf.tolist())
        notes.extend(note_id)
    ys = np.array(ys)
    ps = np.array(ps)
    confs = np.array(confs)
    acc = float((ys == ps).mean()) if len(ys) else 0.0
    # note-level majority vote
    by = defaultdict(list)
    for n, y, p in zip(notes, ys, ps):
        by[n].append((int(y), int(p)))
    n_ok = n_tot = 0
    for pairs in by.values():
        y = pairs[0][0]
        vote = Counter(p for _, p in pairs).most_common(1)[0][0]
        n_ok += int(vote == y)
        n_tot += 1
    note_acc = n_ok / max(n_tot, 1)

    # fake=0 metrics
    fake_mask = ys == 0
    real_mask = ys == 1
    fake_recall = float((ps[fake_mask] == 0).mean()) if fake_mask.any() else 0.0
    real_recall = float((ps[real_mask] == 1).mean()) if real_mask.any() else 0.0
    return {
        "image_acc": acc,
        "note_acc": note_acc,
        "fake_recall": fake_recall,
        "real_recall": real_recall,
        "n_images": int(len(ys)),
        "n_notes": int(n_tot),
        "mean_conf": float(confs.mean()) if len(confs) else 0.0,
    }


def train(args) -> dict:
    set_seed(SEED)
    rows = build_manifest(force=args.rebuild_manifest)
    splits = {
        "train": [r for r in rows if r["split"] == "train"],
        "val": [r for r in rows if r["split"] == "val"],
        "test": [r for r in rows if r["split"] == "test"],
    }
    print({k: len(v) for k, v in splits.items()})
    print(
        "train labels",
        Counter(r["label"] for r in splits["train"]),
        "sources",
        Counter(r["source"] for r in splits["train"]),
    )

    train_loader = DataLoader(
        AuthDataset(splits["train"], train=True),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(AuthDataset(splits["val"], train=False), batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(AuthDataset(splits["test"], train=False), batch_size=args.batch, shuffle=False)

    model = build_model().to(DEVICE)
    # freeze early features first epochs optional — train all for best accuracy
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best = {"val_note_acc": -1.0}
    OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    patience, bad = args.patience, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y, _, _ in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
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
            f"epoch {epoch}/{args.epochs}  loss={running / max(n, 1):.4f}  "
            f"val_img={val_m['image_acc']:.3f}  val_note={val_m['note_acc']:.3f}  "
            f"fake_rec={val_m['fake_recall']:.3f}  real_rec={val_m['real_recall']:.3f}"
        )
        score = val_m["note_acc"]
        # Strict improvement only — plateau at 100% val must still count toward patience.
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
                    "min_conf_default": args.min_conf,
                },
                OUT_WEIGHTS,
            )
            bad = 0
        else:
            if best["val_note_acc"] < 0:
                # first epoch: always save even if score==0
                best = {"val_note_acc": score, **val_m, "epoch": epoch}
                torch.save(
                    {
                        "model": model.state_dict(),
                        "arch": "efficientnet_b0",
                        "classes": ["counterfeit", "genuine"],
                        "img_size": IMG,
                        "val": val_m,
                        "epoch": epoch,
                        "min_conf_default": args.min_conf,
                    },
                    OUT_WEIGHTS,
                )
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {epoch}")
                break

    blob = torch.load(OUT_WEIGHTS, map_location=DEVICE, weights_only=False)
    model.load_state_dict(blob["model"])
    test_m = eval_loader(model, test_loader, DEVICE)

    # confidence-gated accuracy on test
    model.eval()
    gated_correct = gated_total = abstain = 0
    with torch.inference_mode():
        for x, y, _, _ in test_loader:
            prob = F.softmax(model(x.to(DEVICE)), dim=1)
            conf, pred = prob.max(1)
            for c, p, yy in zip(conf.cpu().tolist(), pred.cpu().tolist(), y.tolist()):
                if c < args.min_conf:
                    abstain += 1
                    continue
                gated_total += 1
                gated_correct += int(p == yy)
    gated_acc = gated_correct / max(gated_total, 1)
    coverage = gated_total / max(gated_total + abstain, 1)

    metrics = {
        "best_val": best,
        "test": test_m,
        "gated": {
            "min_conf": args.min_conf,
            "accuracy_when_decided": gated_acc,
            "coverage": coverage,
            "abstain": abstain,
            "decided": gated_total,
        },
        "weights": str(OUT_WEIGHTS),
        "manifest": str(MANIFEST),
        "device": str(DEVICE),
        "note": (
            "Target 95% is aspirational. Report these measured numbers. "
            "Glass should say check-by-hand when abstaining."
        ),
    }
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {OUT_WEIGHTS}")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min-conf", type=float, default=0.90)
    ap.add_argument("--rebuild-manifest", action="store_true")
    args = ap.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
