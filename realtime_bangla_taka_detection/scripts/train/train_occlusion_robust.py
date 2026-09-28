"""Occlusion-robust fine-tune of the deployed PRMVT (Q-DUIG prefix_ft) model.

Why the earlier attempts failed (see paper_evidence/DIAGNOSIS_OCCLUSION.md):
  * occlusion_ft: 2 epochs at lr 3e-4, occlusion on half the batches, CNN frozen;
  * occlusion_matched: stopped after 40 training notes, and its gradients went
    non-finite because of the NaN-entropy bug (fixed 2026-09-28);
  * in every run the ImageNet MobileNet, which gives 576 of the 704 feature dims per
    view, stayed frozen, so the part of the network that sees the black box could not adapt.

This run: every view gets an independent black box with probability --p-occ (area drawn
from U(--occ-min, --occ-max)); the last --unfreeze CNN blocks are trained at a lower LR;
BatchNorm and Dropout stay in eval mode; views are a random prefix half the time so
1-view accuracy is kept. Training reads only the train split. Each epoch is scored on the
validation split (clean 1-view, clean 6-view, 55% occlusion 6-view with the test protocol's
seeding rule applied to validation ids). An epoch is kept only if clean 6-view validation
>= --clean-floor and clean 1-view validation does not drop more than 2 points; among
those, the best occluded validation accuracy wins. The test split is never read.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from models.backbone import novel_env  # noqa: E402
from roboeye.authenticity import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE  # noqa: E402
from roboeye.camva.data import apply_corruption, cached_resized_rgb  # noqa: E402
from roboeye.camva.notes import load_splits  # noqa: E402
from roboeye.config import DEVICE  # noqa: E402
from roboeye.qduig.config_io import load_config  # noqa: E402
from roboeye.qduig.engine import load_qduig, set_seed  # noqa: E402

NORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
JITTER = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
RESIZE = transforms.Resize((IMG_SIZE, IMG_SIZE))


def occlude(img: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    rh, rw = max(1, int(h * frac ** 0.5)), max(1, int(w * frac ** 0.5))
    y0, x0 = int(rng.integers(0, max(h - rh, 1))), int(rng.integers(0, max(w - rw, 1)))
    out = img.copy()
    out[y0:y0 + rh, x0:x0 + rw] = 0
    return out


class TrainNotes(Dataset):
    def __init__(self, ids, records, p_occ, occ_min, occ_max, prefix_prob, seed):
        self.ids, self.records = list(ids), records
        self.p_occ, self.occ_min, self.occ_max, self.prefix_prob = p_occ, occ_min, occ_max, prefix_prob
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch, i))
        rec = self.records[self.ids[i]]
        paths = list(rec["view_paths"])
        k = int(rng.integers(1, len(paths) + 1)) if rng.random() < self.prefix_prob else len(paths)
        views = []
        for p in paths[:k]:
            image = cached_resized_rgb(Path(p))
            if rng.random() < 0.5:
                image = transforms.functional.hflip(image)
            arr = np.asarray(JITTER(image))
            if rng.random() < self.p_occ:
                arr = occlude(arr, float(rng.uniform(self.occ_min, self.occ_max)), rng)
            views.append(NORM(Image.fromarray(arr)))
        return torch.stack(views), int(rec["label"])


def collate(batch):
    v = max(x[0].size(0) for x in batch)
    views = torch.zeros(len(batch), v, *batch[0][0].shape[1:])
    mask = torch.zeros(len(batch), v, dtype=torch.long)
    for i, (x, _) in enumerate(batch):
        views[i, :x.size(0)] = x
        mask[i, :x.size(0)] = 1
    return views, mask, torch.tensor([y for _, y in batch])


class ValNotes(Dataset):
    """Validation views. occ=True applies the test protocol's 55% box, seeded by note index."""

    def __init__(self, ids, records, occ):
        self.ids, self.records, self.occ = list(ids), records, occ

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        rec = self.records[self.ids[i]]
        views = []
        if self.occ:
            np.random.seed(1000 + i)
        for p in rec["view_paths"]:
            if self.occ:
                bgr = apply_corruption(cv2.imread(str(p)), "occlusion", 0.55)
                img = RESIZE(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))  # same resize as the test protocol
            else:
                img = cached_resized_rgb(Path(p))
            views.append(NORM(img))
        return torch.stack(views), int(rec["label"])


@torch.inference_mode()
def accuracy(model, loader, k=None) -> float:
    right = n = 0
    for views, mask, y in loader:
        if k is not None:
            views, mask = views[:, :k], mask[:, :k]
        p = torch.softmax(model(views.to(DEVICE), mask.to(DEVICE))["logits"], 1)[:, 1].cpu()
        right += int(((p >= 0.5).long() == y).sum())
        n += len(y)
    return right / n


def freeze_stats(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.Dropout)):
            m.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(ROOT / "results/qduig/prefix_ft/seed42"))
    ap.add_argument("--out", default=str(ROOT / "results/qduig/occlusion_robust/seed42"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--cnn-lr", type=float, default=1e-5)
    ap.add_argument("--unfreeze", type=int, default=4, help="last N MobileNet feature blocks to train (0 = keep frozen)")
    ap.add_argument("--p-occ", type=float, default=0.7)
    ap.add_argument("--occ-min", type=float, default=0.2)
    ap.add_argument("--occ-max", type=float, default=0.65)
    ap.add_argument("--prefix-prob", type=float, default=0.5)
    ap.add_argument("--clean-floor", type=float, default=0.9375)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cosine", action="store_true", help="cosine-decay both learning rates to 0 over --epochs")
    args = ap.parse_args()

    set_seed(args.seed)
    splits, records = load_splits()
    init = Path(args.init)
    cfg = load_config(init / "config.yaml")
    model = load_qduig(init / "checkpoint.pt", cfg).to(DEVICE)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cnn = model.encoder.cnn
    cnn_params = []
    if args.unfreeze > 0:
        for block in list(cnn.children())[-args.unfreeze:]:
            for p in block.parameters():
                p.requires_grad = True
                cnn_params.append(p)
    cnn_ids = {id(p) for p in cnn_params}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in cnn_ids]
    opt = torch.optim.AdamW([{"params": head_params, "lr": args.lr},
                             {"params": cnn_params, "lr": args.cnn_lr}], weight_decay=1e-4)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs) if args.cosine else None
    train_ds = TrainNotes(splits["train"], records, args.p_occ, args.occ_min, args.occ_max, args.prefix_prob, args.seed)
    kw = {"num_workers": args.workers, "persistent_workers": args.workers > 0}
    val_clean = DataLoader(ValNotes(splits["val"], records, False), batch_size=16, collate_fn=collate, **kw)
    val_occ = DataLoader(ValNotes(splits["val"], records, True), batch_size=16, collate_fn=collate, **kw)

    model.eval()
    start = {"epoch": 0, "val_clean_1": accuracy(model, val_clean, 1), "val_clean_6": accuracy(model, val_clean),
             "val_occ55_6": accuracy(model, val_occ)}
    print(json.dumps(start), flush=True)
    history = [start]
    best = None
    for ep in range(1, args.epochs + 1):
        train_ds.epoch = ep
        loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
                            generator=torch.Generator().manual_seed(args.seed + ep), num_workers=args.workers)
        model.train()
        freeze_stats(model)
        t0, loss_sum, n, skipped = time.time(), 0.0, 0, 0
        for views, mask, y in loader:
            views, mask, y = views.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(views, mask)["logits"], y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                skipped += 1
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            loss_sum += float(loss.detach()) * len(y)
            n += len(y)
        if sched is not None:
            sched.step()
        model.eval()
        row = {"epoch": ep, "train_loss": loss_sum / max(n, 1), "skipped_nonfinite_batches": skipped,
               "val_clean_1": accuracy(model, val_clean, 1), "val_clean_6": accuracy(model, val_clean),
               "val_occ55_6": accuracy(model, val_occ), "seconds": round(time.time() - t0)}
        eligible = row["val_clean_6"] >= args.clean_floor and row["val_clean_1"] >= start["val_clean_1"] - 0.02
        row["eligible"] = eligible
        history.append(row)
        print(json.dumps(row), flush=True)
        if eligible and (best is None or row["val_occ55_6"] > best["val_occ55_6"]):
            best = row
            torch.save({"model": model.state_dict(), "epoch": ep, "config": cfg, "env": novel_env(),
                        "selection": "best val_occ55_6 with val_clean_6 >= floor and val_clean_1 within 2 points",
                        "args": vars(args)}, out / "checkpoint.pt")
    (out / "config.yaml").write_text((init / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (out / "val_metrics.json").write_text(json.dumps({"args": vars(args), "start": start, "selected": best,
                                                       "history": history}, indent=2), encoding="utf-8")
    print("SELECTED", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
