"""Train a small masked reconstructor on training notes only.

The hole is a random box. The checkpoint is the lowest validation hole-L1.
The test split is not read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.engine import set_seed

OUT = ROOT / "results" / "robustness" / "occlusion_inpaint" / "seed42"
SIZE = 128
EPOCHS = 2
BATCH = 8
LR = 1e-3


class Reconstructor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        return self.dec1(torch.cat([d1, e1], dim=1))


def _sample(bgr: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    frac = float(rng.uniform(0.35, 0.55))
    side = max(1, int(SIZE * frac ** 0.5))
    y0 = int(rng.integers(0, SIZE - side + 1))
    x0 = int(rng.integers(0, SIZE - side + 1))
    mask = np.zeros((SIZE, SIZE), np.float32)
    mask[y0:y0 + side, x0:x0 + side] = 1.0
    hole = rgb.copy()
    hole[mask > 0] = 0.0
    return hole, mask, rgb


def _batch(records, pairs, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    holes, masks, targets = [], [], []
    for nid, view_i in pairs:
        path = records[nid]["view_paths"][view_i]
        bgr = cv2.imread(str(path))
        hole, mask, rgb = _sample(bgr, rng)
        holes.append(hole)
        masks.append(mask)
        targets.append(rgb)
    hole_t = torch.from_numpy(np.stack(holes)).permute(0, 3, 1, 2)
    mask_t = torch.from_numpy(np.stack(masks))[:, None]
    target_t = torch.from_numpy(np.stack(targets)).permute(0, 3, 1, 2)
    return torch.cat([hole_t, mask_t], dim=1), mask_t, target_t


def _hole_l1(model, records, ids, seed: int) -> float:
    rng = np.random.default_rng(seed)
    pairs = [(nid, 0) for nid in ids]
    total, count = 0.0, 0.0
    model.eval()
    for start in range(0, len(pairs), BATCH):
        x, mask, target = _batch(records, pairs[start:start + BATCH], rng)
        x, mask, target = x.to(DEVICE), mask.to(DEVICE), target.to(DEVICE)
        with torch.inference_mode():
            pred = model(x)
        err = ((pred - target).abs() * mask).sum()
        total += float(err)
        count += float(mask.sum())
    return total / max(count, 1.0)


def main() -> None:
    set_seed(42)
    splits, records = load_splits()
    train_ids = list(splits["train"])
    val_ids = list(splits["val"])
    OUT.mkdir(parents=True, exist_ok=True)
    model = Reconstructor().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(42)
    pairs = [(nid, view_i) for nid in train_ids for view_i in range(6)]
    best = float("inf")
    history = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(pairs))
        running, seen = 0.0, 0
        for start in range(0, len(order), BATCH):
            chunk = [pairs[int(i)] for i in order[start:start + BATCH]]
            x, mask, target = _batch(records, chunk, rng)
            x, mask, target = x.to(DEVICE), mask.to(DEVICE), target.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = ((pred - target).abs() * mask).sum() / mask.sum().clamp_min(1.0)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.detach()) * len(chunk)
            seen += len(chunk)
            if seen % 200 == 0:
                print(json.dumps({"epoch": ep, "seen": seen, "loss": running / seen}), flush=True)
        val = _hole_l1(model, records, val_ids, 1000 + ep)
        row = {"epoch": ep, "train_hole_l1": running / max(seen, 1), "val_hole_l1": val}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val < best:
            best = val
            torch.save({"model": model.state_dict(), "epoch": ep, "val_hole_l1": val}, OUT / "checkpoint.pt")
    (OUT / "val_metrics.json").write_text(json.dumps({"best_val_hole_l1": best, "history": history}, indent=2), encoding="utf-8")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
