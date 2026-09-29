"""View-count shift on rendered ModelNet40 point clouds.

Rules written before the test set is scored:
- Classes are the first 10 names in sorted order, excluding .vs.
- Within each class, filenames are shuffled with seed 42.
- First 64 objects train, next 16 validate, next 16 test. Objects are disjoint.
- Six views are turntable rotations of 0,60,120,180,240,300 degrees.
- fixed_n: train only on all 6 views. Checkpoint = best validation accuracy at 6 views.
- prefix: each step draws k uniformly from 1..6 and uses the first k views.
  Checkpoint = best mean validation accuracy over k=1..6.
- Test is scored once, after both checkpoints are chosen.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

WORK = Path(__file__).resolve().parents[3]
SRC = WORK / "data set for comparison" / "ModelNet40 normal_resampled" / "modelnet40_normal_resampled"
OUT = WORK / "realtime_bangla_taka_detection" / "results" / "vcds_modelnet" / "subset10_seed42_e40.json"
SEED = 42
N_CLASS = 10
N_TRAIN, N_VAL, N_TEST = 40, 10, 10
ANGLES = (0, 60, 120, 180, 240, 300)
RES = 32
EPOCHS = 40
BATCH = 32
LR = 1e-3


def render(xyz: np.ndarray) -> np.ndarray:
    lo = xyz.min(0)
    hi = xyz.max(0)
    span = np.maximum(hi - lo, 1e-6)
    pts = (xyz - lo) / span * 2 - 1
    views = np.zeros((len(ANGLES), RES, RES), dtype=np.float32)
    yy, xx = pts[:, 1], pts[:, 0]
    zz = pts[:, 2]
    for i, deg in enumerate(ANGLES):
        th = np.deg2rad(deg)
        c, s = np.cos(th), np.sin(th)
        xr = c * xx + s * zz
        zr = -s * xx + c * zz
        # camera looks along z; image plane is x (horizontal) and y (vertical)
        u = np.clip(((xr + 1) * 0.5 * (RES - 1)).astype(np.int32), 0, RES - 1)
        v = np.clip(((yy + 1) * 0.5 * (RES - 1)).astype(np.int32), 0, RES - 1)
        depth = zr
        img = np.full((RES, RES), -2.0, dtype=np.float32)
        # painter: keep the nearest point (largest z toward a camera at +z after rotation)
        order = np.argsort(depth)
        img[v[order], u[order]] = depth[order]
        m = img > -1.5
        if m.any():
            vis = img[m]
            img[m] = (vis - vis.min()) / max(float(vis.max() - vis.min()), 1e-6)
            img[~m] = 0
        else:
            img[:] = 0
        views[i] = img
    return views


def load_xyz(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return raw[:, :3]


class Net(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        b, k, h, w = views.shape
        z = self.conv(views.reshape(b * k, 1, h, w)).reshape(b, k, 64).mean(1)
        return self.fc(z)


def accuracy(model, views, labels, k, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(labels), BATCH):
            x = torch.from_numpy(views[i:i + BATCH, :k]).to(device)
            y = torch.from_numpy(labels[i:i + BATCH]).to(device)
            pred = model(x).argmax(1)
            correct += int((pred == y).sum())
    return correct / len(labels)


def train_one(mode: str, views_tr, y_tr, views_va, y_va, device):
    torch.manual_seed(SEED + (1 if mode == "prefix" else 2))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + (1 if mode == "prefix" else 2))
    model = Net(N_CLASS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    rng = random.Random(SEED + (1 if mode == "prefix" else 2))
    best = -1.0
    best_state = None
    history = []
    n = len(y_tr)
    for epoch in range(EPOCHS):
        model.train()
        order = list(range(n))
        rng.shuffle(order)
        total = 0.0
        seen = 0
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            k = 6 if mode == "fixed" else rng.randint(1, 6)
            x = torch.from_numpy(views_tr[idx, :k]).to(device)
            y = torch.from_numpy(y_tr[idx]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
            seen += len(idx)
        if mode == "fixed":
            score = accuracy(model, views_va, y_va, 6, device)
            key = "val_k6"
        else:
            vals = [accuracy(model, views_va, y_va, k, device) for k in range(1, 7)]
            score = float(np.mean(vals))
            key = "val_mean_k"
        row = {"epoch": epoch + 1, "train_loss": total / seen, key: score}
        if mode == "prefix":
            row["val_by_k"] = vals
        history.append(row)
        print(mode, row, flush=True)
        if score > best + 1e-6:
            best = score
            best_state = {t: v.detach().cpu().clone() for t, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history, best


def main():
    classes = sorted(p.name for p in SRC.iterdir() if p.is_dir() and p.name != ".vs")[:N_CLASS]
    rng = random.Random(SEED)
    splits = {"train": [], "val": [], "test": []}
    labels = {"train": [], "val": [], "test": []}
    for ci, name in enumerate(classes):
        files = sorted((SRC / name).glob("*.txt"))
        rng.shuffle(files)
        need = N_TRAIN + N_VAL + N_TEST
        if len(files) < need:
            raise SystemExit(f"{name} has {len(files)} clouds, need {need}")
        parts = {
            "train": files[:N_TRAIN],
            "val": files[N_TRAIN:N_TRAIN + N_VAL],
            "test": files[N_TRAIN + N_VAL:need],
        }
        for split, group in parts.items():
            for path in group:
                splits[split].append(str(path))
                labels[split].append(ci)
    print("classes", classes, flush=True)
    rendered = {}
    ys = {}
    for split in ("train", "val", "test"):
        acc_v = []
        for i, path in enumerate(splits[split]):
            xyz = load_xyz(Path(path))
            acc_v.append(render(xyz))
            if (i + 1) % 100 == 0:
                print(split, i + 1, flush=True)
        rendered[split] = np.stack(acc_v)
        ys[split] = np.asarray(labels[split], dtype=np.int64)
        print(split, rendered[split].shape, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    fixed, hist_f, best_f = train_one("fixed", rendered["train"], ys["train"], rendered["val"], ys["val"], device)
    prefix, hist_p, best_p = train_one("prefix", rendered["train"], ys["train"], rendered["val"], ys["val"], device)
    report = {
        "source": str(SRC),
        "note": "Orthographic turntable renders of ModelNet40 normal_resampled point clouds. Not photographs. 10-class subset.",
        "classes": classes,
        "seed": SEED,
        "n_train": N_TRAIN * N_CLASS,
        "n_val": N_VAL * N_CLASS,
        "n_test": N_TEST * N_CLASS,
        "angles_deg": list(ANGLES),
        "resolution": RES,
        "epochs": EPOCHS,
        "init_seeded": True,
        "device": str(device),
        "selection": {
            "fixed": "max validation accuracy at 6 views",
            "prefix": "max mean validation accuracy over k=1..6",
        },
        "val_best": {"fixed_k6": best_f, "prefix_mean": best_p},
        "history_fixed": hist_f,
        "history_prefix": hist_p,
        "test_fixed": {f"k{k}": accuracy(fixed, rendered["test"], ys["test"], k, device) for k in range(1, 7)},
        "test_prefix": {f"k{k}": accuracy(prefix, rendered["test"], ys["test"], k, device) for k in range(1, 7)},
    }
    # prefix oracle: best k per object, labels known, analysis only
    prefix.eval()
    correct_any = 0
    with torch.no_grad():
        x = torch.from_numpy(rendered["test"]).to(device)
        logits = []
        for k in range(1, 7):
            logits.append(prefix(x[:, :k]).argmax(1).cpu().numpy())
        y = ys["test"]
        for i in range(len(y)):
            if any(int(logits[k][i]) == int(y[i]) for k in range(6)):
                correct_any += 1
    report["prefix_oracle_any_k"] = correct_any / len(ys["test"])
    report["prefix_oracle_note"] = "analysis only; uses labels to pick k; not a deployable policy"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("WROTE", OUT, flush=True)
    print("TEST_FIXED", report["test_fixed"], flush=True)
    print("TEST_PREFIX", report["test_prefix"], flush=True)


if __name__ == "__main__":
    main()
