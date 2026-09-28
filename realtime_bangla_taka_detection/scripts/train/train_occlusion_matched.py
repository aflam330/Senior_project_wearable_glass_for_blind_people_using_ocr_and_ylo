"""Fine-tune on per-view black-box occlusion followed by median fill.

Batch-norm and dropout stay in eval mode so their running statistics are not
replaced by the occluded batches. An epoch is kept only when clean 6-view
validation accuracy does not fall below the starting checkpoint and occluded
validation accuracy improves. The test split is not read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.authenticity import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from roboeye.camva.data import apply_corruption
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig, set_seed

_RESIZE = transforms.Resize((IMG_SIZE, IMG_SIZE))
_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
EPOCHS = 1
LR = 1e-5
BATCH = 4
MAX_STEPS = 40


def _median_fill(bgr: np.ndarray) -> np.ndarray:
    hole = bgr.sum(axis=2) == 0
    if not np.any(hole):
        return bgr
    fill = np.median(bgr[~hole], axis=0)
    out = bgr.copy()
    out[hole] = fill
    return out


def _occlude_fill(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    frac = float(rng.uniform(0.35, 0.55))
    h, w = bgr.shape[:2]
    rh = max(1, int(h * frac ** 0.5))
    rw = max(1, int(w * frac ** 0.5))
    y0 = int(rng.integers(0, max(h - rh, 1)))
    x0 = int(rng.integers(0, max(w - rw, 1)))
    out = bgr.copy()
    out[y0:y0 + rh, x0:x0 + rw] = 0
    return _median_fill(out)


def _tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _TENSOR(_RESIZE(Image.fromarray(rgb)))


def _load_views(records, nid: str, rng: np.random.Generator | None, test_occlusion: bool) -> torch.Tensor:
    frames = []
    for path in records[nid]["view_paths"]:
        bgr = cv2.imread(str(path))
        if test_occlusion:
            bgr = _median_fill(apply_corruption(bgr, "occlusion", 0.55))
        elif rng is not None:
            bgr = _occlude_fill(bgr, rng)
        frames.append(_tensor(bgr))
    return torch.stack(frames, dim=0)


def _accuracy(model, records, ids, labels, seed: int, occluded: bool) -> float:
    correct = 0
    for start in range(0, len(ids), BATCH):
        chunk = ids[start:start + BATCH]
        batch = []
        for i, nid in enumerate(chunk):
            if occluded:
                np.random.seed(1000 + start + i)
                batch.append(_load_views(records, nid, None, True))
            else:
                batch.append(_load_views(records, nid, None, False))
        views = torch.stack(batch, dim=0).to(DEVICE)
        mask = torch.ones(views.size(0), views.size(1), dtype=torch.long, device=DEVICE)
        with torch.inference_mode():
            prob = torch.softmax(model(views, mask)["logits"], dim=1)[:, 1]
        pred = (prob >= 0.5).detach().cpu().numpy()
        correct += int(np.sum(pred == labels[start:start + len(pred)]))
    return correct / len(ids)


def main() -> None:
    set_seed(42)
    splits, records = load_splits()
    cfg = load_config(ROOT / "configs" / "occlusion_ft.yaml")
    resume = ROOT / "results" / "qduig" / "occlusion_ft" / "seed42" / "checkpoint.pt"
    out = ROOT / "results" / "qduig" / "occlusion_matched" / "seed42"
    out.mkdir(parents=True, exist_ok=True)
    model = load_qduig(resume, cfg).to(DEVICE)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=LR)
    train_ids = list(splits["train"])
    val_ids = list(splits["val"])
    val_y = np.array([int(records[nid]["label"]) for nid in val_ids])
    rng = np.random.default_rng(42)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.eval()
    clean0 = _accuracy(model, records, val_ids, val_y, 42, False)
    occ0 = _accuracy(model, records, val_ids, val_y, 42, True)
    print(json.dumps({"epoch": 0, "val_clean_6": clean0, "val_occ_median_6": occ0}), flush=True)
    if clean0 + 1e-12 < 0.90:
        print("initial clean validation below 0.90, loader does not match the saved model", flush=True)
        raise SystemExit(3)
    floor = clean0
    best_occ = occ0
    history = [{"epoch": 0, "val_clean_6": clean0, "val_occ_median_6": occ0, "clean_floor": floor}]
    for ep in range(1, EPOCHS + 1):
        model.train()
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.Dropout)):
                module.eval()
        order = rng.permutation(len(train_ids))
        running, n = 0.0, 0
        for start in range(0, len(order), BATCH):
            chunk = [train_ids[int(i)] for i in order[start:start + BATCH]]
            views = torch.stack([
                _load_views(records, nid, np.random.default_rng(rng.integers(0, 2**31 - 1)), False)
                for nid in chunk
            ], dim=0).to(DEVICE)
            y = torch.tensor([int(records[nid]["label"]) for nid in chunk], device=DEVICE)
            mask = torch.ones(views.size(0), views.size(1), dtype=torch.long, device=DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(views, mask)["logits"]
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            finite_grad = all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
            if not finite_grad:
                opt.zero_grad(set_to_none=True)
                print(json.dumps({"epoch": ep, "seen": n + len(chunk), "skip": "nonfinite_grad"}), flush=True)
                n += len(chunk)
                if n >= MAX_STEPS:
                    break
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * len(chunk)
            n += len(chunk)
            print(json.dumps({"epoch": ep, "seen": n, "loss": running / n}), flush=True)
            if n >= MAX_STEPS:
                break
        model.eval()
        clean = _accuracy(model, records, val_ids, val_y, 42, False)
        occ = _accuracy(model, records, val_ids, val_y, 42, True)
        row = {"epoch": ep, "train_loss": running / max(n, 1), "val_clean_6": clean, "val_occ_median_6": occ, "clean_floor": floor}
        history.append(row)
        print(json.dumps(row), flush=True)
        if clean + 1e-12 < floor:
            print("clean validation fell below the starting checkpoint, stop and keep the previous weights", flush=True)
            break
        if occ > best_occ:
            best_occ = occ
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save({"model": model.state_dict(), "epoch": "selected_on_val", "clean_floor": floor, "history": history}, out / "checkpoint.pt")
    (out / "val_metrics.json").write_text(json.dumps({"clean_floor": floor, "floor_rule": "do not fall below the starting clean 6-view validation accuracy", "best_val_occ_median_6": best_occ, "history": history}, indent=2), encoding="utf-8")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
