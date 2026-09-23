"""Train/eval handwritten Bangla letters on Ekush (writer-disjoint).

Does not use MatriVasha unless Ekush fails. Not used for currency.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ekush_letters import EkushCNN, labels_from_csv, preprocess_glyph  # noqa: E402

DATA = Path(r"E:\Final SP\data set\OCR bangla data set\Ekush Data set")
IMG_ROOT = DATA / "dataset"
CSV_PATH = DATA / "metaData_img.csv"
OUT_WEIGHTS = ROOT / "models" / "ekush_cnn.pt"
OUT_LABELS = ROOT / "assets" / "ekush_labels.json"
OUT_JSON = ROOT / "results" / "ekush_letters.json"
SEED = 42
BATCH = 256
EPOCHS = 8
LR = 1e-3
NUM_WORKERS = 0


def writer_id(path: Path) -> str:
    parts = path.stem.split("_")
    return parts[-1] if parts else path.stem


class GlyphSet(Dataset):
    def __init__(self, items: list[tuple[Path, int]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, y = self.items[i]
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            im = np.zeros((28, 28), np.uint8)
        x = preprocess_glyph(im)
        return torch.from_numpy(x).unsqueeze(0), y


def collect() -> tuple[dict[int, str], dict[str, list[tuple[Path, int]]]]:
    id_to_char = labels_from_csv(CSV_PATH)
    by_writer: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    n = 0
    for folder in IMG_ROOT.iterdir():
        if not folder.is_dir():
            continue
        try:
            cid = int(folder.name)
        except ValueError:
            continue
        if cid not in id_to_char:
            continue
        with os.scandir(folder) as it:
            for ent in it:
                if not ent.is_file():
                    continue
                p = Path(ent.path)
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                by_writer[writer_id(p)].append((p, cid))
                n += 1
    print(f"images={n} writers={len(by_writer)} classes={len(id_to_char)}")
    return id_to_char, by_writer


def split_writers(by_writer, seed=SEED):
    rng = random.Random(seed)
    writers = sorted(by_writer)
    rng.shuffle(writers)
    n = len(writers)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    tr_w = set(writers[:n_train])
    va_w = set(writers[n_train:n_train + n_val])
    te_w = set(writers[n_train + n_val:])
    def take(ws):
        out = []
        for w in ws:
            out.extend(by_writer[w])
        return out
    return take(tr_w), take(va_w), take(te_w), {
        "n_writers": n,
        "train_writers": len(tr_w),
        "val_writers": len(va_w),
        "test_writers": len(te_w),
        "seed": seed,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    id_to_char, by_writer = collect()
    OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.write_text(
        json.dumps({"id_to_char": {str(k): v for k, v in sorted(id_to_char.items())}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    train_i, val_i, test_i, split_meta = split_writers(by_writer)
    print("split", split_meta, "n_train", len(train_i), "n_val", len(val_i), "n_test", len(test_i))

    train_loader = DataLoader(GlyphSet(train_i), batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(GlyphSet(val_i), batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(GlyphSet(test_i), batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

    model = EkushCNN(len(id_to_char)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val = -1.0
    history = []
    t0 = time.time()
    OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        n_b = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.item())
            n_b += 1
        sched.step()
        val_acc = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "loss": loss_sum / max(n_b, 1), "val_acc": val_acc}
        history.append(row)
        print(f"epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "val_acc": val_acc, "epoch": epoch}, OUT_WEIGHTS)

    ckpt = torch.load(OUT_WEIGHTS, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_acc = evaluate(model, test_loader, device)
    # group acc on test
    groups = {
        "matra_0_9": range(0, 10),
        "vowels_10_20": range(10, 21),
        "consonants_21_59": range(21, 60),
        "conjuncts_60_111": range(60, 112),
        "digits_112_121": range(112, 122),
    }
    model.eval()
    hits = defaultdict(int)
    tot = defaultdict(int)
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).argmax(1).cpu()
            y = y.cpu()
            for p, t in zip(pred.tolist(), y.tolist()):
                tot["all"] += 1
                hits["all"] += int(p == t)
                for g, rng in groups.items():
                    if t in rng:
                        tot[g] += 1
                        hits[g] += int(p == t)
    group_acc = {g: hits[g] / max(tot[g], 1) for g in tot}
    out = {
        "dataset": "Ekush",
        "n_classes": len(id_to_char),
        "split": split_meta,
        "n_train": len(train_i),
        "n_val": len(val_i),
        "n_test": len(test_i),
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "group_acc": group_acc,
        "group_n": dict(tot),
        "history": history,
        "weights": str(OUT_WEIGHTS),
        "seconds": time.time() - t0,
        "device": str(device),
        "note": "Writer-disjoint 80/10/10 by filename last token. Isolated glyphs, not word OCR.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("TEST_ACC", test_acc)
    print("GROUPS", group_acc)
    print("wrote", OUT_JSON, OUT_WEIGHTS)


if __name__ == "__main__":
    main()
