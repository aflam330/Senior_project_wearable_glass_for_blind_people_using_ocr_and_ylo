"""Fine-tuned ResNet-50 baseline on the SERIAL-DISJOINT split (stronger than the frozen probe).

ImageNet ResNet-50; layer3, layer4 and a new 2-way head are trained (earlier layers frozen, BN in eval
mode), on single views of TRAIN notes (all 6 views per note, 224 px, flip + colour jitter). k-view score =
mean genuine probability over the first k views. The epoch is the one with the best VAL mean accuracy
over k = 1..6. TEST read once. Seeds 42, 43, 44.

Default cache and output are the published run (quarter-decoded views, 5 epochs):
  results/serial_split/ft_resnet50/seed<s>.json

Full-resolution cache (do not overwrite the published files):
  python scripts/train/ft_resnet_serial.py 42 --cache cache/views256_full --out results/serial_split/ft_resnet50_fullres --epochs 8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from roboeye.camva.notes import load_splits  # noqa: E402

OUT = ROOT / "results" / "serial_split" / "ft_resnet50"
DEV = torch.device("cuda")
NORM = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
TR = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2, 0.03),
                         transforms.ToTensor(), NORM])
EV = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])


CACHE = ROOT / "cache" / "views256"  # built by scripts/train/cache_views_256.py


def cpath(nid, k):
    return str(CACHE / f"{nid.replace(':', '_')}_{k}.jpg")


def load(p):
    return Image.open(p).convert("RGB")


class Views(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(load(p)), y


def build():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(2048, 2)
    for name, p in m.named_parameters():
        p.requires_grad = name.startswith(("layer3", "layer4", "fc"))
    return m.to(DEV)


def freeze_bn(m):
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()


@torch.inference_mode()
def note_probs(m, ids, records):
    m.eval()
    out = np.zeros((len(ids), 6))
    for i, n in enumerate(ids):
        x = torch.stack([EV(load(cpath(n, k))) for k in range(6)]).to(DEV)
        out[i] = torch.softmax(m(x), 1)[:, 1].cpu().numpy()
    return out


def acc_k(P, y):
    return {k: float(((P[:, :k].mean(1) >= 0.5) == (y == 1)).mean()) for k in range(1, 7)}


def main() -> None:
    global CACHE, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()
    CACHE = args.cache if args.cache.is_absolute() else ROOT / args.cache
    OUT = args.out if args.out.is_absolute() else ROOT / args.out
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    splits, records = load_splits(ROOT / "results" / "serial_split")
    ids = {s: list(splits[s]) for s in ("train", "val", "test")}
    y = {s: np.array([int(records[n]["label"]) for n in ids[s]]) for s in ids}
    items = [(cpath(n, k), int(records[n]["label"])) for n in ids["train"] for k in range(6)]
    missing = [p for p, _ in items[:6] if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"missing cached views, build the cache first: {missing[0]}")
    dl = DataLoader(Views(items, TR), batch_size=24, shuffle=True, num_workers=0, pin_memory=True)
    m = build()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    best, hist = (-1.0, None), []
    for ep in range(1, args.epochs + 1):
        m.train(); freeze_bn(m)
        for x, t in dl:
            loss = nn.functional.cross_entropy(m(x.to(DEV)), t.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        va = acc_k(note_probs(m, ids["val"], records), y["val"])
        score = float(np.mean(list(va.values())))
        hist.append({"epoch": ep, "val_mean": score, "val": va})
        print(seed, hist[-1]["epoch"], round(score, 4), flush=True)
        if score > best[0]:
            best = (score, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()})
    m.load_state_dict(best[1])
    Pv = note_probs(m, ids["val"], records)
    Pt = note_probs(m, ids["test"], records)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"seed": seed, "best_val_mean": best[0], "state_dict": best[1]}, OUT / f"seed{seed}.pt")
    res = {"seed": seed, "cache": str(CACHE), "epochs": args.epochs, "history": hist, "best_val_mean": best[0],
           "val": acc_k(Pv, y["val"]), "test": acc_k(Pt, y["test"]),
           "val_note_ids": ids["val"], "val_probs": Pv.round(6).tolist(),
           "test_note_ids": ids["test"], "test_probs": Pt.round(6).tolist()}
    (OUT / f"seed{seed}.json").write_text(json.dumps(res), encoding="utf-8")
    print("TEST", seed, {k: round(v, 4) for k, v in res["test"].items()}, flush=True)


if __name__ == "__main__":
    main()
