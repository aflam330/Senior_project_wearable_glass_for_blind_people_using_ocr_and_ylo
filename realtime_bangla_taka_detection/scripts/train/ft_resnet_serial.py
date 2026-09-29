"""Fine-tuned ResNet-50 baseline on the SERIAL-DISJOINT split (stronger than the frozen probe).

ImageNet ResNet-50; layer3, layer4 and a new 2-way head are trained (earlier layers frozen, BN in eval
mode), on single views of TRAIN notes (all 6 views per note, 224 px, flip + colour jitter). k-view score =
mean genuine probability over the first k views. Epoch (of 5) chosen on VAL mean accuracy over k = 1..6.
TEST read once. Seeds 42, 43, 44. Per-note TEST probabilities saved for paired tests.
Output: results/serial_split/ft_resnet50/seed<s>.json
"""
from __future__ import annotations

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
    seed = int(sys.argv[1])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    splits, records = load_splits(ROOT / "results" / "serial_split")
    ids = {s: list(splits[s]) for s in ("train", "val", "test")}
    y = {s: np.array([int(records[n]["label"]) for n in ids[s]]) for s in ids}
    items = [(cpath(n, k), int(records[n]["label"])) for n in ids["train"] for k in range(6)]
    dl = DataLoader(Views(items, TR), batch_size=24, shuffle=True, num_workers=0)
    m = build()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    best, hist = (-1.0, None), []
    for ep in range(1, 6):
        m.train(); freeze_bn(m)
        for x, t in dl:
            loss = nn.functional.cross_entropy(m(x.to(DEV)), t.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        va = acc_k(note_probs(m, ids["val"], records), y["val"])
        score = float(np.mean(list(va.values())))
        hist.append({"epoch": ep, "val_mean": score, "val": va})
        print(seed, hist[-1]["epoch"], round(score, 4), flush=True)
        if score > best[0]:
            best = (score, {k: v.detach().clone() for k, v in m.state_dict().items()})
    m.load_state_dict(best[1])
    Pt = note_probs(m, ids["test"], records)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"seed": seed, "history": hist, "best_val_mean": best[0], "test": acc_k(Pt, y["test"]),
           "test_note_ids": ids["test"], "test_probs": Pt.round(6).tolist()}
    (OUT / f"seed{seed}.json").write_text(json.dumps(res), encoding="utf-8")
    print("TEST", seed, {k: round(v, 4) for k, v in res["test"].items()}, flush=True)


if __name__ == "__main__":
    main()
