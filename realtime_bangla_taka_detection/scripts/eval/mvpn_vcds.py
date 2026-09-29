"""View-count distribution shift on MVP-N (NeurIPS 2022), a real multi-view dataset.

Data: data set for comparison/MVP-N-main/data (44 classes; train images grouped by object set;
valid/test sets of 2-6 real views from the official *_list/*_100.pkl files, 4,400 sets each).
Features: frozen ImageNet ResNet-50 (2048-d, L2-normalised), computed once and cached.
Heads (trained on the features, 3 seeds):
  concat  : 6 view slots, zero-padded, concatenated -> MLP (MVCNN-style joint fusion; count-sensitive)
  attn    : per-view MLP -> attention pooling over present views -> classifier
Regimes: fixed = every training sample has 6 views; prefix = random k in 1..6 views.
Training samples: 6 views drawn from one train object set when it has 6, else from the class's train
images. Epoch chosen on VALID (mean accuracy over k = 1..6 prefixes). TEST read once.
Test at k: the first k views of every test set that has at least k views.
Output: results/mvpn/vcds.json
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "data set for comparison" / "MVP-N-main" / "data"
OUT = ROOT / "results" / "mvpn"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = sorted(os.listdir(DATA / "train"))
K = 6


@torch.inference_mode()
def features() -> dict[str, np.ndarray]:
    cache = OUT / "feats_resnet50.npz"
    if cache.is_file():
        d = np.load(cache, allow_pickle=True)
        return dict(zip(d["keys"], d["x"]))
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Identity()
    m = m.eval().to(DEV)
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    paths = sorted(glob.glob(str(DATA / "train" / "*" / "*" / "*.png")))
    for split in ("valid", "test"):
        paths += sorted(glob.glob(str(DATA / split / "*" / "*" / "*.png")))
    keys, xs = [], []
    for i in range(0, len(paths), 64):
        batch = paths[i:i + 64]
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch]).to(DEV)
        z = m(x).float().cpu().numpy()
        xs.append(z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8))
        keys += [str(Path(p).relative_to(DATA)).replace("\\", "/") for p in batch]
        if (i // 64) % 40 == 0:
            print("features", i, "/", len(paths), flush=True)
    x = np.concatenate(xs).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(cache, keys=np.array(keys), x=x)
    return dict(zip(keys, x))


def eval_sets(split: str) -> list[tuple[list[str], int]]:
    out = []
    for ci, c in enumerate(CLASSES):
        for s in pickle.load(open(DATA / f"{split}_list" / f"{split}_{c}_100.pkl", "rb")):
            out.append(([f"{split}/{p}" for p in s], ci))
    return out


class Concat(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(K * d, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n))

    def forward(self, x, m):  # x (B,K,d), m (B,K)
        return self.net((x * m[..., None]).flatten(1))


class Attn(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 512), nn.ReLU())
        self.att = nn.Linear(512, 1)
        self.cls = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, n))

    def forward(self, x, m):
        h = self.enc(x)
        a = self.att(h).squeeze(-1).masked_fill(m == 0, -1e9)
        return self.cls((torch.softmax(a, 1)[..., None] * h).sum(1))


def batch_tensor(sets, feats, k_of):
    x = np.zeros((len(sets), K, 2048), np.float32)
    m = np.zeros((len(sets), K), np.float32)
    for i, v in enumerate(sets):
        k = k_of(len(v))
        for j in range(k):
            x[i, j] = feats[v[j]]
        m[i, :k] = 1
    return torch.from_numpy(x).to(DEV), torch.from_numpy(m).to(DEV)


@torch.inference_mode()
def acc_by_k(model, sets, feats) -> dict[int, float]:
    model.eval()
    res = {}
    for k in range(1, K + 1):
        sub = [(v, y) for v, y in sets if len(v) >= k]
        x, m = batch_tensor([v for v, _ in sub], feats, lambda n: k)
        y = torch.tensor([y for _, y in sub], device=DEV)
        res[k] = float((model(x, m).argmax(1) == y).float().mean())
    return res


def train_samples(rng: random.Random, n: int) -> list[tuple[list[str], int]]:
    by_class = {}
    for ci, c in enumerate(CLASSES):
        imgs = sorted(glob.glob(str(DATA / "train" / c / "*" / "*.png")))
        by_class[ci] = [str(Path(p).relative_to(DATA)).replace("\\", "/") for p in imgs]
    out = []
    for _ in range(n):
        ci = rng.randrange(len(CLASSES))
        out.append((rng.sample(by_class[ci], K), ci))
    return out


def run(head: str, regime: str, seed: int, feats, val, test) -> dict:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = (Concat if head == "concat" else Attn)(2048, len(CLASSES)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, hist = -1.0, None, []
    for ep in range(15):
        model.train()
        samples = train_samples(rng, 4000)
        for i in range(0, len(samples), 64):
            b = samples[i:i + 64]
            k_of = (lambda n: K) if regime == "fixed" else (lambda n: rng.randint(1, K))
            x, m = batch_tensor([v for v, _ in b], feats, k_of)
            y = torch.tensor([y for _, y in b], device=DEV)
            loss = F.cross_entropy(model(x, m), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        va = acc_by_k(model, val, feats)
        score = float(np.mean(list(va.values())))
        hist.append({"epoch": ep, "val_mean": score})
        if score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {"head": head, "regime": regime, "seed": seed, "val_mean_best": best,
            "val": acc_by_k(model, val, feats), "test": acc_by_k(model, test, feats), "history": hist}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feats = features()
    val, test = eval_sets("valid"), eval_sets("test")
    rows = []
    for head in ("concat", "attn"):
        for regime in ("fixed", "prefix"):
            for seed in (42, 43, 44):
                r = run(head, regime, seed, feats, val, test)
                rows.append(r)
                print(head, regime, seed, {k: round(v, 3) for k, v in r["test"].items()}, flush=True)
    summ = {}
    for head in ("concat", "attn"):
        for regime in ("fixed", "prefix"):
            rs = [r for r in rows if r["head"] == head and r["regime"] == regime]
            summ[f"{head}/{regime}"] = {str(k): {"mean": float(np.mean([r["test"][k] for r in rs])),
                                                 "sd": float(np.std([r["test"][k] for r in rs], ddof=1))} for k in range(1, K + 1)}
    blob = {"n_test_sets": len(test), "n_val_sets": len(val), "classes": len(CLASSES), "summary": summ, "runs": rows}
    (OUT / "vcds.json").write_text(json.dumps(blob, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
