"""Fusion-general fix on MVP-N: why prefix training failed for the concatenation head, and five repairs.

Same data, features, protocol and seeds as mvpn_vcds.py (VALID chooses the epoch; TEST read once).
Heads:
  concat_mask   concatenation of zero-padded slots + the 6-d view mask
  concat_count  concatenation + one-hot view count
  concat_scaled concatenation with present slots rescaled by 6 / k
  concat_mix    plain concatenation, prefix regime drawing k = 6 half the time (curriculum-like mix)
  meanpool      per-view MLP, mean over present views, classifier (count-invariant by construction)
Each head is trained fixed (k = 6) and prefix (k uniform in 1..6, except concat_mix).
Output: results/mvpn/fusion_fix.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import mvpn_vcds as mv  # noqa: E402

K, D = mv.K, 2048


class ConcatVar(nn.Module):
    def __init__(self, n, mode):
        super().__init__()
        self.mode = mode
        extra = K if mode in ("mask", "count") else 0
        self.net = nn.Sequential(nn.Linear(K * D + extra, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n))

    def forward(self, x, m):
        k = m.sum(1, keepdim=True).clamp(min=1)
        z = x * m[..., None]
        if self.mode == "scaled":
            z = z * (K / k)[..., None]
        z = z.flatten(1)
        if self.mode == "mask":
            z = torch.cat([z, m], 1)
        if self.mode == "count":
            z = torch.cat([z, F.one_hot((k.squeeze(1) - 1).long(), K).float()], 1)
        return self.net(z)


class MeanPool(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 512), nn.ReLU())
        self.cls = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, n))

    def forward(self, x, m):
        h = self.enc(x) * m[..., None]
        return self.cls(h.sum(1) / m.sum(1, keepdim=True).clamp(min=1))


def run(make, regime, seed, feats, val, test):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = make().to(mv.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state = -1.0, None
    for ep in range(15):
        model.train()
        samples = mv.train_samples(rng, 4000)
        for i in range(0, len(samples), 64):
            b = samples[i:i + 64]
            if regime == "fixed":
                k_of = lambda n: K
            elif regime == "mix":
                k_of = lambda n: K if rng.random() < 0.5 else rng.randint(1, K)
            else:
                k_of = lambda n: rng.randint(1, K)
            x, m = mv.batch_tensor([v for v, _ in b], feats, k_of)
            y = torch.tensor([y for _, y in b], device=mv.DEV)
            loss = F.cross_entropy(model(x, m), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        s = float(np.mean(list(mv.acc_by_k(model, val, feats).values())))
        if s > best:
            best, state = s, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return mv.acc_by_k(model, test, feats)


def main() -> None:
    d = np.load(mv.OUT / "feats_resnet50.npz", allow_pickle=True)
    feats = dict(zip(d["keys"], d["x"]))
    val, test = mv.eval_sets("valid"), mv.eval_sets("test")
    n = len(mv.CLASSES)
    heads = {"concat_mask": (lambda: ConcatVar(n, "mask"), ("fixed", "prefix")),
             "concat_count": (lambda: ConcatVar(n, "count"), ("fixed", "prefix")),
             "concat_scaled": (lambda: ConcatVar(n, "scaled"), ("fixed", "prefix")),
             "concat_mix": (lambda: ConcatVar(n, "plain"), ("mix",)),
             "meanpool": (lambda: MeanPool(n), ("fixed", "prefix"))}
    out = {}
    for name, (make, regimes) in heads.items():
        for regime in regimes:
            runs = [run(make, regime, s, feats, val, test) for s in (42, 43, 44)]
            out[f"{name}/{regime}"] = {str(k): {"mean": float(np.mean([r[k] for r in runs])), "sd": float(np.std([r[k] for r in runs], ddof=1))}
                                       for k in range(1, K + 1)}
            o = out[f"{name}/{regime}"]
            print(f"{name}/{regime}: k1 {o['1']['mean']*100:.1f}  k6 {o['6']['mean']*100:.1f}", flush=True)
    (mv.OUT / "fusion_fix.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
