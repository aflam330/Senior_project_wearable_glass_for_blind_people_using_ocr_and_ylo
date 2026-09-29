"""When does view-count shift hurt? Feature-shift diagnostic on two datasets, three seeds.

For a model trained only on N = 6 views that averages per-view features, the input to its
classifier at k views is the mean of the first k view embeddings. Define the shift

    s_k = mean over objects ||e_k - e_6||  /  mean distance between class centroids of e_6

(e = the vector the classifier head receives). Hypothesis: the accuracy drop Acc(6) - Acc(k)
grows with s_k, and s_k is large when views show different content (JaalTaka: portrait,
thread, hologram, serial) and small when views are near-copies (ModelNet turntable renders).

  JaalTaka: CNN+ViT baseline seeds 42, 43, 44 (results/camva/checkpoints), test split, e = the
            concatenated [mean CNN feature of first k views, ViT feature of view 1].
  ModelNet: the fixed-view net of scripts/eval/train_modelnet_vcds.py, 10 classes,
            64/16/20 objects per class, seeds 42, 43, 44, 40 epochs, e = mean conv feature.

Descriptive analysis on two datasets; not a universal law. Writes results/vcds_mechanism.json.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))


def shift_and_acc(emb_by_k: dict[int, np.ndarray], logits_by_k, y):
    e6 = emb_by_k[6]
    cents = np.stack([e6[y == c].mean(0) for c in np.unique(y)])
    d = [np.linalg.norm(cents[i] - cents[j]) for i in range(len(cents)) for j in range(i + 1, len(cents))]
    scale = float(np.mean(d))
    out = {}
    for k in range(1, 7):
        acc = float((logits_by_k[k].argmax(1) == y).mean())
        s = float(np.linalg.norm(emb_by_k[k] - e6, axis=1).mean() / scale)
        out[k] = {"acc": acc, "shift": s}
    return out


@torch.inference_mode()
def jaaltaka(seed: int):
    from roboeye.camva.engine import load_baseline, make_loader
    from roboeye.camva.notes import load_splits
    from roboeye.config import DEVICE
    splits, records = load_splits()
    m = load_baseline(ROOT / f"results/camva/checkpoints/baseline_cnnvit_seed{seed}.pt").to(DEVICE).eval()
    emb, logit, ys = {}, {}, None
    for k in range(1, 7):
        es, ls, yy = [], [], []
        for b in make_loader(splits["test"], records, n_views=k, train=False, batch=16):
            v = b["views"].to(DEVICE)
            e = torch.cat([m.encode_cnn(v), m.vit(v[:, 0])], dim=1)
            es.append(e.cpu().numpy()); ls.append(m.fusion(e).cpu().numpy()); yy.append(b["label"].numpy())
        emb[k], logit[k], ys = np.concatenate(es), np.concatenate(ls), np.concatenate(yy)
    return shift_and_acc(emb, logit, ys)


def modelnet(seed: int, cache: dict):
    import train_modelnet_vcds as mn
    mn.SEED, mn.N_TRAIN, mn.N_VAL, mn.N_TEST = seed, 64, 16, 20
    classes = sorted(p.name for p in mn.SRC.iterdir() if p.is_dir() and p.name != ".vs")[:mn.N_CLASS]
    rng = random.Random(seed)
    data = {s: ([], []) for s in ("train", "val", "test")}
    for ci, name in enumerate(classes):
        files = sorted((mn.SRC / name).glob("*.txt"))
        rng.shuffle(files)
        parts = {"train": files[:64], "val": files[64:80], "test": files[80:100]}
        for split, group in parts.items():
            for p in group:
                if p not in cache:
                    cache[p] = mn.render(mn.load_xyz(p))
                data[split][0].append(cache[p]); data[split][1].append(ci)
    arr = {s: (np.stack(v), np.asarray(y, dtype=np.int64)) for s, (v, y) in data.items()}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = mn.train_one("fixed", *arr["train"], *arr["val"], dev)
    prefix, _, _ = mn.train_one("prefix", *arr["train"], *arr["val"], dev)
    model.eval()
    x, y = arr["test"]
    prefix_acc = {k: mn.accuracy(prefix, x, y, k, dev) for k in range(1, 7)}
    emb, logit = {}, {}
    with torch.inference_mode():
        for k in range(1, 7):
            v = torch.from_numpy(x[:, :k]).to(dev)
            b, kk, h, w = v.shape
            z = model.conv(v.reshape(b * kk, 1, h, w)).reshape(b, kk, 64).mean(1)
            emb[k], logit[k] = z.cpu().numpy(), model.fc(z).cpu().numpy()
    out = shift_and_acc(emb, logit, y)
    for k in range(1, 7):
        out[k]["prefix_acc"] = prefix_acc[k]
    return out


def prmvt_acc(seed: int) -> dict:
    """Prefix-trained reference on JaalTaka: PRMVT re-evaluated after the NaN fix (different architecture)."""
    import json as _j
    f = ROOT / f"results/qduig/prefix_ft/seed{seed}/test_views_20260928/views_1_to_6.json"
    return {int(k): v for k, v in _j.loads(f.read_text())["new"].items()}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    res, points = {"jaaltaka": {}, "modelnet": {}}, []
    cache = {}
    for seed in (42, 43, 44):
        res["jaaltaka"][seed] = jaaltaka(seed)
        for k, a in prmvt_acc(seed).items():
            res["jaaltaka"][seed][k]["prefix_acc"] = a
        res["modelnet"][seed] = modelnet(seed, cache)
        for ds in ("jaaltaka", "modelnet"):
            r = res[ds][seed]
            print(ds, seed, {k: (round(v["acc"], 3), round(v["shift"], 3)) for k, v in r.items()}, flush=True)
            for k in range(1, 6):
                points.append((ds, seed, k, r[k]["shift"], r[6]["acc"] - r[k]["acc"]))
    summary = {}
    for ds in ("jaaltaka", "modelnet", "both"):
        pts = [p for p in points if ds == "both" or p[0] == ds]
        summary[ds] = {"points": len(pts), "spearman_shift_vs_drop": round(spearman([p[3] for p in pts], [p[4] for p in pts]), 3),
                       "mean_shift_k1": round(float(np.mean([p[3] for p in pts if p[2] == 1])), 3),
                       "mean_drop_k1": round(float(np.mean([p[4] for p in pts if p[2] == 1])), 4)}
    for ds in ("jaaltaka", "modelnet"):
        for k in (1, 2, 3):
            fx = [res[ds][s_][k]["acc"] for s_ in (42, 43, 44)]
            pf = [res[ds][s_][k]["prefix_acc"] for s_ in (42, 43, 44)]
            f6 = [res[ds][s_][6]["acc"] for s_ in (42, 43, 44)]
            summary[ds][f"k{k}"] = {"fixed_acc_mean": round(float(np.mean(fx)), 4), "prefix_acc_mean": round(float(np.mean(pf)), 4),
                                    "fixed_acc_k6_mean": round(float(np.mean(f6)), 4),
                                    "total_drop_fixed(6->k)": round(float(np.mean(f6) - np.mean(fx)), 4),
                                    "excess_loss(prefix-fixed at k)": round(float(np.mean(pf) - np.mean(fx)), 4),
                                    "excess_loss_per_seed": [round(p_ - f_, 4) for p_, f_ in zip(pf, fx)]}
    out = {"per_seed": res, "summary": summary, "points(ds,seed,k,shift,drop)": points,
           "note": "JaalTaka prefix reference is PRMVT (Q-DUIG), not the same architecture as the CNN+ViT baseline"}
    dst = ROOT / "results/vcds_mechanism.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
