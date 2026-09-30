"""Equal-weight ensemble of the watermark MobileNetV2, plus a validation stack with DMWR.

Cited system, fixed before the test scores are read:
  train MobileNetV2 on the published watermark crops for seeds 42, 43 and 44,
  same 10-epoch AdamW recipe as the published run, checkpoint = best validation AUC;
  average those three genuine-probabilities with the published checkpoint (seed 0);
  choose the decision threshold by maximum validation accuracy
  (ties keep the threshold closer to 0.5).

Also reported, not used to replace the cited system after the test is known:
  a logistic regression fit on validation only, with two features
  (ensemble probability, and the mean probability of the three saved DMWR checkpoints).
  Its threshold stays 0.5 so the same validation rows are not used twice.

Published weights are not overwritten.
Output: results/watermark_ensemble/summary.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
from roboeye.camva.notes import load_splits  # noqa: E402
import train_watermark_dmwr as dmwr  # noqa: E402

WM = ROOT / "results" / "watermark"
OUT = ROOT / "results" / "watermark_ensemble"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
TR = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomRotation(6),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.03),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    transforms.ToTensor(), NORM,
])
EV = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])
SEEDS = (42, 43, 44)


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)
    return {"n01_first_only_correct": n01, "n10_second_only_correct": n10, "p": p}


def metrics(prob, y, threshold=0.5):
    pred = prob >= threshold
    genuine = y == 1
    return {
        "accuracy": float((pred == genuine).mean()),
        "auc": float(roc_auc_score(y, prob)),
        "threshold": float(threshold),
        "false_counterfeit_on_genuine": int((~pred & genuine).sum()),
        "genuine_n": int(genuine.sum()),
        "counterfeit_missed": int((pred & ~genuine).sum()),
        "counterfeit_n": int((~genuine).sum()),
    }


def val_threshold(prob, y):
    best = (-1.0, 1.0, 0.5)
    for t in np.unique(np.round(prob, 6)):
        acc = float(((prob >= t) == (y == 1)).mean())
        gap = abs(float(t) - 0.5)
        if acc > best[0] or (acc == best[0] and gap < best[1]):
            best = (acc, gap, float(t))
    return best[2]


class Crops(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        nid, y = self.items[i]
        img = Image.open(WM / "crops" / (nid.replace(":", "_") + ".png")).convert("RGB")
        return self.tf(img), y, nid


def mobilenet():
    net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, 2)
    return net


@torch.inference_mode()
def scores(model, loader):
    model.eval()
    p, y, ids = [], [], []
    for x, t, nid in loader:
        p.append(torch.softmax(model(x.to(DEV)), 1)[:, 1].cpu().numpy())
        y.append(t.numpy())
        ids.extend(nid)
    return np.concatenate(p), np.concatenate(y), ids


def train_seed(seed, loaders):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = mobilenet().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best, hist = (-1.0, None), []
    for ep in range(1, 11):
        model.train()
        for x, y, _ in loaders["train"]:
            loss = nn.functional.cross_entropy(model(x.to(DEV)), y.to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv, yv, _ = scores(model, loaders["val"])
        auc = float(roc_auc_score(yv, pv))
        hist.append({"epoch": ep, "val_auc": auc, "val_acc": float(((pv >= 0.5) == (yv == 1)).mean())})
        print("mobilenet", seed, hist[-1], flush=True)
        if auc > best[0]:
            best = (auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    path = ROOT / "models" / f"watermark_mobilenetv2_ens_seed{seed}.pt"
    torch.save({"state_dict": model.state_dict(), "arch": "mobilenet_v2", "seed": seed, "val_auc": best[0],
                "classes": ["counterfeit", "genuine"]}, path)
    return model, best[0], hist


def load_published():
    blob = torch.load(ROOT / "models" / "watermark_mobilenetv2.pt", map_location="cpu", weights_only=False)
    net = models.mobilenet_v2(weights=None)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, 2)
    net.load_state_dict(blob["state_dict"])
    return net.to(DEV).eval()


def dmwr_probs(note_ids, images, rows_by_id, protos, mu, sd):
    acc = np.zeros(len(note_ids), np.float64)
    for seed in SEEDS:
        blob = torch.load(ROOT / "models" / f"watermark_dmwr_both_seed{seed}.pt", map_location="cpu", weights_only=False)
        model = dmwr.DMWR("both")
        model.load_state_dict(blob["state_dict"])
        model.to(DEV).eval()
        probs = []
        with torch.inference_mode():
            for i in range(0, len(note_ids), 64):
                qs, rs, ss = [], [], []
                for nid in note_ids[i:i + 64]:
                    row = rows_by_id[nid]
                    q = images[nid]
                    proto = protos[row["proto_key"]]
                    resid = q - proto
                    resid = (resid - resid.mean()) / (resid.std() + 1e-6)
                    sc = ((row["raw_scalars"] - mu) / sd).astype(np.float32)
                    qs.append(q)
                    rs.append(resid)
                    ss.append(sc)
                logit = model(
                    torch.from_numpy(np.stack(qs)[:, None]).to(DEV),
                    torch.from_numpy(np.stack(rs)[:, None]).to(DEV),
                    torch.from_numpy(np.stack(ss)).to(DEV),
                )
                probs.append(torch.softmax(logit, 1)[:, 1].cpu().numpy())
        acc += np.concatenate(probs)
    return acc / len(SEEDS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    splits, records = load_splits(ROOT / "results" / "serial_split")
    ok = {r["note_id"] for r in json.loads((WM / "features.json").read_text(encoding="utf-8")) if r["ok"]}
    items = {s: [(n, int(records[n]["label"])) for n in splits[s] if n in ok] for s in ("train", "val", "test")}
    loaders = {
        s: DataLoader(Crops(items[s], TR if s == "train" else EV), batch_size=32, shuffle=(s == "train"))
        for s in items
    }
    seed_runs = []
    val_ps, test_ps = [], []
    for seed in SEEDS:
        path = ROOT / "models" / f"watermark_mobilenetv2_ens_seed{seed}.pt"
        if path.exists():
            blob = torch.load(path, map_location="cpu", weights_only=False)
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
            model.load_state_dict(blob["state_dict"])
            model = model.to(DEV)
            val_auc, hist = float(blob["val_auc"]), []
            print("loaded", path.name, "val_auc", val_auc, flush=True)
        else:
            model, val_auc, hist = train_seed(seed, loaders)
        pv, yv, val_ids = scores(model, loaders["val"])
        pt, yt, test_ids = scores(model, loaders["test"])
        val_ps.append(pv)
        test_ps.append(pt)
        seed_runs.append({"seed": seed, "best_val_auc": val_auc, "history": hist, "test": metrics(pt, yt)})
        del model
        torch.cuda.empty_cache()

    published = load_published()
    pub_val, yv, val_ids = scores(published, loaders["val"])
    pub_test, yt, test_ids = scores(published, loaders["test"])
    val_ps.append(pub_val)
    test_ps.append(pub_test)

    ens_val = np.mean(val_ps, axis=0)
    ens_test = np.mean(test_ps, axis=0)
    thr = val_threshold(ens_val, yv)
    y_test = yt
    cited = metrics(ens_test, y_test, thr)
    at_half = metrics(ens_test, y_test, 0.5)
    published_m = metrics(pub_test, y_test, 0.5)
    cited_correct = (ens_test >= thr) == (y_test == 1)
    pub_correct = (pub_test >= 0.5) == (y_test == 1)

    feat = {r["note_id"]: r for r in json.loads((WM / "features.json").read_text(encoding="utf-8")) if r["ok"]}
    train_genuine = [n for n, y in items["train"] if y == 1]
    needed = [n for n, _ in items["train"]] + list(val_ids) + list(test_ids)
    images = {nid: dmwr.quotient(WM / "crops" / (nid.replace(":", "_") + ".png")) for nid in dict.fromkeys(needed)}
    by_denom: dict[str, list] = {}
    for nid in train_genuine:
        by_denom.setdefault(str(feat[nid].get("denom") or "ALL"), []).append(nid)
    global_proto = np.median(np.stack([images[n] for n in train_genuine]), axis=0).astype(np.float32)
    protos = {"ALL": global_proto}
    for denom, ids in by_denom.items():
        if len(ids) >= dmwr.MIN_PROTO:
            protos[denom] = np.median(np.stack([images[n] for n in ids]), axis=0).astype(np.float32)

    def proto_key(nid):
        key = str(feat[nid].get("denom") or "ALL")
        return key if key in protos else "ALL"

    rows_by_id = {}
    train_raw = []
    for nid, _ in items["train"]:
        key = proto_key(nid)
        raw = dmwr.scalars_of(images[nid], protos[key])
        rows_by_id[nid] = {"proto_key": key, "raw_scalars": raw}
        train_raw.append(raw)
    mu = np.stack(train_raw).mean(0)
    sd = np.stack(train_raw).std(0) + 1e-6
    for nid in list(val_ids) + list(test_ids):
        key = proto_key(nid)
        rows_by_id[nid] = {"proto_key": key, "raw_scalars": dmwr.scalars_of(images[nid], protos[key])}

    d_val = dmwr_probs(val_ids, images, rows_by_id, protos, mu, sd)
    d_test = dmwr_probs(test_ids, images, rows_by_id, protos, mu, sd)
    stack = LogisticRegression(max_iter=1000)
    stack.fit(np.column_stack([ens_val, d_val]), yv)
    stack_test = stack.predict_proba(np.column_stack([ens_test, d_test]))[:, 1]
    stacked = metrics(stack_test, y_test, 0.5)

    summary = {
        "cited": "equal mean of published MobileNetV2 and seeds 42, 43, 44; threshold maximises validation accuracy",
        "n": {s: len(v) for s, v in items.items()},
        "device": str(DEV),
        "seeds": seed_runs,
        "published_rescored": published_m,
        "ensemble_at_0.5": at_half,
        "ensemble_val_threshold": cited,
        "val_threshold": thr,
        "val_accuracy_at_threshold": float(((ens_val >= thr) == (yv == 1)).mean()),
        "stack_with_dmwr_threshold_0.5": stacked,
        "stack_coef": np.round(stack.coef_[0], 6).tolist(),
        "mcnemar_ensemble_vs_published": mcnemar(cited_correct, pub_correct),
        "mcnemar_stack_vs_published": mcnemar((stack_test >= 0.5) == (y_test == 1), pub_correct),
        "test_note_ids": list(test_ids),
        "ensemble_test_probs": np.round(ens_test, 6).tolist(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "published_rescored", "ensemble_at_0.5", "ensemble_val_threshold", "stack_with_dmwr_threshold_0.5",
        "mcnemar_ensemble_vs_published", "mcnemar_stack_vs_published")}, indent=1))


if __name__ == "__main__":
    main()
