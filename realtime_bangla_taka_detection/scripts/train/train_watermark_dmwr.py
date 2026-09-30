"""Denomination-matched watermark residual (DMWR).

A back-lit watermark is a thickness portrait of one denomination, not a generic texture.
This script builds that portrait from training genuine crops only, then classifies the
window by how it agrees with the portrait and what is left over.

Quotient image: grey crop divided by a wide Gaussian, then z-scored, so backlight
level drops out. Prototype T_d is the pixel median of training genuine quotient images
of denomination d (global training-genuine median if that denomination has fewer than
8 such notes). Residual R = Q - T_d. The matched-filter score is the normalised
correlation of Q with T_d.

Cited model (precommitted, not chosen after seeing the test): dual stream.
  portrait stream on Q, residual stream on R, plus four scalars
  (correlation, residual std, residual Laplacian variance, fine-detail energy).
Ablations trained the same way and not used to pick a winner: portrait stream only,
residual stream plus scalars, and a logistic regression on the scalars and the four
hand-crafted crop features already in features.json.

Protocol matches the published MobileNetV2 watermark run: serial-disjoint split,
crops that registered, epoch by validation AUC, decision threshold 0.5, test read
once. Seeds 42, 43, 44. The logistic threshold is the validation accuracy maximum.
Published weights are not overwritten.

Output: results/watermark_dmwr/summary.json and models/watermark_dmwr_<mode>_seed<k>.pt
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from roboeye.camva.notes import load_splits  # noqa: E402

WM = ROOT / "results" / "watermark"
OUT = ROOT / "results" / "watermark_dmwr"
SIDE = 128
SIGMA = 16.0
MIN_PROTO = 8
SEEDS = (42, 43, 44)
EPOCHS = 12
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)
    return {"n01_first_only_correct": n01, "n10_second_only_correct": n10, "p": p}


def quotient(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (SIDE, SIDE), interpolation=cv2.INTER_AREA).astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), SIGMA)
    q = g / np.clip(blur, 1.0, None)
    return (q - q.mean()) / (q.std() + 1e-6)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = a - a.mean(), b - b.mean()
    return float((aa * bb).sum() / (np.sqrt((aa * aa).sum() * (bb * bb).sum()) + 1e-6))


def scalars_of(q: np.ndarray, proto: np.ndarray) -> np.ndarray:
    r = q - proto
    fine = np.abs(q - cv2.GaussianBlur(q, (0, 0), 2))
    return np.array(
        [ncc(q, proto), float(r.std()), float(cv2.Laplacian(r, cv2.CV_32F).var()), float(fine.mean())],
        np.float32,
    )


def augment(q: np.ndarray) -> np.ndarray:
    ang = float(np.random.uniform(-5, 5))
    shift = np.random.uniform(-3, 3, size=2)
    centre = ((SIDE - 1) / 2, (SIDE - 1) / 2)
    m = cv2.getRotationMatrix2D(centre, ang, 1.0)
    m[0, 2] += shift[0]
    m[1, 2] += shift[1]
    return cv2.warpAffine(q, m, (SIDE, SIDE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class Crops(Dataset):
    def __init__(self, rows, images, protos, train: bool):
        self.rows, self.images, self.protos, self.train = rows, images, protos, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        q = self.images[row["note_id"]]
        if self.train:
            q = augment(q)
        proto = self.protos[row["proto_key"]]
        resid = q - proto
        resid = (resid - resid.mean()) / (resid.std() + 1e-6)
        return (
            torch.from_numpy(q[None]),
            torch.from_numpy(resid[None]),
            torch.from_numpy(row["scalars"]),
            row["label"],
        )


class Stream(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class DMWR(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.portrait = Stream()
        self.residual = Stream()
        width = 4
        if mode != "residual":
            width += 64
        if mode != "portrait":
            width += 64
        self.fc = nn.Linear(width, 2)

    def forward(self, q, resid, sc):
        parts = []
        if self.mode != "residual":
            parts.append(self.portrait(q))
        if self.mode != "portrait":
            parts.append(self.residual(resid))
        parts.append(sc)
        return self.fc(torch.cat(parts, 1))


@torch.inference_mode()
def predict(model, loader):
    model.eval()
    prob, y = [], []
    for q, resid, sc, t in loader:
        logit = model(q.to(DEV), resid.to(DEV), sc.to(DEV))
        prob.append(torch.softmax(logit, 1)[:, 1].cpu().numpy())
        y.append(t.numpy())
    return np.concatenate(prob), np.concatenate(y)


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
    """Highest validation accuracy. Ties keep the threshold closer to 0.5."""
    best = ( -1.0, 1.0, 0.5)
    for t in np.unique(prob):
        acc = float(((prob >= t) == (y == 1)).mean())
        gap = abs(float(t) - 0.5)
        if acc > best[0] or (acc == best[0] and gap < best[1]):
            best = (acc, gap, float(t))
    return best[2]


def train_one(mode, seed, rows, images, protos, scalar_mu, scalar_sd):
    torch.manual_seed(seed)
    np.random.seed(seed)
    packs = {}
    for split, group in rows.items():
        packed = []
        for row in group:
            sc = (row["raw_scalars"] - scalar_mu) / scalar_sd
            packed.append({**row, "scalars": sc.astype(np.float32)})
        packs[split] = packed
    loaders = {
        s: DataLoader(Crops(packs[s], images, protos, train=(s == "train")), batch_size=32, shuffle=(s == "train"))
        for s in packs
    }
    model = DMWR(mode).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, hist = (-1.0, None), []
    for ep in range(1, EPOCHS + 1):
        model.train()
        for q, resid, sc, y in loaders["train"]:
            loss = nn.functional.cross_entropy(model(q.to(DEV), resid.to(DEV), sc.to(DEV)), y.to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv, yv = predict(model, loaders["val"])
        auc = float(roc_auc_score(yv, pv))
        hist.append({"epoch": ep, "val_auc": auc, "val_acc": float(((pv >= 0.5) == (yv == 1)).mean())})
        print(mode, seed, hist[-1], flush=True)
        if auc > best[0]:
            best = (auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    pt, yt = predict(model, loaders["test"])
    pv, yv = predict(model, loaders["val"])
    torch.save(
        {"state_dict": model.state_dict(), "arch": "dmwr", "mode": mode, "seed": seed,
         "classes": ["counterfeit", "genuine"], "side": SIDE, "val_auc": best[0]},
        ROOT / "models" / f"watermark_dmwr_{mode}_seed{seed}.pt",
    )
    return {
        "mode": mode, "seed": seed, "best_val_auc": best[0], "history": hist,
        "val": metrics(pv, yv), "test": metrics(pt, yt),
        "test_note_ids": [r["note_id"] for r in packs["test"]],
        "test_probs": np.round(pt, 6).tolist(),
        "test_correct": ((pt >= 0.5) == (yt == 1)).tolist(),
    }


def published_mobilenet(note_ids):
    blob = torch.load(ROOT / "models" / "watermark_mobilenetv2.pt", map_location="cpu", weights_only=False)
    net = models.mobilenet_v2(weights=None)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, 2)
    net.load_state_dict(blob["state_dict"])
    net.to(DEV).eval()
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    probs = []
    with torch.inference_mode():
        for i in range(0, len(note_ids), 32):
            batch = []
            for nid in note_ids[i:i + 32]:
                bgr = cv2.imread(str(WM / "crops" / (nid.replace(":", "_") + ".png")), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                batch.append(tf(rgb))
            x = torch.stack(batch).to(DEV)
            probs.append(torch.softmax(net(x), 1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    splits, records = load_splits(ROOT / "results" / "serial_split")
    feat = {r["note_id"]: r for r in json.loads((WM / "features.json").read_text(encoding="utf-8")) if r["ok"]}
    images = {nid: quotient(WM / "crops" / (nid.replace(":", "_") + ".png")) for nid in feat}
    train_genuine = []
    for nid in splits["train"]:
        if nid in feat and int(records[nid]["label"]) == 1:
            train_genuine.append(nid)
    by_denom: dict[str, list[str]] = {}
    for nid in train_genuine:
        by_denom.setdefault(str(feat[nid].get("denom") or "ALL"), []).append(nid)
    global_proto = np.median(np.stack([images[n] for n in train_genuine]), axis=0).astype(np.float32)
    protos = {"ALL": global_proto}
    proto_n = {"ALL": len(train_genuine)}
    for denom, ids in by_denom.items():
        if len(ids) >= MIN_PROTO:
            protos[denom] = np.median(np.stack([images[n] for n in ids]), axis=0).astype(np.float32)
            proto_n[denom] = len(ids)
    np.savez_compressed(OUT / "prototypes.npz", **{k.replace(" ", "_"): v for k, v in protos.items()})

    def proto_key(nid):
        key = str(feat[nid].get("denom") or "ALL")
        return key if key in protos else "ALL"

    rows = {s: [] for s in ("train", "val", "test")}
    for split, ids in splits.items():
        for nid in ids:
            if nid not in feat:
                continue
            key = proto_key(nid)
            raw = scalars_of(images[nid], protos[key])
            hand = feat[nid]
            extra = np.array([hand["lap_var"], hand["std"], hand["edges"], hand["rel_mean"]], np.float32)
            rows[split].append({
                "note_id": nid, "label": int(records[nid]["label"]), "proto_key": key,
                "raw_scalars": raw, "logistic": np.concatenate([raw, extra]),
            })
    mu = np.stack([r["raw_scalars"] for r in rows["train"]]).mean(0)
    sd = np.stack([r["raw_scalars"] for r in rows["train"]]).std(0) + 1e-6

    x_mu = np.stack([r["logistic"] for r in rows["train"]]).mean(0)
    x_sd = np.stack([r["logistic"] for r in rows["train"]]).std(0) + 1e-6
    pack = lambda group: np.stack([(r["logistic"] - x_mu) / x_sd for r in group])
    y = {s: np.array([r["label"] for r in rows[s]]) for s in rows}
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(pack(rows["train"]), y["train"])
    val_p = lr.predict_proba(pack(rows["val"]))[:, 1]
    test_p = lr.predict_proba(pack(rows["test"]))[:, 1]
    thr = val_threshold(val_p, y["val"])
    logistic = {
        "features": ["ncc", "resid_std", "resid_lap", "fine_energy", "lap_var", "std", "edges", "rel_mean"],
        "coef": np.round(lr.coef_[0], 6).tolist(),
        "val": metrics(val_p, y["val"], thr),
        "test": metrics(test_p, y["test"], thr),
        "test_at_0.5": metrics(test_p, y["test"], 0.5),
    }

    runs = []
    for mode in ("both", "portrait", "residual"):
        for seed in SEEDS:
            runs.append(train_one(mode, seed, rows, images, protos, mu, sd))

    cited = [r for r in runs if r["mode"] == "both"]
    published_ids = cited[0]["test_note_ids"]
    pub = published_mobilenet(published_ids)
    y_test = np.array([records[n]["label"] for n in published_ids])
    pub_correct = (pub >= 0.5) == (y_test == 1)
    comparisons = []
    for run in cited:
        correct = np.array(run["test_correct"], dtype=bool)
        comparisons.append({"seed": run["seed"], "mcnemar_dmwr_vs_published_mobilenet": mcnemar(correct, pub_correct)})

    def mean_std(mode, field):
        vals = [r["test"][field] for r in runs if r["mode"] == mode]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "values": vals}

    summary = {
        "algorithm": "denomination-matched watermark residual",
        "cited_mode": "both",
        "selection": "epoch by validation AUC; threshold 0.5; cited mode precommitted as dual stream",
        "split": "serial_split",
        "n": {s: len(v) for s, v in rows.items()},
        "prototype_notes": proto_n,
        "device": str(DEV),
        "logistic_matched_filter": logistic,
        "runs": [{k: v for k, v in r.items() if k != "test_correct"} for r in runs],
        "test_summary": {
            mode: {"accuracy": mean_std(mode, "accuracy"), "auc": mean_std(mode, "auc")}
            for mode in ("both", "portrait", "residual")
        },
        "published_mobilenet_on_same_test_notes": metrics(pub, y_test),
        "mcnemar": comparisons,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({"logistic_test": logistic["test"], "cited": summary["test_summary"]["both"],
                      "published": summary["published_mobilenet_on_same_test_notes"], "mcnemar": comparisons}, indent=1))


if __name__ == "__main__":
    main()
