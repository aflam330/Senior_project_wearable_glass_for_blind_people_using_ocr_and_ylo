"""Task 1: beat the ResNet-50 baselines on unseen counterfeit prints (serial-disjoint split), 3 seeds.

Methods (all fitted on TRAIN, chosen / weighted on VAL, TEST read once):
  PROBE      frozen ResNet-50 view features + logistic regression (deterministic; C on VAL)
  FT_RESNET  fine-tuned ResNet-50 (scripts/train/ft_resnet_serial.py), if its files exist
  NETWORK    prefix network, shared BN (results/serial_split/runs/...)
  FUSION     attention fusion over tokens [view_1..view_k features, watermark-window feature]
             (frozen ImageNet ResNet-50 features for both; views and window from different photos),
             trained on TRAIN with random k in 1..6 and the watermark token dropped with p = 0.2;
             epoch (of 40) chosen on VAL mean accuracy over k with the watermark token
  WATERMARK  MobileNetV2 watermark classifier (chosen on VAL AUC earlier)
  ENSEMBLE   mean of NETWORK, FUSION and WATERMARK probabilities weighted by 1 / VAL log-loss
  HYBRID_FINAL  logistic regression on VAL over [NETWORK logit, WATERMARK logit, missing flag] (hybrid_final_seeds.json)
Paired exact McNemar against PROBE and FT_RESNET per seed, and a pooled McNemar over the 3 seeds.
Output: results/serial_split/beat_resnet.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
from roboeye.camva.notes import load_splits  # noqa: E402
from hybrid_serial_split import RUNS, preds  # noqa: E402

R = ROOT / "results"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mcnemar(a, b):
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n = n01 + n10
    return [n01, n10, 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)]


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


class Fusion(nn.Module):
    def __init__(self, d=2048, h=256):
        super().__init__()
        self.view = nn.Sequential(nn.Linear(d, h), nn.GELU())
        self.wm = nn.Sequential(nn.Linear(d, h), nn.GELU())
        self.kind = nn.Parameter(torch.zeros(2, h))
        self.att = nn.Linear(h, 1)
        self.cls = nn.Sequential(nn.Dropout(0.3), nn.Linear(h, 2))

    def forward(self, V, vm, W, wmask):  # V (B,6,d), vm (B,6), W (B,d), wmask (B,)
        tok = torch.cat([self.view(V) + self.kind[0], (self.wm(W) + self.kind[1]).unsqueeze(1)], 1)
        mask = torch.cat([vm, wmask.unsqueeze(1)], 1)
        a = self.att(tok).squeeze(-1).masked_fill(mask == 0, -1e9)
        return self.cls((torch.softmax(a, 1).unsqueeze(-1) * tok).sum(1))


def main() -> None:
    splits, records = load_splits(R / "serial_split")
    ids = {s: list(splits[s]) for s in ("train", "val", "test")}
    y = {s: np.array([int(records[n]["label"]) for n in ids[s]]) for s in ids}
    fv = np.load(R / "sota/feats_resnet50.npz", allow_pickle=True)
    pos = {n: i for i, n in enumerate(fv["ids"])}
    V = {s: fv["x"][[pos[n] for n in ids[s]]].astype(np.float32) for s in ids}
    fw = np.load(R / "watermark/crop_feats_resnet50.npz", allow_pickle=True)
    wpos = {n: i for i, n in enumerate(fw["ids"])}
    W = {s: np.stack([fw["x"][wpos[n]] if n in wpos else np.zeros(2048, np.float32) for n in ids[s]]).astype(np.float32) for s in ids}
    Wm = {s: np.array([n in wpos for n in ids[s]], np.float32) for s in ids}
    res = {"per_seed": {}}
    # PROBE (deterministic)
    best = None
    for c in (0.01, 0.1, 1.0, 10.0, 100.0):
        m = LogisticRegression(C=c, max_iter=3000).fit(V["train"].reshape(-1, 2048), np.repeat(y["train"], 6))
        pv = m.predict_proba(V["val"].reshape(-1, 2048))[:, 1].reshape(-1, 6)
        a = np.mean([((pv[:, :k].mean(1) >= 0.5) == (y["val"] == 1)).mean() for k in range(1, 7)])
        if best is None or a > best[0]:
            best = (a, m)
    probe = best[1].predict_proba(V["test"].reshape(-1, 2048))[:, 1].reshape(-1, 6)
    # watermark MobileNetV2 scores
    from hybrid_v2_serial_split import WM as WMDIR  # noqa: F401
    import hybrid_v2_serial_split as hv
    hv.WM_TAG = "watermark_mobilenetv2"
    ok = {r["note_id"] for r in json.loads((R / "watermark/features.json").read_text(encoding="utf-8")) if r["ok"]}
    wmp = {s: hv.wm_scores(ids[s], ok) for s in ("val", "test")}
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    ll = lambda p, t: float(-np.mean(t * np.log(np.clip(p, 1e-6, 1)) + (1 - t) * np.log(np.clip(1 - p, 1e-6, 1))))
    for seed, run in RUNS.items():
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        model = Fusion().to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        T = {s: [torch.tensor(a[s]).to(DEV) for a in (V, W, Wm)] for s in ids}
        yt = torch.tensor(y["train"]).to(DEV)

        def infer(s, k, use_wm=True):
            model.eval()
            Vs, Ws, Ms = T[s]
            vm = torch.zeros(len(ids[s]), 6, device=DEV); vm[:, :k] = 1
            with torch.no_grad():
                return torch.softmax(model(Vs, vm, Ws, Ms * float(use_wm)), 1)[:, 1].cpu().numpy()
        bestf = (-1, None)
        for ep in range(40):
            model.train()
            perm = rng.permutation(len(ids["train"]))
            for i in range(0, len(perm), 64):
                b = torch.tensor(perm[i:i + 64]).to(DEV)
                k = torch.tensor(rng.integers(1, 7, len(b))).to(DEV)
                vm = (torch.arange(6, device=DEV)[None] < k[:, None]).float()
                wm = T["train"][2][b] * (torch.rand(len(b), device=DEV) > 0.2).float()
                loss = nn.functional.cross_entropy(model(T["train"][0][b], vm, T["train"][1][b], wm), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
            s_val = np.mean([((infer("val", k) >= 0.5) == (y["val"] == 1)).mean() for k in range(1, 7)])
            if s_val > bestf[0]:
                bestf = (s_val, {kk: v.detach().clone() for kk, v in model.state_dict().items()})
        model.load_state_dict(bestf[1])
        out = {}
        for k in (1, 6):
            net = {s: np.array([preds(run, f"{s}_views", k)[n] for n in ids[s]]) for s in ("val", "test")}
            fus = {s: infer(s, k) for s in ("val", "test")}
            wts = np.array([1 / ll(net["val"], y["val"]), 1 / ll(fus["val"], y["val"]), 1 / ll(wmp["val"], y["val"])])
            wts = wts / wts.sum()
            ens = {s: wts[0] * net[s] + wts[1] * fus[s] + wts[2] * wmp[s] for s in ("val", "test")}
            has = {sp: np.array([n in ok for n in ids[sp]]) for sp in ("val", "test")}
            Fh = {sp: np.column_stack([lg(net[sp]), lg(wmp[sp]), (~has[sp]).astype(float)]) for sp in ("val", "test")}
            hyb = LogisticRegression(max_iter=2000).fit(Fh["val"], y["val"]).predict_proba(Fh["test"])[:, 1]
            methods = {"PROBE": probe[:, :k].mean(1), "NETWORK": net["test"], "FUSION": fus["test"], "HYBRID_FINAL": hyb,
                       "FUSION_no_watermark": infer("test", k, use_wm=False), "ENSEMBLE": ens["test"]}
            ft = R / "serial_split/ft_resnet50" / f"seed{seed}.json"
            if ft.is_file():
                fj = json.loads(ft.read_text(encoding="utf-8"))
                fp = dict(zip(fj["test_note_ids"], np.array(fj["test_probs"])[:, :k].mean(1)))
                methods["FT_RESNET"] = np.array([fp[n] for n in ids["test"]])
            right = {m: (p >= 0.5) == (y["test"] == 1) for m, p in methods.items()}
            out[f"k{k}"] = {"ensemble_weights [network, fusion, watermark]": wts.round(3).tolist()}
            for m, p in methods.items():
                r = right[m]
                gen = y["test"] == 1
                out[f"k{k}"][m] = {"accuracy": float(r.mean()), "wilson95": wilson(int(r.sum()), len(r)),
                                   "false_counterfeit_on_genuine": int((~(p >= 0.5) & gen).sum()),
                                   "counterfeit_missed": int(((p >= 0.5) & ~gen).sum()),
                                   "right": r.astype(int).tolist()}
            for m in ("NETWORK", "FUSION", "ENSEMBLE", "HYBRID_FINAL"):
                out[f"k{k}"][m]["mcnemar_vs_PROBE"] = mcnemar(right[m], right["PROBE"])
                if "FT_RESNET" in right:
                    out[f"k{k}"][m]["mcnemar_vs_FT_RESNET"] = mcnemar(right[m], right["FT_RESNET"])
            print(seed, k, {m: round(v["accuracy"], 4) for m, v in out[f"k{k}"].items() if isinstance(v, dict)}, flush=True)
        res["per_seed"][seed] = out
    summ = {}
    for k in ("k1", "k6"):
        for m in res["per_seed"][42][k]:
            if not isinstance(res["per_seed"][42][k][m], dict):
                continue
            a = [res["per_seed"][s][k][m]["accuracy"] for s in res["per_seed"] if m in res["per_seed"][s][k]]
            summ[f"{k}/{m}"] = {"mean": float(np.mean(a)), "sd": float(np.std(a, ddof=1)) if len(a) > 1 else None, "n_seeds": len(a)}
        for m in ("NETWORK", "FUSION", "ENSEMBLE", "HYBRID_FINAL"):
            for base in ("PROBE", "FT_RESNET"):
                seeds = [s for s in res["per_seed"] if base in res["per_seed"][s][k]]
                if not seeds:
                    continue  # baseline not run yet (e.g. FT_RESNET before its training finishes)
                A = np.concatenate([np.array(res["per_seed"][s][k][m]["right"], bool) for s in seeds])
                B = np.concatenate([np.array(res["per_seed"][s][k][base]["right"], bool) for s in seeds])
                summ[f"{k}/{m}_vs_{base}_pooled_mcnemar"] = mcnemar(A, B)
    res["summary"] = summ
    (R / "serial_split/beat_resnet.json").write_text(json.dumps(res), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
