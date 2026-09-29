"""Unsupervised domain adaptation for the whole-note counterfeit check (approaches 8 and 9 of SYNTHETIC_FIX.md).

Adaptation data (unlabelled, never scored): Bangla Money "Testing" photos (333, no labels) whose top YOLO box
is 500 or 1000 BDT, cropped like the app (640 px, landscape) and cut into the four view windows.
  ADABN  PRMVT (prefix_ft seed 42): batch-norm running statistics re-estimated on the adaptation views
         (cumulative average, no gradient, no labels); weights unchanged.
  CORAL  frozen ResNet-50 view features: adaptation-set features whitened and re-coloured to the JaalTaka
         TRAIN view-feature covariance and mean; then the JaalTaka-trained logistic regression (C = 100).
Scored once on the cached whole-note crops (results/jaal_whole/crops): counterfeit-set originals (AUC and
counterfeit passed at 0.5), and genuine photos of the counterfeit set, Bangla Money Training and BanglaTaka
(share called counterfeit at 0.5). Baselines (S4 PRMVT, R2 probe without adaptation) recomputed the same way.
Success rule (fixed before scoring): AUC >= baseline AND genuine called counterfeit < 5 % on Bangla Money and BanglaTaka.
Output: results/jaal_whole/domain_adapt.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORK / "savior_glass"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
from roboeye.authenticity import default_transform  # noqa: E402
from roboeye.camva.notes import SPLIT_DIR, load_splits  # noqa: E402
from roboeye.qduig.engine import load_qduig  # noqa: E402
import eval_backbone_probes as probes  # noqa: E402
from train_whole_note_synth import cut_views  # noqa: E402

J = ROOT / "results" / "jaal_whole"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TF = default_transform(train=False)
TESTING = WORK / "data set for comparison" / "Bangla Money dataset Kaggle" / "bangla_banknote_v2" / "Testing"


def adaptation_crops():
    from modes.currency_mode import CurrencyMode
    mode = CurrencyMode()
    mode._load_yolo()
    out = []
    for p in sorted(TESTING.iterdir()):
        img = cv2.imread(str(p))
        if img is None:
            continue
        hits = mode.detect_live(img)
        if not hits or hits[0]["name"] not in ("500_taka", "1000_taka"):
            continue
        x, y, w, h = hits[0]["bbox"]
        c = img[max(0, y):y + h, max(0, x):x + w]
        c = cv2.resize(c, None, fx=640 / c.shape[1], fy=640 / c.shape[1], interpolation=cv2.INTER_AREA)
        out.append(cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE) if c.shape[0] > c.shape[1] else c)
    return out


def view_tensor(crop):
    return torch.stack([TF(Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))) for v in cut_views(crop)])


@torch.inference_mode()
def prmvt_prob(net, crop):
    x = view_tensor(crop).unsqueeze(0).to(DEV)
    return float(net(x, torch.ones(1, 4, dtype=torch.long, device=DEV))["prob"].reshape(-1)[0])


def adabn(net, crops):
    net = copy.deepcopy(net).eval()
    bns = [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    with torch.no_grad():
        for i in range(0, len(crops), 8):
            x = torch.stack([view_tensor(c) for c in crops[i:i + 8]]).to(DEV)
            net(x, torch.ones(x.size(0), 4, dtype=torch.long, device=DEV))
    return net.eval()


def main() -> None:
    adapt = adaptation_crops()
    print("adaptation crops (500/1000 in unlabelled Testing):", len(adapt), flush=True)
    rows = [r for r in json.loads((J / "crops_index.json").read_text(encoding="utf-8"))
            if r.get("crop") and r["yolo"] in ("500_taka", "1000_taka") and not r["augmented"]]
    crops = [cv2.imread(str(J / "crops" / r["crop"])) for r in rows]
    net = load_qduig(ROOT / "results/qduig/prefix_ft/seed42/checkpoint.pt").to(DEV).eval()
    net_a = adabn(net, adapt)
    # ResNet-50 probe with and without CORAL
    rmodel, rtf = probes.backbone("resnet50")
    splits, records = load_splits(SPLIT_DIR)
    f = np.load(ROOT / "results/sota/feats_resnet50.npz", allow_pickle=True)
    pos = {n: i for i, n in enumerate(f["ids"])}
    xtr = f["x"][[pos[n] for n in splits["train"]]].reshape(-1, 2048)
    lr = LogisticRegression(C=100.0, max_iter=3000).fit(xtr, np.repeat([int(records[n]["label"]) for n in splits["train"]], 6))

    @torch.inference_mode()
    def feats(crop):
        v = torch.stack([rtf(Image.fromarray(cv2.cvtColor(w, cv2.COLOR_BGR2RGB))) for w in cut_views(crop)[:2]]).to(DEV)
        z = rmodel(v).float().cpu().numpy()
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    fa = np.concatenate([feats(c) for c in adapt])
    eps = 1e-3
    cs = np.cov(fa, rowvar=False) + eps * np.eye(2048)
    ct = np.cov(xtr, rowvar=False) + eps * np.eye(2048)

    def msqrt(a, inv=False):
        w, v = np.linalg.eigh(a)
        w = np.clip(w, eps, None)
        return (v * (w ** (-0.5 if inv else 0.5))) @ v.T
    A = msqrt(cs, inv=True) @ msqrt(ct)
    mu_s, mu_t = fa.mean(0), xtr.mean(0)
    scores = {"S4": [], "S4_adabn": [], "R2": [], "R2_coral": []}
    for c in crops:
        scores["S4"].append(prmvt_prob(net, c))
        scores["S4_adabn"].append(prmvt_prob(net_a, c))
        z = feats(c)
        scores["R2"].append(float(lr.predict_proba(z)[:, 1].mean()))
        scores["R2_coral"].append(float(lr.predict_proba((z - mu_s) @ A + mu_t)[:, 1].mean()))
    res = {"adaptation_crops": len(adapt), "methods": {}}
    for name, s in scores.items():
        s = np.array(s)
        e = {}
        cf = np.array([r["set"] == "cf" for r in rows])
        yg = np.array([r["truth"] == "genuine" for r in rows])
        e["auc_cf_originals"] = float(roc_auc_score(yg[cf], s[cf]))
        e["cf_counterfeit_passed_at_0.5"] = f"{int((s[cf & ~yg] >= 0.5).sum())}/{int((cf & ~yg).sum())}"
        for setname in ("cf", "bm", "bt"):
            m = np.array([r["set"] == setname for r in rows]) & yg
            e[f"{setname}_genuine_called_counterfeit"] = float((s[m] < 0.5).mean())
        e["success_rule_met"] = bool(e["bm_genuine_called_counterfeit"] < 0.05 and e["bt_genuine_called_counterfeit"] < 0.05)
        res["methods"][name] = e
        print(name, e, flush=True)
    (J / "domain_adapt.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
