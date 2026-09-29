"""Approach A evaluation, following results/jaal_whole/approach_a/PREREGISTERED.md.

tau_A = highest p(genuine) of any synthetic VAL counterfeit composite (4 views, fixed composites).
Scored once: synthetic TEST composites; real whole-note crops (results/jaal_whole/crops) with the
fine-tuned model (A) and with the deployed PRMVT (S4, tau 0.99959) for the pre-registered comparison.
Output: results/jaal_whole/approach_a/eval.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from roboeye.qduig.engine import load_qduig  # noqa: E402
from train_whole_note_synth import COCO, DEV, OUT, PRMVT, SYN, TF, SynthNotes, cut_views  # noqa: E402
from jaal_whole_note import wilson  # noqa: E402

TAU_S4 = 0.9995918869972229


@torch.inference_mode()
def probs_loader(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        x = x.to(DEV)
        m = torch.ones(x.size(0), x.size(1), dtype=torch.long, device=DEV)
        ps.append(model(x, m)["prob"].reshape(-1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


@torch.inference_mode()
def prob_crop(model, crop) -> float:
    x = torch.stack([TF(Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))) for v in cut_views(crop)]).unsqueeze(0).to(DEV)
    return float(model(x, torch.ones(1, 4, dtype=torch.long, device=DEV))["prob"].reshape(-1)[0])


def main() -> None:
    idx = [r for r in json.loads((SYN / "synth_index.json").read_text(encoding="utf-8"))["notes"] if r.get("kept")]
    bgs = sorted(COCO.glob("*.jpg"))
    model_a = load_qduig(OUT / "checkpoint.pt").to(DEV).eval()
    model_s = load_qduig(PRMVT).to(DEV).eval()
    res = {"rules": "results/jaal_whole/approach_a/PREREGISTERED.md",
           "train_log": json.loads((OUT / "train_log.json").read_text(encoding="utf-8"))}
    pv, yv = probs_loader(model_a, DataLoader(SynthNotes([r for r in idx if r["split"] == "val"], bgs, 777000), batch_size=24, num_workers=2))
    tau = float(pv[yv == 0].max())
    res["tau_A"] = tau
    for name, model, t in (("A", model_a, tau), ("S4_deployed", model_s, TAU_S4)):
        pt, yt = probs_loader(model, DataLoader(SynthNotes([r for r in idx if r["split"] == "test"], bgs, 888000), batch_size=24, num_workers=2))
        res[f"synthetic_test_{name}"] = {
            "n": int(len(yt)), "accuracy_0.5": float(((pt >= 0.5) == (yt == 1)).mean()),
            "auc": float(roc_auc_score(yt, pt)),
            "counterfeit_passed": int((pt[yt == 0] > t).sum()), "counterfeit_n": int((yt == 0).sum()),
            "counterfeit_passed_wilson95": wilson(int((pt[yt == 0] > t).sum()), int((yt == 0).sum())),
            "genuine_confirmed": int((pt[yt == 1] > t).sum()), "genuine_n": int((yt == 1).sum())}
    crops = json.loads((SYN / "crops_index.json").read_text(encoding="utf-8"))
    real = {"A": {}, "S4_deployed": {}}
    for name, model, t in (("A", model_a, tau), ("S4_deployed", model_s, TAU_S4)):
        scores = []
        for r in crops:
            if r.get("crop") and r["yolo"] in ("500_taka", "1000_taka") and not r["augmented"]:
                scores.append((r["set"], r["truth"], r["group"], prob_crop(model, cv2.imread(str(SYN / "crops" / r["crop"])))))
        out = {}
        for s, truth in (("cf", "counterfeit"), ("cf", "genuine"), ("bm", "genuine"), ("bt", "genuine")):
            p = np.array([x[3] for x in scores if x[0] == s and x[1] == truth])
            out[f"{s}/{truth}"] = {"n": int(len(p)), "said_likely_genuine": int((p > t).sum()),
                                   "wilson95": wilson(int((p > t).sum()), int(len(p))),
                                   "called_counterfeit_at_0.5": int((p < 0.5).sum()),
                                   "max_p": float(p.max()) if len(p) else None}
        cfp = [(x[1] == "genuine", x[3]) for x in scores if x[0] == "cf"]
        out["auc_cf_originals"] = float(roc_auc_score([a for a, _ in cfp], [b for _, b in cfp]))
        real[name] = out
    res["real_whole_note"] = real
    a, s = real["A"], real["S4_deployed"]
    conf_a = a["bm/genuine"]["said_likely_genuine"] + a["bt/genuine"]["said_likely_genuine"]
    conf_s = s["bm/genuine"]["said_likely_genuine"] + s["bt/genuine"]["said_likely_genuine"]
    res["decision"] = {"A_passes_no_real_counterfeit": a["cf/counterfeit"]["said_likely_genuine"] == 0,
                       "A_confirms_more_independent_genuine": conf_a > conf_s,
                       "independent_genuine_confirmed": {"A": conf_a, "S4": conf_s}}
    res["decision"]["switch_app_to_A"] = res["decision"]["A_passes_no_real_counterfeit"] and res["decision"]["A_confirms_more_independent_genuine"]
    (OUT / "eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "train_log"}, indent=1))


if __name__ == "__main__":
    main()
