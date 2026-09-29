"""Image-quality gate in front of confidence rejection (PRMVT, seed 42).

Why: results/safety/rejection_seed42.json shows that confidence rejection alone still gives many
confident wrong verdicts under low light and over-exposure. A corrupted image can look confident.

Rule, fixed on CLEAN VALIDATION views before any test number was read:
  per view: mean luma Y, saturated share (Y >= 250), near-black share (Y <= 5)
  thresholds: dark if mean Y < 1st percentile of clean VAL views
              blown if saturated share > 99th percentile of clean VAL views
              blocked if near-black share > 99th percentile of clean VAL views
  a note is answered only if every view it uses passes the gate AND
  max(p, 1-p) >= t, with t chosen as in eval_safety_rejection.py (largest t on 0.50..0.99 that
  answers >= 95 % of clean VAL notes, gate applied).
  Otherwise the glass asks for better light / to move the finger instead of answering.
Corruptions: same protocol as eval_safety_rejection.py (np.random.seed(1000 + i) per note).
Caveat: the synthetic occlusion is a black box, which the near-black test finds easily; a real
finger is not black, so the occlusion row is optimistic.

Output: results/safety/quality_gate_seed42.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from roboeye.camva.data import apply_corruption  # noqa: E402
from roboeye.camva.notes import load_splits  # noqa: E402
from roboeye.config import DEVICE  # noqa: E402
from roboeye.qduig.config_io import load_config  # noqa: E402
from roboeye.qduig.engine import load_qduig  # noqa: E402
from eval_safety_rejection import CONDITIONS, RUN, TF, wilson  # noqa: E402

OUT = ROOT / "results" / "safety" / "quality_gate_seed42.json"


def stats(bgr: np.ndarray) -> tuple[float, float, float]:
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(y.mean()), float((y >= 250).mean()), float((y <= 5).mean())


@torch.inference_mode()
def run(model, records, ids, corr, sev, k):
    probs, st = [], []
    for i, nid in enumerate(ids):
        if corr:
            np.random.seed(1000 + i)
        views, s = [], []
        for p in records[nid]["view_paths"][:k]:
            bgr = cv2.imread(str(p))
            if corr:
                bgr = apply_corruption(bgr, corr, sev)
            s.append(stats(bgr))
            views.append(TF(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
        v = torch.stack(views).unsqueeze(0).to(DEVICE)
        m = torch.ones(1, v.size(1), dtype=torch.long, device=DEVICE)
        probs.append(float(torch.softmax(model(v, m)["logits"], 1)[0, 1]))
        st.append(s)
    return np.array(probs), st


def gate_ok(st, thr) -> np.ndarray:
    return np.array([all(m >= thr["dark_mean_y"] and sat <= thr["blown_sat"] and blk <= thr["blocked_black"]
                         for m, sat, blk in note) for note in st])


def main() -> None:
    splits, records = load_splits()
    model = load_qduig(RUN / "checkpoint.pt", load_config(RUN / "config.yaml")).to(DEVICE).eval()
    val, test = list(splits["val"]), list(splits["test"])
    yv = np.array([int(records[n]["label"]) for n in val])
    yt = np.array([int(records[n]["label"]) for n in test])
    out = {"model": str(RUN.relative_to(ROOT)), "rule": __doc__.split("Corruptions")[0].strip(), "views": {}}
    for k in (1, 6):
        pv, sv = run(model, records, val, None, None, k)
        flat = np.array([x for note in sv for x in note])
        thr = {"dark_mean_y": float(np.quantile(flat[:, 0], 0.01)),
               "blown_sat": float(np.quantile(flat[:, 1], 0.99)),
               "blocked_black": float(np.quantile(flat[:, 2], 0.99))}
        gv = gate_ok(sv, thr)
        conf_v = np.maximum(pv, 1 - pv)
        grid = [round(0.50 + 0.01 * i, 2) for i in range(50)]
        ok = [t for t in grid if ((conf_v >= t) & gv).mean() >= 0.95]
        t = max(ok) if ok else 0.5
        res = {"gate_thresholds_from_clean_val": thr, "confidence_threshold": t,
               "clean_val_answered": float(((conf_v >= t) & gv).mean())}
        for name, corr, sev in CONDITIONS:
            pt, stt = run(model, records, test, corr, sev, k)
            right = (pt >= 0.5) == (yt == 1)
            conf_ans = np.maximum(pt, 1 - pt) >= t
            g = gate_ok(stt, thr)
            for label, ans in (("confidence_only", conf_ans), ("gate_and_confidence", conf_ans & g)):
                wrong = int((ans & ~right).sum())
                res.setdefault(name, {})[label] = {
                    "answered": int(ans.sum()), "coverage": float(ans.mean()), "wrong_verdicts": wrong,
                    "wrong_verdict_rate": wrong / len(yt), "wrong_verdict_wilson95": wilson(wrong, len(yt)),
                    "selective_accuracy": float(right[ans].mean()) if ans.any() else None}
            res[name]["gate_rejected_notes"] = int((~g).sum())
            r = res[name]
            print(f"k={k} {name}: conf-only wrong {r['confidence_only']['wrong_verdicts']} ans {r['confidence_only']['answered']} | "
                  f"gate+conf wrong {r['gate_and_confidence']['wrong_verdicts']} ans {r['gate_and_confidence']['answered']}", flush=True)
        out["views"][str(k)] = res
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
