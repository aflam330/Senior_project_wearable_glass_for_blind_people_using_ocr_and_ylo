"""Decide flip averaging, seed ensembling and a rejection threshold on VALIDATION, then
report test. Rules were fixed before looking at any test number:

  1. flip averaging is used for a model only if it raises its occluded val accuracy;
  2. the seed ensemble (mean probability) is used only if its occluded val accuracy beats
     seed 42 alone;
  3. rejection: confidence = max(p, 1-p); the threshold is the largest value on a 0.50-0.99
     grid that still answers >= 95% of clean val notes. Rejected notes get "please
     reposition the note" instead of a verdict.

Every option's test result is written too (per-seed, ensemble, flip), so nothing is hidden.
Input: results/robustness/occ_probs/<name>.npz from occlusion_probs.py.
Output: results/robustness/occlusion_decision.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/robustness/occ_probs"


def acc(p, y):
    return float(((p >= 0.5).astype(int) == y).mean())


def main(names: list[str]) -> None:
    z = {n: dict(np.load(D / f"{n}.npz")) for n in names}
    yv, yt = z[names[0]]["val_y"], z[names[0]]["test_y"]
    out = {"models": names, "rules": __doc__.split("\n\n")[1].strip(), "per_model": {}}
    chosen = {}
    for n in names:
        a = z[n]
        use_flip = acc((a["val_occ"] + a["val_occ_flip"]) / 2, yv) > acc(a["val_occ"], yv)
        pick = (lambda s: (a[s] + a[s + "_flip"]) / 2) if use_flip else (lambda s: a[s])
        chosen[n] = {s: pick(s) for s in ("val_clean", "val_occ", "test_clean", "test_occ")}
        out["per_model"][n] = {
            "flip_averaging_used": bool(use_flip),
            "val": {"clean": acc(chosen[n]["val_clean"], yv), "occ55": acc(chosen[n]["val_occ"], yv)},
            "test": {"clean": acc(chosen[n]["test_clean"], yt), "occ55": acc(chosen[n]["test_occ"], yt)},
            "test_without_flip": {"clean": acc(a["test_clean"], yt), "occ55": acc(a["test_occ"], yt)},
        }
    tests = [out["per_model"][n]["test"]["occ55"] for n in names]
    out["test_occ55_mean_over_models"] = float(np.mean(tests))
    out["test_occ55_sd_over_models"] = float(np.std(tests, ddof=1)) if len(tests) > 1 else None

    final = chosen[names[0]]
    out["final_method"] = f"{names[0]} alone"
    if len(names) > 1:
        ens = {s: np.mean([chosen[n][s] for n in names], axis=0) for s in chosen[names[0]]}
        out["ensemble"] = {"val": {"clean": acc(ens["val_clean"], yv), "occ55": acc(ens["val_occ"], yv)},
                           "test": {"clean": acc(ens["test_clean"], yt), "occ55": acc(ens["test_occ"], yt)}}
        if out["ensemble"]["val"]["occ55"] > out["per_model"][names[0]]["val"]["occ55"]:
            final, out["final_method"] = ens, f"mean of {len(names)} seeds"

    conf_v = np.maximum(final["val_clean"], 1 - final["val_clean"])
    thr = 0.5
    for t in np.arange(0.50, 0.995, 0.01):
        if (conf_v >= t).mean() >= 0.95:
            thr = float(round(t, 2))
    rej = {"threshold_from_val": thr}
    for s, y in (("val_clean", yv), ("val_occ", yv), ("test_clean", yt), ("test_occ", yt)):
        p = final[s]
        keep = np.maximum(p, 1 - p) >= thr
        rej[s] = {"answered": float(keep.mean()),
                  "accuracy_on_answered": acc(p[keep], y[keep]) if keep.any() else None,
                  "wrong_verdicts_share_of_all_notes": float(((p >= 0.5).astype(int) != y)[keep].sum() / len(y))}
    out["rejection"] = rej
    out["final_test"] = {"clean": acc(final["test_clean"], yt), "occ55": acc(final["test_occ"], yt)}
    dst = ROOT / "results/robustness/occlusion_decision.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or ["occrobust_s42"])
