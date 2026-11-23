"""Choose the SARA Trust Index genuine threshold on VALIDATION data only, then score it once on the locked test split.

Data: data set/BD_Auth_Combined/manifest.csv (EfficientNet-B0 counterfeit checker, roboeye/bd_counterfeit_auth.py).
The test rows are scored with every candidate threshold for the report table, but the threshold is SELECTED from the
validation rows alone, by a rule fixed in advance (SELECTION_RULE below).

Reported for validation and test, for the full Trust policy (quality gates + p(genuine) >= tau + trust >= 80) and for
the raw classifier threshold alone:
  counterfeit false acceptance rate (FAR)  = counterfeits answered LIKELY_GENUINE / counterfeits
  genuine automatic acceptance rate (GAR)  = genuine answered LIKELY_GENUINE / genuine
  genuine false rejection rate (FRR)       = 1 - GAR   (these go to manual verification, nothing is called fake)
  manual verification coverage            = all notes sent to "check by hand" / all notes
  expected calibration error (ECE, 10 bins) of p(genuine), before and after temperature scaling fitted on validation
FAR upper bounds are one-sided 95% Clopper-Pearson bounds (0 errors in n gives 1 - 0.05**(1/n)).

Limit of this data (thesis audit): every genuine image comes from PaperCurrencyRaw and every counterfeit from
BD_Counterfeit, and the five views of a note are rotations of one photo. Separation here is partly source separation.

Output: results/trust_threshold/calibration.json, .csv and gar_vs_far.png
Run: python scripts/eval/calibrate_trust_threshold.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.bd_counterfeit_auth import BDCounterfeitAuth  # noqa: E402
from roboeye.config import DATA_SET  # noqa: E402
from roboeye.sara_trust import sara_trust  # noqa: E402

MANIFEST = DATA_SET / "BD_Auth_Combined" / "manifest.csv"
OUT = ROOT / "results" / "trust_threshold"
REPORT_TAUS = [0.50, 0.80, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999]
SELECTION_RULE = ("smallest tau above the highest p(genuine) of any VALIDATION counterfeit (so 0 validation counterfeits "
                  "pass), never below the deployed 0.95; ties broken toward the larger tau")
DENOM_CONF = 0.85     # as in eval_sara_trust.py: the denomination detector is not part of this test


def cp_upper(k: int, n: int, a: float = 0.05) -> float:
    return 1.0 if n == 0 else (1.0 if k >= n else float(beta.ppf(1 - a, k + 1, n - k)))


def score_rows(clf, rows):
    out = []
    for r in rows:
        bgr = cv2.imread(r["path"])
        if bgr is None:
            continue
        p = float(clf.predict_bgr(bgr)["genuine_prob"])
        out.append({**{k: r[k] for k in ("note_id", "denom", "label", "split", "view", "source")}, "p": p, "bgr": bgr})
    return out


def policy(rows, tau, use_trust=True):
    fa = ga = n_f = n_g = manual = 0
    for r in rows:
        if use_trust:
            t = sara_trust(denom_name=f"{r['denom']}_taka", denom_conf=DENOM_CONF, p_real=r["p"], auth_min_conf=tau,
                           crop_bgr=r["bgr"])
            accept = t.verdict == "LIKELY_GENUINE"
        else:
            accept = r["p"] >= tau
        manual += int(not accept)
        if r["label"] == "0":
            n_f += 1
            fa += int(accept)
        else:
            n_g += 1
            ga += int(accept)
    return {"tau": tau, "counterfeits": n_f, "genuine": n_g, "counterfeit_accepted": fa,
            "FAR": fa / max(n_f, 1), "FAR_upper95": cp_upper(fa, n_f), "genuine_accepted": ga, "GAR": ga / max(n_g, 1),
            "FRR": 1 - ga / max(n_g, 1), "manual_coverage": manual / max(len(rows), 1)}


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (p >= lo) & (p < lo + 1 / bins) if lo < 1 - 1 / bins else (p >= lo)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def temp_scale(p, T):
    z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
    return 1 / (1 + np.exp(-z / T))


def fit_T(p, y):
    best = (1e9, 1.0)
    for T in np.exp(np.linspace(np.log(0.25), np.log(20), 400)):
        q = np.clip(temp_scale(p, T), 1e-7, 1 - 1e-7)
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        best = min(best, (nll, float(T)))
    return best[1]


def main() -> int:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    clf = BDCounterfeitAuth(min_conf=0.0)
    assert clf.loaded, "counterfeit checker weights not found"
    val = score_rows(clf, [r for r in rows if r["split"] == "val"])
    test = score_rows(clf, [r for r in rows if r["split"] == "test"])
    # note-level leakage check: no note in two splits
    notes = {s: {r["note_id"] for r in rows if r["split"] == s} for s in ("train", "val", "test")}
    leak = {f"{a}&{b}": len(notes[a] & notes[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}

    # ---- selection on validation only
    max_fake_val = max(r["p"] for r in val if r["label"] == "0")
    cand = sorted({0.95, *[t for t in np.round(np.linspace(0.95, 0.99999, 2000), 6)]})
    tau = next(t for t in cand if t > max_fake_val)

    table = {"val": {}, "test": {}}
    for name, rs in (("val", val), ("test", test)):
        for t in sorted(set(REPORT_TAUS + [tau])):
            table[name][f"{t:.6g}"] = {"trust_policy": policy(rs, t), "classifier_only": policy(rs, t, use_trust=False)}

    yv = np.array([r["label"] == "1" for r in val], float)
    pv = np.array([r["p"] for r in val])
    yt = np.array([r["label"] == "1" for r in test], float)
    pt = np.array([r["p"] for r in test])
    T = fit_T(pv, yv)
    calib = {"temperature_fitted_on_val": T, "ECE_val_raw": ece(pv, yv), "ECE_test_raw": ece(pt, yt),
             "ECE_val_scaled": ece(temp_scale(pv, T), yv), "ECE_test_scaled": ece(temp_scale(pt, T), yt),
             "note": "temperature scaling keeps the ranking, so FAR/GAR at a matched threshold do not change"}

    # GAR vs FAR curves (classifier score), validation and test
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=150)
    for name, p, y, c in (("validation", pv, yv, "#1F5F8B"), ("test", pt, yt, "#D7263D")):
        ts = np.unique(np.concatenate([[0, 1.0000001], p]))
        far = [((p >= t) & (y == 0)).sum() / max((y == 0).sum(), 1) for t in ts]
        gar = [((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1) for t in ts]
        ax.plot(far, gar, color=c, label=name)
        sel = policy(val if name == "validation" else test, tau, use_trust=False)
        ax.scatter([sel["FAR"]], [sel["GAR"]], color=c, zorder=3)
    ax.set_xlabel("Counterfeit false acceptance rate")
    ax.set_ylabel("Genuine automatic acceptance rate")
    ax.set_title(f"Counterfeit checker, BD_Auth (dot: selected tau = {tau:.4f})")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.grid(alpha=0.3)
    ax.legend()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / "gar_vs_far.png")

    res = {"data": str(MANIFEST), "n_val": len(val), "n_test": len(test), "note_overlap_between_splits": leak,
           "selection_rule": SELECTION_RULE, "max_validation_counterfeit_p": max_fake_val, "selected_tau": tau,
           "deployed_tau": 0.95, "selected_on_test": {"trust_policy": policy(test, tau), "classifier_only": policy(test, tau, use_trust=False)},
           "deployed_on_test": {"trust_policy": policy(test, 0.95), "classifier_only": policy(test, 0.95, use_trust=False)},
           "calibration": calib, "table": table,
           "data_limit": "genuine and counterfeit come from different sources; views are rotations of one photo"}
    (OUT / "calibration.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    with (OUT / "calibration.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "policy", "tau", "FAR", "FAR_upper95", "GAR", "FRR", "manual_coverage", "counterfeit_accepted",
                    "counterfeits", "genuine_accepted", "genuine"])
        for s in table:
            for t, d in table[s].items():
                for pol, m in d.items():
                    w.writerow([s, pol, t, round(m["FAR"], 4), round(m["FAR_upper95"], 4), round(m["GAR"], 4),
                                round(m["FRR"], 4), round(m["manual_coverage"], 4), m["counterfeit_accepted"],
                                m["counterfeits"], m["genuine_accepted"], m["genuine"]])
    print(json.dumps({k: res[k] for k in ("n_val", "n_test", "note_overlap_between_splits", "max_validation_counterfeit_p",
                                          "selected_tau", "selected_on_test", "deployed_on_test", "calibration")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
