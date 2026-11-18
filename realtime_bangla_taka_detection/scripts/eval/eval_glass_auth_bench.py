"""Glass-style authenticity bench on locked BD_Auth test images (no webcam required).

Mirrors savior_glass currency policy:
  - decide REAL only if p(genuine) >= tau
  - otherwise CHECK_BY_HAND (never speak counterfeit)
  - report latency, coverage, false REAL, per-denom breakdown

Usage:
  python scripts/eval/eval_glass_auth_bench.py
Output:
  results/glass_auth_bench.json
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.bd_counterfeit_auth import BDCounterfeitAuth
from roboeye.config import BD_COUNTERFEIT_AUTH_WEIGHTS, DATA_SET, DEVICE

MANIFEST = DATA_SET / "BD_Auth_Combined" / "manifest.csv"
OUT = ROOT / "results" / "glass_auth_bench.json"


def main() -> int:
    rows = [r for r in csv.DictReader(MANIFEST.open(encoding="utf-8")) if r["split"] == "test"]
    clf = BDCounterfeitAuth(min_conf=0.0)
    assert clf.loaded

    taus = (0.80, 0.90, 0.95, 0.99)
    per_tau = {}
    # collect raw once
    items = []
    for r in rows:
        bgr = cv2.imread(r["path"])
        if bgr is None:
            continue
        t0 = time.perf_counter()
        out = clf.predict_bgr(bgr)
        ms = (time.perf_counter() - t0) * 1000.0
        items.append(
            {
                "y": int(r["label"]),
                "denom": r["denom"],
                "note_id": r["note_id"],
                "p_real": float(out["genuine_prob"]),
                "conf": float(out["confidence"]),
                "pred": 1 if out["model_class"] == "genuine" else 0,
                "ms": ms,
            }
        )

    for tau in taus:
        spoken = []
        for it in items:
            if it["p_real"] >= tau:
                spoken.append("REAL")
            else:
                spoken.append("CHECK_BY_HAND")
        y = np.array([it["y"] for it in items])
        speak = np.array(spoken)
        # false REAL = fake note announced REAL
        false_real = int(((y == 0) & (speak == "REAL")).sum())
        true_real_ok = int(((y == 1) & (speak == "REAL")).sum())
        check = int((speak == "CHECK_BY_HAND").sum())
        by_denom = {}
        for d in sorted({it["denom"] for it in items}):
            idx = [i for i, it in enumerate(items) if it["denom"] == d]
            yd = y[idx]
            sd = speak[idx]
            by_denom[d] = {
                "n": len(idx),
                "false_real": int(((yd == 0) & (sd == "REAL")).sum()),
                "genuine_confirmed": int(((yd == 1) & (sd == "REAL")).sum()),
                "check_by_hand": int((sd == "CHECK_BY_HAND").sum()),
            }
        per_tau[str(tau)] = {
            "n": len(items),
            "real_announced": int((speak == "REAL").sum()),
            "check_by_hand": check,
            "coverage_real_announcements": round(float((speak == "REAL").mean()), 4),
            "false_real_on_fakes": false_real,
            "false_real_rate": round(false_real / max(int((y == 0).sum()), 1), 4),
            "genuine_confirm_rate": round(true_real_ok / max(int((y == 1).sum()), 1), 4),
            "latency_ms_median": round(float(np.median([it["ms"] for it in items])), 2),
            "by_denom": by_denom,
        }

    report = {
        "weights": str(BD_COUNTERFEIT_AUTH_WEIGHTS),
        "device": str(DEVICE),
        "policy": "announce REAL only if p_real >= tau else CHECK_BY_HAND; never speak counterfeit",
        "per_tau": per_tau,
        "recommended_tau": 0.90,
        "paper_claim": (
            f"At tau=0.90: false_REAL={per_tau['0.9']['false_real_on_fakes']} "
            f"({per_tau['0.9']['false_real_rate']:.1%} of fakes), "
            f"genuine_confirm={per_tau['0.9']['genuine_confirm_rate']:.1%}, "
            f"median latency {per_tau['0.9']['latency_ms_median']:.1f} ms."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
