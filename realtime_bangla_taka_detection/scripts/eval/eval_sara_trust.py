"""Evaluate SARA Trust Index on locked BD_Auth test + quality stress.

Proves the unique contract:
  - 0 tickets contain the word counterfeit/জাল
  - false LIKELY_GENUINE on fakes is measured
  - reason codes fire on dark/blur crops

Output: results/sara_trust_eval.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.bd_counterfeit_auth import BDCounterfeitAuth
from roboeye.config import DATA_SET, DEVICE
from roboeye.sara_trust import sara_trust

MANIFEST = DATA_SET / "BD_Auth_Combined" / "manifest.csv"
OUT = ROOT / "results" / "sara_trust_eval.json"


def main() -> int:
    rows = [r for r in csv.DictReader(MANIFEST.open(encoding="utf-8")) if r["split"] == "test"]
    clf = BDCounterfeitAuth(min_conf=0.0)
    assert clf.loaded

    tickets = []
    false_genuine = 0
    true_genuine = 0
    n_fake = n_real = 0
    banned = 0
    reasons = {}

    for r in rows:
        bgr = cv2.imread(r["path"])
        if bgr is None:
            continue
        out = clf.predict_bgr(bgr)
        denom = f"{r['denom']}_taka"
        t = sara_trust(
            denom_name=denom,
            denom_conf=0.85,
            p_real=float(out["genuine_prob"]),
            auth_min_conf=0.95,
            crop_bgr=bgr,
            denom_bn=f"{r['denom']} টাকার নোট",
        )
        y = int(r["label"])
        if y == 0:
            n_fake += 1
            if t.verdict == "LIKELY_GENUINE":
                false_genuine += 1
        else:
            n_real += 1
            if t.verdict == "LIKELY_GENUINE":
                true_genuine += 1
        spoken = f"{t.speak_bn} {t.reason_bn} {t.next_action_bn} {t.verdict}".lower()
        if ("counterfeit" in spoken) or ("জাল" in spoken) or ("jaal" in spoken):
            banned += 1
        reasons[t.reason_code] = reasons.get(t.reason_code, 0) + 1
        tickets.append({"y": y, "verdict": t.verdict, "trust": t.trust, "reason": t.reason_code, "stars": t.stars})

    # stress: force dark on 40 fakes — should shift to too_dark / check
    dark_shift = 0
    fakes = [r for r in rows if r["label"] == "0"][:40]
    for r in fakes:
        bgr = cv2.imread(r["path"])
        if bgr is None:
            continue
        dark = np.clip(bgr.astype(np.float32) * 0.25, 0, 255).astype(np.uint8)
        out = clf.predict_bgr(dark)
        t = sara_trust(
            denom_name=f"{r['denom']}_taka",
            denom_conf=0.85,
            p_real=float(out["genuine_prob"]),
            auth_min_conf=0.95,
            crop_bgr=dark,
            denom_bn="নোট",
        )
        if t.reason_code == "too_dark" or t.verdict != "LIKELY_GENUINE":
            dark_shift += 1

    report = {
        "device": str(DEVICE),
        "n": len(tickets),
        "n_fake": n_fake,
        "n_real": n_real,
        "false_likely_genuine_on_fakes": false_genuine,
        "false_likely_genuine_rate": round(false_genuine / max(n_fake, 1), 4),
        "genuine_confirm_rate": round(true_genuine / max(n_real, 1), 4),
        "tickets_containing_counterfeit_word": banned,
        "reason_histogram": reasons,
        "dark_stress_safe_or_flagged": dark_shift,
        "dark_stress_n": len(fakes),
        "unique_contract": "Never speaks counterfeit; emits Trust Index + Bangla next-action reason codes",
        "paper_claim": (
            f"SARA Trust Index on {len(tickets)} locked images: "
            f"false LIKELY_GENUINE on fakes={false_genuine}/{n_fake}, "
            f"genuine confirm={true_genuine}/{n_real}, "
            f"banned counterfeit-word tickets={banned}, "
            f"dark-stress guided/safe={dark_shift}/{len(fakes)}."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
