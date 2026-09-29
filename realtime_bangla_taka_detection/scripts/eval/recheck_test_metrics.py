"""Recompute accuracy from saved per-note predictions for every test_metrics.json that has them.

For each test_metrics.json with a sibling test_predictions.json holding y_true and genuine_score,
accuracy is recomputed at threshold 0.5 and compared with the stored value. Non-finite scores are
counted as wrong answers only if the stored run did so; a mismatch is reported either way.
Output: results/scan/test_metrics_recheck.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SKIP = {"venv", "cache", "runs", "__pycache__"}


def main() -> None:
    rows, n_metrics = [], 0
    for m in ROOT.rglob("test_metrics.json"):
        if SKIP & set(m.parts):
            continue
        n_metrics += 1
        p = m.with_name("test_predictions.json")
        if not p.is_file():
            continue
        try:
            pred = json.loads(p.read_text(encoding="utf-8"))
            met = json.loads(m.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            rows.append({"file": str(m.relative_to(ROOT)), "status": "unreadable"})
            continue
        if not (isinstance(pred, dict) and "y_true" in pred and "genuine_score" in pred and "accuracy" in met):
            continue
        y = np.asarray(pred["y_true"]) == 1
        s = np.asarray(pred["genuine_score"], dtype=float)
        acc = float(((s >= 0.5) == y).mean())
        nonfinite = int((~np.isfinite(s)).sum())
        diff = abs(acc - float(met["accuracy"]))
        rows.append({"file": str(m.relative_to(ROOT)), "stored": met["accuracy"], "recomputed": acc,
                     "nonfinite_scores": nonfinite, "status": "ok" if diff <= 1e-9 else "MISMATCH"})
    bad = [r for r in rows if r["status"] != "ok"]
    out = {"test_metrics_files": n_metrics, "with_predictions": len(rows), "match": len(rows) - len(bad),
           "mismatch": len(bad), "mismatches": bad}
    dst = ROOT / "results" / "scan" / "test_metrics_recheck.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print({k: v for k, v in out.items() if k != "mismatches"})
    for r in bad[:15]:
        print(r)


if __name__ == "__main__":
    main()
