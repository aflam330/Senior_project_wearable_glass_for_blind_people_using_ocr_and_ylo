"""Every JaalTaka note tested once: 5-fold cross-validation of the frozen ResNet-50 probe.

Two fold designs over all 1,390 notes:
  note     stratified random folds of physical notes (seed 42)
  serial   counterfeit notes grouped by OCR serial prefix (6 digits; unread = own group) and whole
           groups kept in one fold; genuine notes (unique serials) spread at random
Per fold: logistic regression on single TRAIN views (the other 4 folds), C = 100 (the value chosen on the
standard split's validation set; no fold's test notes were used to choose it); k views = mean probability.
Pooled accuracy over all notes at k = 1..6, and per-fold spread.
Output: results/sota/cv_probe.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from roboeye.camva.notes import SPLIT_DIR, load_splits  # noqa: E402


def main() -> None:
    _, records = load_splits(SPLIT_DIR)
    f = np.load(ROOT / "results/sota/feats_resnet50.npz", allow_pickle=True)
    ids = list(f["ids"])
    X = f["x"]
    y = np.array([int(records[n]["label"]) for n in ids])
    audit = {r["note_id"]: r for r in json.loads((ROOT / "results/jaal_whole/serial_audit.json").read_text(encoding="utf-8"))["rows"]}
    groups = []
    rng = np.random.default_rng(42)
    for i, n in enumerate(ids):
        s = (audit.get(n) or {}).get("serial")
        groups.append(f"cf:{s[:6]}" if y[i] == 0 and s else f"note:{n}:{rng.integers(1_000_000)}")
    designs = {"note": list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)),
               "serial": list(GroupKFold(5).split(X, y, groups))}
    out = {}
    for name, folds in designs.items():
        pred = np.zeros((len(ids), 6))
        per_fold = []
        for tr, te in folds:
            clf = LogisticRegression(C=100.0, max_iter=3000).fit(X[tr].reshape(-1, X.shape[2]), np.repeat(y[tr], 6))
            p = clf.predict_proba(X[te].reshape(-1, X.shape[2]))[:, 1].reshape(len(te), 6)
            pred[te] = p
            per_fold.append({k: float(((p[:, :k].mean(1) >= 0.5) == (y[te] == 1)).mean()) for k in (1, 6)})
        pooled = {k: float(((pred[:, :k].mean(1) >= 0.5) == (y == 1)).mean()) for k in range(1, 7)}
        out[name] = {"pooled_accuracy_all_1390_notes": pooled,
                     "fold_k1": [r[1] for r in per_fold], "fold_k6": [r[6] for r in per_fold],
                     "fold_counterfeit_counts": [int((y[te] == 0).sum()) for _, te in folds]}
        print(name, {k: round(v, 4) for k, v in pooled.items()}, "fold k1", [round(r[1], 3) for r in per_fold])
    (ROOT / "results/sota/cv_probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
