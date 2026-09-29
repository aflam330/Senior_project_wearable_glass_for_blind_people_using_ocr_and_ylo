"""MVP-N reference: how much one view can tell, independent of any fusion head.

A logistic regression on single TRAIN views (frozen ResNet-50 features, cached by mvpn_vcds.py),
C chosen on VALID at k = 1. It is scored on TEST at k = 1 (first view) and, by averaging class
probabilities, at k = 2..6. A per-view scorer cannot suffer view-count shift, so its k = 1 accuracy
estimates the information available in one view.
Output: results/mvpn/reference.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from mvpn_vcds import CLASSES, DATA, K, OUT, eval_sets  # noqa: E402


def main() -> None:
    d = np.load(OUT / "feats_resnet50.npz", allow_pickle=True)
    feats = dict(zip(d["keys"], d["x"]))
    tr = [(k, CLASSES.index(k.split("/")[1])) for k in feats if k.startswith("train/")]
    xtr = np.stack([feats[k] for k, _ in tr])
    ytr = np.array([y for _, y in tr])
    val, test = eval_sets("valid"), eval_sets("test")

    def acc(clf, sets, k):
        sub = [(v, y) for v, y in sets if len(v) >= k]
        p = np.mean([clf.predict_proba(np.stack([feats[v[j]] for v, _ in sub])) for j in range(k)], axis=0)
        return float((p.argmax(1) == np.array([y for _, y in sub])).mean())

    best = None
    for c in (0.1, 1.0, 10.0, 100.0):
        clf = LogisticRegression(C=c, max_iter=3000).fit(xtr, ytr)
        a = acc(clf, val, 1)
        if best is None or a > best[0]:
            best = (a, c, clf)
    _, c, clf = best
    out = {"C_chosen_on_valid_k1": c, "valid_k1": best[0],
           "test": {str(k): acc(clf, test, k) for k in range(1, K + 1)}}
    (OUT / "reference.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
