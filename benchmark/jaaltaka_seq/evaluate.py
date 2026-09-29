"""Score a JaalTaka-Seq submission.

Submission JSON:
{
  "name": "my-policy", "seed": 42, "description": "...",
  "predictions": [
    {"note_id": "genuine:note_001", "genuine_prob": 0.97, "views_used": [0, 1]},
    {"note_id": "counterfeit:note_021", "genuine_prob": null, "views_used": [0]}   # null = abstain
  ]
}
Rules: one entry per test note in split.json; views_used lists the view indices (0-5) the policy
looked at, in acquisition order; genuine_prob in [0, 1] or null to abstain. Labels come from the
note_id prefix. Nothing about the test split may be used for training or model selection.

Metrics: accuracy over all notes (abstain counts as not correct), coverage, selective accuracy,
wrong-verdict rate (answered and wrong, over all notes), mean views, cost-adjusted accuracy
(accuracy - 0.02 * mean views; 0.02 is the validation-chosen cost used in the project), and
ECE (10 bins) on answered notes.

Usage: python evaluate.py submission.json [--split split.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
COST = 0.02


def load_split(path: Path) -> list[str]:
    return list(json.loads(path.read_text(encoding="utf-8"))["test"])


def label(note_id: str) -> int:
    kind = note_id.split(":", 1)[0]
    if kind not in {"genuine", "counterfeit"}:
        raise ValueError(f"bad note_id {note_id}")
    return 1 if kind == "genuine" else 0


def ece(probs: list[float], ys: list[int], bins: int = 10) -> float | None:
    if not probs:
        return None
    total, err = len(probs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (max(p, 1 - p) >= lo and (max(p, 1 - p) < hi or b == bins - 1))]
        if not idx:
            continue
        conf = sum(max(probs[i], 1 - probs[i]) for i in idx) / len(idx)
        acc = sum((probs[i] >= 0.5) == (ys[i] == 1) for i in idx) / len(idx)
        err += len(idx) / total * abs(conf - acc)
    return err


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [math.nan, math.nan]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def score(sub: dict, test_ids: list[str]) -> dict:
    preds = {p["note_id"]: p for p in sub["predictions"]}
    missing = sorted(set(test_ids) - set(preds))
    extra = sorted(set(preds) - set(test_ids))
    if missing or extra:
        raise ValueError(f"submission must cover exactly the {len(test_ids)} test notes "
                         f"(missing {len(missing)}, extra {len(extra)})")
    n = len(test_ids)
    correct = wrong = answered = 0
    views, probs, ys = [], [], []
    for nid in test_ids:
        p = preds[nid]
        used = p.get("views_used") or []
        if len(set(used)) != len(used) or any(not (0 <= int(v) <= 5) for v in used) or not used:
            raise ValueError(f"{nid}: views_used must be 1-6 distinct indices in 0..5")
        views.append(len(used))
        y = label(nid)
        g = p.get("genuine_prob")
        if g is None:
            continue
        g = float(g)
        if not 0.0 <= g <= 1.0:
            raise ValueError(f"{nid}: genuine_prob outside [0, 1]")
        answered += 1
        probs.append(g)
        ys.append(y)
        if (g >= 0.5) == (y == 1):
            correct += 1
        else:
            wrong += 1
    mean_views = sum(views) / n
    return {
        "name": sub.get("name"), "seed": sub.get("seed"), "n": n,
        "accuracy": correct / n, "accuracy_wilson95": wilson(correct, n),
        "coverage": answered / n,
        "selective_accuracy": correct / answered if answered else None,
        "wrong_verdict_rate": wrong / n,
        "mean_views": mean_views,
        "cost_adjusted_accuracy": correct / n - COST * mean_views,
        "ece": ece(probs, ys),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--split", default=str(HERE / "split.json"))
    args = ap.parse_args()
    sub = json.loads(Path(args.submission).read_text(encoding="utf-8"))
    print(json.dumps(score(sub, load_split(Path(args.split))), indent=2))


if __name__ == "__main__":
    main()
