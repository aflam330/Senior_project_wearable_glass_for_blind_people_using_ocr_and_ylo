"""Aggregate seed 42/43/44 metrics. Never invent missing seeds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from roboeye.qduig.artifacts import QDUIG_ROOT, save_json


def _load(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarise(values: list[float]) -> dict:
    arr = np.array(values, dtype=np.float64)
    return {
        "n_seeds": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": [float(x) for x in arr],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(QDUIG_ROOT / "aggregate"))
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    keys = ["accuracy", "macro_f1", "balanced_accuracy", "roc_auc", "ece", "brier", "average_views"]
    collected: dict[str, dict[str, list]] = {}
    missing = []
    for seed in (42, 43, 44):
        mpath = QDUIG_ROOT / "eval" / f"seed{seed}" / "proposed_6view" / "test_metrics.json"
        m = _load(mpath)
        if m is None:
            missing.append(str(mpath))
            continue
        for k in keys:
            if k in m and m[k] is not None:
                collected.setdefault("proposed_6view", {}).setdefault(k, []).append(m[k])
        bpath = QDUIG_ROOT / "eval" / f"seed{seed}" / "baseline" / "baseline_6view" / "test_metrics.json"
        b = _load(bpath)
        if b:
            for k in keys:
                if k in b and b[k] is not None:
                    collected.setdefault("baseline_6view", {}).setdefault(k, []).append(b[k])

    summary = {name: {k: summarise(vs) for k, vs in metrics.items()} for name, metrics in collected.items()}
    summary["missing_artifacts"] = missing
    save_json(out / "seed_aggregate.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
