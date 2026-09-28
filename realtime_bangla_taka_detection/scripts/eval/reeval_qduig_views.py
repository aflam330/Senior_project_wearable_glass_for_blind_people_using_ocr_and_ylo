"""Re-evaluate every Q-DUIG checkpoint at 1-6 views after the 2026-09-28 fixes.

Fixes that change Q-DUIG outputs:
  * binary_entropy_t used eps=1e-8, which rounds away in float32; a fully confident
    note (p == 1) gave NaN entropy, NaN HGEF logits, and forward() replaced them with
    p = 0.5 ("genuine"). 23 of 208 test notes were affected for prefix_ft seed 42.
  * logdet_gram gave masked-out views log(tau) each, so a prefix inside padded slots
    got a different volume than the same views alone (affects padded/masked batches
    and the sequential policies, not plain k-view test loaders).

Old results in <run>/test_views are left untouched. New results go to
<run>/test_views_20260928/<k>view, and a side-by-side summary to
results/qduig/reeval_20260928.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.camva.engine import make_loader
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.artifacts import save_json
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig, pack_and_save, predict_qduig

QDUIG = ROOT / "results" / "qduig"
OUT_NAME = "test_views_20260928"


def stored_acc(run: Path, k: int) -> float | None:
    p = run / "test_views" / f"{k}view" / "test_metrics.json"
    return json.loads(p.read_text(encoding="utf-8"))["accuracy"] if p.is_file() else None


def main() -> None:
    splits, records = load_splits()
    loaders = {k: make_loader(splits["test"], records, n_views=k, train=False, batch=8) for k in range(1, 7)}
    summary = {}
    for ckpt in sorted(QDUIG.rglob("checkpoint.pt")):
        run = ckpt.parent
        name = str(run.relative_to(QDUIG)).replace("\\", "/")
        if not (run / "config.yaml").is_file():
            summary[name] = {"status": "skipped: no config.yaml"}
            continue
        try:
            model = load_qduig(ckpt, load_config(run / "config.yaml"))
        except Exception as exc:  # noqa: BLE001
            summary[name] = {"status": f"load failed: {type(exc).__name__}: {exc}"[:300]}
            print(name, summary[name]["status"], flush=True)
            continue
        row = {"status": "ok", "new": {}, "stored": {}}
        for k, loader in loaders.items():
            m = pack_and_save(predict_qduig(model, loader, DEVICE), f"reeval_{k}view", run / OUT_NAME / f"{k}view")
            row["new"][k] = m["accuracy"]
            row["stored"][k] = stored_acc(run, k)
        save_json(run / OUT_NAME / "views_1_to_6.json", row)
        summary[name] = row
        fmt = lambda d: " ".join("-" if d[k] is None else f"{d[k]:.3f}" for k in range(1, 7))
        print(f"{name:45s} new {fmt(row['new'])} | stored {fmt(row['stored'])}", flush=True)
        save_json(QDUIG / "reeval_20260928.json", summary)
    save_json(QDUIG / "reeval_20260928.json", summary)


if __name__ == "__main__":
    main()
