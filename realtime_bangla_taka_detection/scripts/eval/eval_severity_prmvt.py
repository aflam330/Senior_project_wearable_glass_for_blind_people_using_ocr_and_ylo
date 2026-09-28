"""Recompute only the PRMVT column of the severity sweep after the 2026-09-28 Q-DUIG fixes.

The baseline and NDAL models do not use Q-DUIG code, so their columns in the existing
results/robustness/severity_seed42.json cannot change and are kept. PRMVT is re-run with
the same corruptions, severities, seeds and chunking as eval_severity_curves.py (whose
helpers are imported), and its accuracy and drop columns are replaced.

Usage:
  python scripts/eval/eval_severity_prmvt.py --previous <old severity_seed42.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.camva.data import apply_corruption
from roboeye.camva.notes import load_splits
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig
from scripts.eval.eval_severity_curves import CHUNK, SWEEP, _acc, _forward, _read, _views_from_bgr

OUT = ROOT / "results" / "robustness" / "severity_seed42.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--previous", default=str(OUT), help="severity file whose baseline/NDAL columns are kept")
    args = p.parse_args()
    previous = json.loads(Path(args.previous).read_text(encoding="utf-8"))

    splits, records = load_splits()
    test = list(splits["test"])
    labels = np.array([int(records[n]["label"]) for n in test], dtype=np.int64)
    prmvt = load_qduig(ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt",
                       load_config(ROOT / "configs" / "proposed_prefix_ft.yaml"))
    keys = [("clean", 0.0)] + [(name, float(sev)) for name, levels in SWEEP.items() for sev in levels]
    store = {key: [] for key in keys}
    for start in range(0, len(test), CHUNK):
        ids = test[start:start + CHUNK]
        raw = [_read(records, nid) for nid in ids]
        store[("clean", 0.0)].append(_forward(prmvt, torch.stack([_views_from_bgr(i) for i in raw]), "other"))
        for corr, levels in SWEEP.items():
            for severity in levels:
                batch = []
                for i, imgs in enumerate(raw):
                    if corr == "occlusion":
                        np.random.seed(1000 + start + i)  # same occluder boxes as eval_severity_curves.py
                    batch.append(_views_from_bgr([apply_corruption(bgr, corr, severity) for bgr in imgs]))
                store[(corr, float(severity))].append(_forward(prmvt, torch.stack(batch), "other"))
        print(f"notes {start + len(ids)}/{len(test)}", flush=True)

    clean = _acc(labels, np.concatenate(store[("clean", 0.0)]))
    old_rows = {(r["corruption"], float(r["severity"])): r for r in previous["rows"]}
    rows = []
    for corr, levels in SWEEP.items():
        for severity in levels:
            entry = dict(old_rows[(corr, float(severity))])
            entry["prmvt"] = _acc(labels, np.concatenate(store[(corr, float(severity))]))
            entry["prmvt_drop"] = clean - entry["prmvt"]
            rows.append(entry)
            print(corr, severity, "prmvt", round(entry["prmvt"], 4), flush=True)
    clean_all = dict(previous["clean_6view"])
    clean_all["prmvt"] = clean
    OUT.write_text(json.dumps({
        "clean_6view": clean_all,
        "rows": rows,
        "note": "prmvt re-evaluated 2026-09-28 after the Q-DUIG NaN-entropy fix "
                "(scripts/eval/eval_severity_prmvt.py); baseline and ndal columns unchanged.",
    }, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
