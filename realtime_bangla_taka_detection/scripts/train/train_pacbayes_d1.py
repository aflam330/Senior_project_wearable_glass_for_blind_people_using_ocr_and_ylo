"""Train a Q-DUIG model on one half of the training notes.

D2 is the other half. It is saved and not read. The validation split is used
only to pick the checkpoint. The test split is not read.
The split matches compute_pacbayes_head.py: permutation seed 42 of the
training-note order.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.camva.notes import load_splits
from roboeye.qduig.artifacts import init_run
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import build_model, set_seed, train_qduig

OUT = ROOT / "results" / "theory" / "pacbayes_d1_model" / "seed42"
CFG = ROOT / "configs" / "ablation_prefix" / "her_base.yaml"


def split_ids(train_ids: list[str]) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(42)
    order = rng.permutation(len(train_ids))
    mid = len(train_ids) // 2
    d1 = [train_ids[int(i)] for i in order[:mid]]
    d2 = [train_ids[int(i)] for i in order[mid:]]
    return d1, d2


def main() -> None:
    splits, records = load_splits()
    train_ids = list(splits["train"])
    d1, d2 = split_ids(train_ids)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split_ids.json").write_text(json.dumps({
        "seed": 42,
        "n_train": len(train_ids),
        "d1": d1,
        "d2": d2,
        "note": "D2 is held out of this training run. Test ids are not stored.",
    }, indent=2), encoding="utf-8")
    cfg = load_config(CFG)
    init_run(OUT, cfg, 42, "PAC-Bayes D1 model", "Trains on D1 only. D2 and test are not read.")
    set_seed(42)
    model = build_model(cfg)
    summary = train_qduig(
        model,
        d1,
        list(splits["val"]),
        records,
        config=cfg,
        seed=42,
        output_dir=OUT,
    )
    print(json.dumps({"n_d1": len(d1), "n_d2": len(d2), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
