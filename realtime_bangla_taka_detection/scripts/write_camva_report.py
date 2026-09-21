"""Assemble CAMVA final_report.* only from saved artifacts. Never invent numbers."""
from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboeye.camva.notes import CAMVA_ROOT  # noqa: E402

NOT = "NOT MEASURED"


def _load(path: Path):
    if not path.is_file():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def main() -> None:
    split_meta = _load(CAMVA_ROOT / "splits" / "split_metadata.json")
    views = _load(CAMVA_ROOT / "metrics" / "views_1_to_6.json")
    ablations = _load(CAMVA_ROOT / "metrics" / "ablations.json")
    adaptive = _load(CAMVA_ROOT / "metrics" / "adaptive.json")
    robust = _load(CAMVA_ROOT / "metrics" / "robustness.json")
    cal_t = _load(CAMVA_ROOT / "calibration" / "temperature.json")
    cal_raw = _load(CAMVA_ROOT / "calibration" / "test_uncalibrated.json")
    cal_ts = _load(CAMVA_ROOT / "calibration" / "test_temperature.json")
    train_cfg = _load(CAMVA_ROOT / "configs" / "train_config.json")
    train_sum = _load(CAMVA_ROOT / "configs" / "train_summary.json")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "JaalTaka",
        "split_protocol": "physical_note_id disjoint; all 6 views of a note stay in one split",
        "split_metadata": split_meta,
        "model_baseline": "MultiViewCNNVIT (existing architecture, retrained on CAMVA split; old authenticity_cnn_vit.pt untouched)",
        "model_camva": "CAMVANet quality-attention fusion",
        "training": train_cfg,
        "training_runs": train_sum,
        "views_1_to_6": views,
        "ablations": ablations,
        "adaptive": adaptive,
        "robustness": robust,
        "calibration": {"temperature": cal_t, "uncalibrated_test": cal_raw, "temperature_test": cal_ts},
        "latency_device": (train_cfg or {}).get("device"),
        "latency_rtx4060": NOT,
        "latency_raspberry_pi_5": NOT,
        "cross_session_camera": NOT,
        "reason_cross_session": "JaalTaka folders do not expose camera/session IDs",
        "user_study": NOT,
        "three_seeds": NOT if not train_sum or len(train_sum) < 3 else "see training_runs",
        "limitations": [
            "Quality-order ranking on stored views uses all captured images to order fusion; live causal capture should use fixed/random order.",
            "No participant study in this package.",
            "Do not call CAMVA SOTA. Compare only numbers in this report.",
        ],
        "python": sys.version,
        "platform": platform.platform(),
    }
    (CAMVA_ROOT / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    flat = []
    if isinstance(views, list):
        for r in views:
            flat.append({"section": "views", **r})
    csv_path = CAMVA_ROOT / "final_report.csv"
    if flat:
        keys = sorted({k for r in flat for k in r})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=keys)
            w.writeheader()
            w.writerows(flat)

    def fmt_views() -> str:
        if not views:
            return "NOT MEASURED (run evaluate_camva.py after training)\n"
        lines = ["| views | baseline acc | CAMVA acc | abs Δ | paired p |", "|---:|---:|---:|---:|---:|"]
        for r in views:
            lines.append(
                f"| {r['n_views']} | {r['baseline_accuracy']:.4f} | {r['camva_accuracy']:.4f} | "
                f"{r['absolute_improvement_acc']:.4f} | {r['paired_p_accuracy']:.4f} |"
            )
        return "\n".join(lines) + "\n"

    readme = f"""# CAMVA experiment report

Generated: {report['created_utc']}

Every number below is copied from a file under `results/camva/`. If a section says **NOT MEASURED**, the experiment was not run.

## Dataset and split

- Dataset: JaalTaka (`real_notes` / `fake_notes`)
- Independent unit: physical note ID (`genuine:note_XXX` / `counterfeit:note_XXX`)
- All six views of one note stay in exactly one of train/val/test
- Seed and counts: see `splits/split_metadata.json`

## Training

See `configs/train_config.json`. Checkpoints in `checkpoints/` — **does not overwrite** `models/authenticity_cnn_vit.pt`.

## Baseline vs CAMVA (1–6 views)

{fmt_views()}

Predictions: `predictions/baseline_{{k}}view.json` and `predictions/camva_{{k}}view.json`.

## Ablations / adaptive / calibration / robustness

- Ablations: `metrics/ablations.json` or NOT MEASURED
- Adaptive thresholds: `metrics/adaptive.json` or NOT MEASURED
- Calibration: `calibration/` (T fitted on **val** only)
- Robustness: `metrics/robustness.json` or NOT MEASURED

## Latency

- Device used in training config: {report['latency_device']}
- RTX 4060 Laptop: NOT MEASURED (this machine may differ)
- Raspberry Pi 5: NOT MEASURED

## Not claimed

- SOTA
- Automatic improvement (read the table)
- User-study outcomes
- Cross-camera generalization (no camera IDs in JaalTaka)

## Limitations

- Offline confidence ordering peeks at stored views to rank them; sequential **stopping** still uses only the prefix already fused.
- Single-seed training unless `train_summary.json` lists three seeds.
"""
    (CAMVA_ROOT / "README.md").write_text(readme, encoding="utf-8")
    print("wrote", CAMVA_ROOT / "final_report.json")
    print("wrote", CAMVA_ROOT / "README.md")


if __name__ == "__main__":
    main()
