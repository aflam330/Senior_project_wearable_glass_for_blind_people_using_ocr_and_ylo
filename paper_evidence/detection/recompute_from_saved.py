"""Recompute wild-note metrics from saved currency_boost.json (no new inference)."""
from __future__ import annotations

import json
from pathlib import Path

SRC = Path(r"E:\Final SP\savior_glass\results\currency_boost.json")
OUT = Path(r"E:\Final SP\paper_evidence\detection")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(SRC.read_text(encoding="utf-8"))
    wild = data["wild_existing_yolo"]
    n = int(wild["n"])
    per = wild["per_true"]
    det = 0
    top1 = 0
    none = 0
    for true_cls, counts in per.items():
        for pred, c in counts.items():
            c = int(c)
            if pred == "none":
                none += c
            else:
                det += c
            if pred == true_cls:
                top1 += c
    out = {
        "source_file": str(SRC),
        "recomputed_from": "per_true confusion counts",
        "n": n,
        "detected": det,
        "missed": none,
        "correct_denomination": top1,
        "detection_rate": det / n,
        "top1_accuracy": top1 / n,
        "per_true": per,
        "weights": wild.get("weights"),
        "yolo_val": data.get("yolo_val"),
        "note": "Recomputed from saved counts. Not a new YOLO pass.",
    }
    (OUT / "wild_note_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n", "detection_rate", "top1_accuracy")}, indent=2))


if __name__ == "__main__":
    main()
