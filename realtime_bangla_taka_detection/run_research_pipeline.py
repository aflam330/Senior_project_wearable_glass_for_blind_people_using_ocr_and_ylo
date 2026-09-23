"""Master entry: python run_research_pipeline.py --config configs/final_research.yaml"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PY = sys.executable
sys.path.insert(0, str(ROOT))

from roboeye.camva.notes import assert_disjoint_splits, load_splits
from roboeye.qduig.artifacts import QDUIG_ROOT, save_json
from roboeye.qduig.config_io import load_config


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def phase1_data() -> dict:
    splits, records = load_splits()
    assert_disjoint_splits(splits["train"], splits["val"], splits["test"])
    meta = json.loads((ROOT / "results" / "camva" / "splits" / "split_metadata.json").read_text(encoding="utf-8"))
    inter = {
        "train_val": len(set(splits["train"]) & set(splits["val"])),
        "train_test": len(set(splits["train"]) & set(splits["test"])),
        "val_test": len(set(splits["val"]) & set(splits["test"])),
    }
    ok = all(v == 0 for v in inter.values())
    return {"status": "PASS" if ok else "FAIL", "intersections": inter, "meta": meta, "n_records": len(records)}


def write_static_docs() -> None:
    ev = WORKSPACE / "paper_evidence"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "DATA_SPLITS.md").write_text(
        (ROOT / "results" / "camva" / "splits" / "split_metadata.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (ev / "EXPERIMENT_PROTOCOL.md").write_text(
        "\n".join(
            [
                "# Experiment protocol",
                "",
                "- Independent unit: physical note ID (`genuine|counterfeit:<folder>`).",
                "- Split seed 42, val/test 15% each, stratified by label. All 6 views stay with the note.",
                "- Test notes are never used for training, quality-threshold calibration, temperature/HER, λ/α/β/γ, or early-stop.",
                "- Baseline: CNN+ViT MultiViewCNNVIT, same loader, 128px, Adam 1e-3, 6 epochs, frozen MobileNetV3-Small.",
                "- Proposed: Q-DUIG-CAMVA, same split and preprocessing.",
                "- Seeds 42, 43, 44. Missing seeds stay NOT_MEASURED.",
                "- Predefined comparison: FULL PROPOSED vs CNN+ViT BASELINE.",
                "- Hypotheses H1–H4 are not revised after seeing test numbers.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "final_research.yaml"))
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-ablations", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    QDUIG_ROOT.mkdir(parents=True, exist_ok=True)

    print("PHASE 1 data")
    data = phase1_data()
    save_json(QDUIG_ROOT / "phase1_data.json", {k: v for k, v in data.items() if k != "meta"})
    write_static_docs()
    if data["status"] != "PASS":
        print("DATA FAIL — stopping")
        sys.exit(1)

    print("PHASE 2–17 experiments")
    rc = run(
        [
            PY,
            str(ROOT / "scripts" / "run_all_experiments.py"),
            "--config",
            args.config,
            *(["--skip-train"] if args.skip_train else []),
            *(["--skip-ablations"] if args.skip_ablations else []),
            *(["--skip-eval"] if args.skip_eval else []),
        ]
    )
    if rc != 0:
        print("experiment runner rc", rc, "continuing to export what exists")

    print("PHASE 18 figures/tables")
    run([PY, str(ROOT / "scripts" / "generate_paper_tables.py")])
    run([PY, str(ROOT / "scripts" / "generate_paper_figures.py")])

    print("PHASE 19–21 evidence")
    run([PY, str(ROOT / "scripts" / "write_qduig_evidence.py")])
    run([PY, str(ROOT / "scripts" / "validate_claims.py")])

    print("PHASE 22 status written by write_qduig_evidence.py")
    print("pipeline finished", datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    main()
