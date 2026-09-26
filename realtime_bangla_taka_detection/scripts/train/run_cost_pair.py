"""Matched prefix trainings that differ only by a cost term with a gradient.

cost_entropy adds mean predictive entropy to the cost loss.
no_cost_matched uses the same recipe with that term removed.
The saved PRMVT checkpoint is not overwritten.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
PREFIX = ROOT / "configs" / "proposed_prefix.yaml"
FT = ROOT / "configs" / "proposed_prefix_ft.yaml"
OUT_ROOT = ROOT / "results" / "qduig" / "ablation_prefix"
CFG_ROOT = ROOT / "configs" / "ablation_prefix"

FULL = {
    "use_quality": True,
    "use_uncertainty": True,
    "use_diversity": True,
    "use_info_gain": True,
    "use_hgef": True,
    "use_cost": True,
    "use_redundancy": True,
    "cost_from_entropy": False,
}
LAMBDAS = {"lambda_q": 0.25, "lambda_u": 0.15, "lambda_d": 0.10, "lambda_g": 0.20, "lambda_c": 0.05}

VARIANTS = {
    "cost_entropy": {**FULL, "use_cost": True, "cost_from_entropy": True},
    "no_cost_matched": {**FULL, "use_cost": False, "cost_from_entropy": False},
}


def _write(base: dict, name: str, flags: dict, *, finetune: bool) -> Path:
    cfg = json.loads(json.dumps(base))
    cfg["name"] = f"ablation_prefix_{name}" + ("_ft" if finetune else "")
    cfg["seeds"] = [42]
    cfg["model"] = dict(cfg.get("model", {}))
    cfg["model"].update(flags)
    loss = dict(cfg.get("loss", {}))
    loss.update(LAMBDAS)
    if not flags["use_cost"]:
        loss["lambda_c"] = 0.0
    loss["lambda_auth"] = 1.0
    cfg["loss"] = loss
    path = CFG_ROOT / f"{name}{'_ft' if finetune else ''}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _train(name: str, flags: dict) -> None:
    done = OUT_ROOT / name / "seed42" / "test_views" / "views_1_to_6.json"
    if done.is_file():
        print("skip", name, flush=True)
        return
    stage1 = OUT_ROOT / name / "stage1"
    stage2 = OUT_ROOT / name / "seed42"
    cfg1 = _write(yaml.safe_load(PREFIX.read_text(encoding="utf-8")), name, flags, finetune=False)
    cfg2 = _write(yaml.safe_load(FT.read_text(encoding="utf-8")), name, flags, finetune=True)
    if not (stage1 / "checkpoint.pt").is_file():
        code = subprocess.run(
            [PY, str(ROOT / "scripts" / "train_qduig.py"), "--config", str(cfg1), "--seed", "42", "--output-dir", str(stage1)],
            cwd=ROOT,
        ).returncode
        if code != 0:
            print("FAIL stage1", name, code, flush=True)
            return
    code = subprocess.run(
        [PY, str(ROOT / "scripts" / "train_qduig.py"), "--config", str(cfg2), "--seed", "42", "--output-dir", str(stage2), "--resume", str(stage1 / "checkpoint.pt")],
        cwd=ROOT,
    ).returncode
    if code != 0:
        print("FAIL stage2", name, code, flush=True)
        return
    subprocess.run(
        [PY, str(ROOT / "scripts" / "eval_prefix_views.py"), "--checkpoint", str(stage2 / "checkpoint.pt"), "--config", str(cfg2), "--split", "test", "--seed", "42", "--output-dir", str(stage2 / "test_views")],
        cwd=ROOT,
    )


def main() -> None:
    for name, flags in VARIANTS.items():
        _train(name, flags)
    print("COST PAIR DONE", flush=True)


if __name__ == "__main__":
    main()
