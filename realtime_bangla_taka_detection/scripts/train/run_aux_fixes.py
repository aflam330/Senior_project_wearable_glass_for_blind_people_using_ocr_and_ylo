"""Sequential auxiliary-stack fixes. Seed 42. Does not overwrite prefix_ft or her_base."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
TRAIN = ROOT / "scripts" / "train_qduig.py"
EVAL = ROOT / "scripts" / "eval_prefix_views.py"
PREFIX = ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt"
HER = ROOT / "results" / "qduig" / "ablation_prefix" / "her_base" / "seed42" / "checkpoint.pt"
CFG_DIR = ROOT / "configs" / "aux_fixes"

BASE_MODEL = """
  freeze_cnn: true
  use_quality: true
  use_uncertainty: true
  use_diversity: true
  use_info_gain: true
  use_hgef: true
  use_cost: true
  use_redundancy: true
  view_self_gate: true
  view_count_bn: true
  max_views: 6
"""

HER_MODEL = """
  freeze_cnn: true
  use_quality: false
  use_uncertainty: false
  use_diversity: false
  use_info_gain: true
  use_hgef: true
  use_cost: false
  use_redundancy: false
  view_self_gate: true
  view_count_bn: true
  aux_stopgrad: true
  max_views: 6
"""


def yaml_text(name, epochs, loss, model_block, extra=""):
    return f"""name: {name}
kind: qduig
epochs: {epochs}
batch: 8
lr: 0.0003
views: 6
view_dropout: mixed
full_view_prob: 0.5
curriculum: false
multitask_single: 0.5
contrastive: 0.05
{extra}model:
{model_block}loss:
  lambda_auth: 1.0
  lambda_q: {loss[0]}
  lambda_u: {loss[1]}
  lambda_d: {loss[2]}
  lambda_g: {loss[3]}
  lambda_c: {loss[4]}
cost:
  alpha: 1.0
  beta: 0.05
  gamma: 0.0
  lambda_cost_policy: 0.02
"""


FULL = (0.25, 0.15, 0.10, 0.20, 0.05)


def scaled(s):
    return tuple(v * s for v in FULL)


JOBS = [
    ("auxfix_scale_0p1", yaml_text("auxfix_scale_0p1", 2, scaled(0.1), BASE_MODEL), PREFIX),
    ("auxfix_scale_0p01", yaml_text("auxfix_scale_0p01", 2, scaled(0.01), BASE_MODEL), PREFIX),
    ("auxfix_scale_0p001", yaml_text("auxfix_scale_0p001", 2, scaled(0.001), BASE_MODEL), PREFIX),
    ("auxfix_scale_0p0001", yaml_text("auxfix_scale_0p0001", 2, scaled(0.0001), BASE_MODEL), PREFIX),
    ("auxfix_scale_divcost", yaml_text("auxfix_scale_divcost", 2, (0.25, 0.15, 0.001, 0.20, 0.001), BASE_MODEL), PREFIX),
    ("auxfix_pcgrad", yaml_text("auxfix_pcgrad", 2, FULL, BASE_MODEL, extra="aux_pcgrad: true\n"), PREFIX),
    ("auxfix_stopgrad", yaml_text("auxfix_stopgrad", 2, FULL, HER_MODEL, extra="multitask_single: 0.0\ncontrastive: 0.0\n").replace(
        "multitask_single: 0.5\ncontrastive: 0.05\nmultitask_single: 0.0\ncontrastive: 0.0\n",
        "multitask_single: 0.0\ncontrastive: 0.0\n",
    ), HER),
    ("auxfix_kendall", yaml_text("auxfix_kendall", 2, FULL, BASE_MODEL, extra="kendall: true\n"), PREFIX),
    ("auxfix_curriculum", yaml_text(
        "auxfix_curriculum", 6, FULL, BASE_MODEL,
        extra="aux_curriculum:\n  - [1, 0.0]\n  - [3, 0.01]\n  - [5, 0.1]\n",
    ), PREFIX),
    ("auxfix_separate", yaml_text(
        "auxfix_separate", 2, FULL,
        HER_MODEL.replace("aux_stopgrad: true\n", "separate_aux: true\n"),
        extra="separate_aux: true\nmultitask_single: 0.0\ncontrastive: 0.0\n",
    ).replace(
        "multitask_single: 0.5\ncontrastive: 0.05\nseparate_aux: true\nmultitask_single: 0.0\ncontrastive: 0.0\n",
        "multitask_single: 0.0\ncontrastive: 0.0\nseparate_aux: true\n",
    ), HER),
    ("auxfix_regularize", yaml_text("auxfix_regularize", 2, FULL, BASE_MODEL, extra="aux_every: 4\n"), PREFIX),
]


def run(cmd):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NOVEL_WORKERS"] = "0"
    env["NOVEL_COMPILE"] = "0"
    print("CMD", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def main():
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, text, resume in JOBS:
        cfg = CFG_DIR / f"{name}.yaml"
        cfg.write_text(text, encoding="utf-8")
        out = ROOT / "results" / "qduig" / name / "seed42"
        views = out / "test_views" / "views_1_to_6.json"
        if not views.is_file():
            if not (out / "checkpoint.pt").is_file():
                run([PY, str(TRAIN), "--config", str(cfg), "--seed", "42", "--output-dir", str(out), "--resume", str(resume)])
            run([
                PY, str(EVAL),
                "--checkpoint", str(out / "checkpoint.pt"),
                "--config", str(cfg),
                "--split", "test",
                "--seed", "42",
                "--output-dir", str(out / "test_views"),
            ])
        blob = json.loads(views.read_text(encoding="utf-8"))
        accs = [row["accuracy"] for row in blob["proposed"]]
        one = accs[0]
        print(f"RESULT {name} 1-view {one}", flush=True)
        summary.append({"name": name, "acc": accs})
        (ROOT / "results" / "qduig" / "auxfix_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
