"""Prefix-protocol ablations, seed 42.

Full Q-DUIG is the existing 6+3 epoch PRMVT checkpoint. It is not retrained.
Baseline is the existing CNN+ViT checkpoint. It is not retrained.
-cost and -calibration do not change forced-view gradients; they are evaluated
on that same PRMVT checkpoint. Every other row is trained with the prefix recipe.
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

# Flags that change the loss or the fusion path. hgef stays on so the
# classifier is the same module as PRMVT; disabled losses are zeroed.
FULL = {
    "use_quality": True,
    "use_uncertainty": True,
    "use_diversity": True,
    "use_info_gain": True,
    "use_hgef": True,
    "use_cost": True,
    "use_redundancy": True,
}


def _spec(flags: dict, lambdas: dict, *, like_prmvt: bool) -> dict:
    return {"flags": flags, "lambdas": lambdas, "like_prmvt": like_prmvt}


def _off(**on: bool) -> dict:
    flags = {k: False for k in FULL}
    flags["use_hgef"] = True
    flags.update(on)
    return flags


VARIANTS = {
    "rsqa": _spec(_off(use_quality=True), {"lambda_q": 0.25}, like_prmvt=False),
    "cvr": _spec(_off(use_diversity=True, use_redundancy=True), {"lambda_d": 0.10}, like_prmvt=False),
    "her_base": _spec(_off(), {}, like_prmvt=False),
    "pcr_ig": _spec(_off(use_info_gain=True), {"lambda_g": 0.20}, like_prmvt=False),
    "qd": _spec(_off(use_quality=True, use_diversity=True, use_redundancy=True), {"lambda_q": 0.25, "lambda_d": 0.10}, like_prmvt=False),
    "ud": _spec(_off(use_uncertainty=True, use_diversity=True, use_redundancy=True), {"lambda_u": 0.15, "lambda_d": 0.10}, like_prmvt=False),
    "no_redundancy": _spec({**FULL, "use_redundancy": False}, {"lambda_q": 0.25, "lambda_u": 0.15, "lambda_d": 0.0, "lambda_g": 0.20, "lambda_c": 0.05}, like_prmvt=True),
    "no_infogain": _spec({**FULL, "use_info_gain": False}, {"lambda_q": 0.25, "lambda_u": 0.15, "lambda_d": 0.10, "lambda_g": 0.0, "lambda_c": 0.05}, like_prmvt=True),
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _write_cfg(base: dict, name: str, spec: dict, *, finetune: bool) -> Path:
    cfg = json.loads(json.dumps(base))
    cfg["name"] = f"ablation_prefix_{name}" + ("_ft" if finetune else "")
    cfg["seeds"] = [42]
    cfg["model"] = dict(cfg.get("model", {}))
    cfg["model"].update(spec["flags"])
    loss = dict(cfg.get("loss", {}))
    for key in ("lambda_q", "lambda_u", "lambda_d", "lambda_g", "lambda_c"):
        loss[key] = 0.0
    loss.update(spec["lambdas"])
    loss["lambda_auth"] = 1.0
    cfg["loss"] = loss
    if not spec["like_prmvt"]:
        cfg["multitask_single"] = 0.0
        cfg["contrastive"] = 0.0
    path = CFG_ROOT / f"{name}{'_ft' if finetune else ''}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _run(cmd: list[str]) -> int:
    _log("RUN " + " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


def _train_one(name: str, spec: dict) -> None:
    done = OUT_ROOT / name / "seed42" / "test_views" / "views_1_to_6.json"
    if done.is_file():
        _log(f"skip {name}")
        return
    stage1 = OUT_ROOT / name / "stage1"
    stage2 = OUT_ROOT / name / "seed42"
    cfg1 = _write_cfg(yaml.safe_load(PREFIX.read_text(encoding="utf-8")), name, spec, finetune=False)
    cfg2 = _write_cfg(yaml.safe_load(FT.read_text(encoding="utf-8")), name, spec, finetune=True)
    if not (stage1 / "checkpoint.pt").is_file():
        code = _run([PY, str(ROOT / "scripts" / "train_qduig.py"), "--config", str(cfg1), "--seed", "42", "--output-dir", str(stage1)])
        if code != 0:
            _log(f"FAIL stage1 {name} {code}")
            return
    code = _run([
        PY, str(ROOT / "scripts" / "train_qduig.py"),
        "--config", str(cfg2), "--seed", "42", "--output-dir", str(stage2),
        "--resume", str(stage1 / "checkpoint.pt"),
    ])
    if code != 0 or not (stage2 / "checkpoint.pt").is_file():
        _log(f"FAIL stage2 {name} {code}")
        return
    _run([
        PY, str(ROOT / "scripts" / "eval_prefix_views.py"),
        "--checkpoint", str(stage2 / "checkpoint.pt"),
        "--config", str(cfg2), "--split", "test", "--seed", "42",
        "--output-dir", str(stage2 / "test_views"),
    ])


def _accs(path: Path, key: str = "proposed") -> dict[int, float] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get(key) or payload.get("views") or []
    return {int(row["k"]): float(row["accuracy"]) for row in rows}


def _write_md() -> None:
    base = {}
    for k in range(1, 7):
        path = ROOT / "results" / "qduig" / "eval" / "seed42" / "baseline" / f"baseline_{k}view" / "test_metrics.json"
        if path.is_file():
            base[k] = float(json.loads(path.read_text(encoding="utf-8"))["accuracy"])
    prmvt = _accs(ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "test_views" / "views_1_to_6.json")
    order = [
        ("baseline", base, "existing CNN+ViT, not retrained"),
        ("rsqa", _accs(OUT_ROOT / "rsqa" / "seed42" / "test_views" / "views_1_to_6.json"), "quality loss only, prefix 6+3"),
        ("cvr", _accs(OUT_ROOT / "cvr" / "seed42" / "test_views" / "views_1_to_6.json"), "diversity and redundancy only, prefix 6+3"),
        ("her_base", _accs(OUT_ROOT / "her_base" / "seed42" / "test_views" / "views_1_to_6.json"), "prefix encoder, auxiliary losses off; HER is an eval map, see CALIBRATION_RESULTS.md"),
        ("pcr_ig", _accs(OUT_ROOT / "pcr_ig" / "seed42" / "test_views" / "views_1_to_6.json"), "information-gain loss only, prefix 6+3"),
        ("qd", _accs(OUT_ROOT / "qd" / "seed42" / "test_views" / "views_1_to_6.json"), "quality plus diversity, prefix 6+3"),
        ("ud", _accs(OUT_ROOT / "ud" / "seed42" / "test_views" / "views_1_to_6.json"), "uncertainty plus diversity, prefix 6+3"),
        ("full_prmvt", prmvt, "existing prefix_ft seed 42, 6+3 epochs, not retrained"),
        ("no_cost", prmvt, "same weights as full. Cost has no parameter gradient, so a retrain would not be a distinct classifier. Adaptive-policy numbers are in the cost file if present."),
        ("no_calibration", prmvt, "same weights as full. Removing HER does not change the 0.5-threshold accuracy of the uncalibrated checkpoint. ECE is in CALIBRATION_RESULTS.md."),
        ("no_redundancy", _accs(OUT_ROOT / "no_redundancy" / "seed42" / "test_views" / "views_1_to_6.json"), "PRMVT recipe with redundancy off"),
        ("no_infogain", _accs(OUT_ROOT / "no_infogain" / "seed42" / "test_views" / "views_1_to_6.json"), "PRMVT recipe with information-gain loss off"),
    ]
    lines = [
        "# Ablation results",
        "",
        "Seed 42, test split, n=208. Forced first-k views. Full is the saved prefix-robust checkpoint.",
        "The earlier 4-epoch runs in `results/qduig/ablations/` are unchanged and are not this table.",
        "no_cost and no_calibration matched that 4-epoch full run because those switches do not change forced-view training. That cause is recorded here instead of training a duplicate.",
        "",
        "| config | 1 | 2 | 3 | 4 | 5 | 6 | note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    full = prmvt or {}
    for name, accs, note in order:
        if not accs:
            cells = " | ".join(["NOT_MEASURED"] * 6)
            delta = ""
        else:
            cells = " | ".join(f"{accs[k]:.4f}" for k in range(1, 7))
            if name not in {"full_prmvt", "baseline"} and full:
                delta = " 1-view minus full: " + f"{accs[1] - full[1]:+.4f}"
            else:
                delta = ""
        lines.append(f"| {name} | {cells} | {note}{delta} |")
    lines.append("")
    path = ROOT.parent / "paper_evidence" / "ABLATION_RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    _log(str(path))


def main() -> None:
    for name, spec in VARIANTS.items():
        _train_one(name, spec)
    _write_md()
    _log("ABLATION PREFIX DONE")


if __name__ == "__main__":
    main()
