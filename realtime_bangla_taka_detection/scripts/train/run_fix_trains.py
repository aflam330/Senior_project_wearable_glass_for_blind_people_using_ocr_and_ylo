"""Fine-tunes for cost, auxiliary weight, and corruption. Seed 42. New directories."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def _run(args: list[str]) -> int:
    print("RUN", " ".join(args), flush=True)
    return subprocess.run(args, cwd=ROOT).returncode


def _train(config: str, resume: str, out: str) -> None:
    done = Path(out) / "checkpoint.pt"
    if done.is_file():
        print("skip", out, flush=True)
        return
    code = _run([
        PY, str(ROOT / "scripts" / "train_qduig.py"),
        "--config", str(ROOT / config),
        "--seed", "42",
        "--output-dir", out,
        "--resume", resume,
    ])
    if code != 0:
        print("FAIL train", out, code, flush=True)


def main() -> None:
    prefix = str(ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt")
    nocost = str(ROOT / "results" / "qduig" / "ablation_prefix" / "no_cost_matched" / "seed42" / "checkpoint.pt")
    _train("configs/cost_norm_ft.yaml", nocost, str(ROOT / "results" / "qduig" / "cost_norm_ft" / "seed42"))
    _train("configs/aux_small_ft.yaml", prefix, str(ROOT / "results" / "qduig" / "aux_small_ft" / "seed42"))
    _train("configs/robust_ft.yaml", prefix, str(ROOT / "results" / "qduig" / "robust_ft" / "seed42"))
    for name, cfg in (
        ("cost_norm_ft", "configs/cost_norm_ft.yaml"),
        ("aux_small_ft", "configs/aux_small_ft.yaml"),
    ):
        ckpt = ROOT / "results" / "qduig" / name / "seed42" / "checkpoint.pt"
        dest = ckpt.parent / "test_views"
        if not ckpt.is_file() or (dest / "views_1_to_6.json").is_file():
            continue
        _run([
            PY, str(ROOT / "scripts" / "eval_prefix_views.py"),
            "--checkpoint", str(ckpt),
            "--config", str(ROOT / cfg),
            "--split", "test",
            "--seed", "42",
            "--output-dir", str(dest),
        ])
    robust = ROOT / "results" / "qduig" / "robust_ft" / "seed42" / "checkpoint.pt"
    robust_out = ROOT / "results" / "robustness" / "robust_ft_seed42.json"
    if robust.is_file() and not robust_out.is_file():
        _run([
            PY, str(ROOT / "scripts" / "eval" / "eval_robust_checkpoint.py"),
            "--checkpoint", str(robust),
            "--config", str(ROOT / "configs" / "robust_ft.yaml"),
            "--output", str(robust_out),
        ])
    pac = ROOT / "results" / "theory" / "pacbayes_head_d1d2_seed42.json"
    if not pac.is_file():
        _run([PY, str(ROOT / "scripts" / "eval" / "compute_pacbayes_head.py")])
    print("FIX TRAINS DONE", flush=True)


if __name__ == "__main__":
    main()
