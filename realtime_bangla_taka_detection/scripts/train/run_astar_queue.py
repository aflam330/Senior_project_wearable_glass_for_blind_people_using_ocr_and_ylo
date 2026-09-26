"""Seeds 43 and 44 for the six leading models, then the Q-DUIG ablations.

Does not overwrite seed-42 artifacts. Skips a run whose test file already exists.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
NOVEL = [
    ("ndal", ROOT / "configs" / "v2" / "ndal.yaml"),
    ("pravt", ROOT / "configs" / "v2" / "pravt.yaml"),
    ("vat", ROOT / "configs" / "v2" / "vat.yaml"),
    ("ugf", ROOT / "configs" / "ugf.yaml"),
    ("cvs", ROOT / "configs" / "v2" / "cvs.yaml"),
]


def _log(msg: str) -> None:
    print(msg, flush=True)
    path = ROOT / "results" / "astar" / "queue.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(msg + "\n")


def _run(cmd: list[str], env: dict | None = None) -> int:
    _log("RUN " + " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def _novel_seeds() -> None:
    env = os.environ.copy()
    env["NOVEL_WORKERS"] = "0"
    env["NOVEL_COMPILE"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    for algo, cfg in NOVEL:
        for seed in (43, 44):
            out = ROOT / "results" / "novel_v2" / algo / f"seed{seed}"
            metrics = out / "test" / "test_metrics.json"
            if metrics.is_file():
                _log(f"skip {algo} seed {seed}")
                continue
            code = _run([
                PY, str(ROOT / "scripts" / "train" / "train_novel.py"),
                "--algo", algo, "--seed", str(seed), "--config", str(cfg), "--output-dir", str(out),
            ], env)
            if code != 0 or not (out / "checkpoint.pt").is_file():
                _log(f"FAIL train {algo} seed {seed} exit {code}")
                continue
            _run([
                PY, str(ROOT / "scripts" / "eval" / "eval_novel.py"),
                "--algo", algo, "--seed", str(seed), "--config", str(cfg),
                "--checkpoint", str(out / "checkpoint.pt"), "--split", "test",
                "--output-dir", str(out / "test"),
            ], env)
    _log("PHASE 1 novel seeds finished")


def _prmvt_seeds() -> None:
    for seed in (43, 44):
        done = ROOT / "results" / "qduig" / "prefix_ft" / f"seed{seed}" / "test_views" / "views_1_to_6.json"
        if done.is_file():
            _log(f"skip prmvt seed {seed}")
            continue
        code = _run([PY, str(ROOT / "scripts" / "run_phase1_multiseed.py")])
        _log(f"prmvt phase1 exit {code}")
        return
    _log("PHASE 1 PRMVT seeds finished")


def _ablations() -> None:
    for yml in sorted((ROOT / "configs").glob("ablation_*.yaml")):
        if yml.name == "ablation_baseline.yaml":
            _log("baseline ablation uses the existing CNN+ViT checkpoint")
            continue
        name = yml.stem.replace("ablation_", "")
        out = ROOT / "results" / "qduig" / "ablations" / name / "seed42"
        views = out / "test_views" / "views_1_to_6.json"
        if views.is_file():
            _log(f"skip ablation {name}")
            continue
        if not (out / "checkpoint.pt").is_file():
            code = _run([
                PY, str(ROOT / "scripts" / "train_qduig.py"),
                "--config", str(yml), "--seed", "42", "--output-dir", str(out),
            ])
            if code != 0:
                _log(f"FAIL ablation train {name} exit {code}")
                continue
        _run([
            PY, str(ROOT / "scripts" / "eval_prefix_views.py"),
            "--checkpoint", str(out / "checkpoint.pt"), "--config", str(yml),
            "--split", "test", "--seed", "42", "--output-dir", str(out / "test_views"),
        ])
    _log("PHASE 2 ablations finished")


def main() -> None:
    _log("PHASE 1 start")
    _novel_seeds()
    _prmvt_seeds()
    _log("PHASE 2 start")
    _ablations()
    _log("PHASE 6 robustness start")
    _run([PY, str(ROOT / "scripts" / "eval" / "eval_robustness_top.py")])
    _log("PHASE 1 stats start")
    _run([PY, str(ROOT / "scripts" / "eval" / "aggregate_multiseed.py")])
    _log("ASTAR QUEUE DONE")


if __name__ == "__main__":
    main()
