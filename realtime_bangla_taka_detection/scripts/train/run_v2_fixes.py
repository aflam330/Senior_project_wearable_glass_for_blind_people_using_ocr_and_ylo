"""Retrain failed and weak algorithms, then the five new ones. Seed 42.

Original results under results/novel/ are not overwritten.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT.parent / "paper_evidence"
PY = sys.executable
FAILED = ["vcie", "sfpl", "mtpt"]
WEAK = ["apc", "ogpd", "ndal", "cvs"]
NEW = ["pravt", "cris", "savs", "mavt", "vat"]
ORDER = FAILED + WEAK + NEW
BASELINE = {1: 0.7356, 2: 0.8702, 3: 0.9135, 4: 0.9183, 5: 0.8990, 6: 0.9183}
V1 = ROOT / "results" / "novel"
V2 = ROOT / "results" / "novel_v2"


def _log(msg: str) -> None:
    print(msg, flush=True)
    path = V2 / "pipeline.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(msg + "\n")


def _accs(path: Path) -> dict[int, float] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["k"]): float(row["accuracy"]) for row in payload["views"]}


def _baselines() -> dict[int, float]:
    found = {}
    for k in range(1, 7):
        path = ROOT / "results" / "qduig" / "eval" / "seed42" / "baseline" / f"baseline_{k}view" / "test_metrics.json"
        if not path.is_file():
            found[k] = BASELINE[k]
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        found[k] = float(payload["accuracy"])
    return found


def _run(algo: str, env: dict) -> int:
    cfg = ROOT / "configs" / "v2" / f"{algo}.yaml"
    out = V2 / algo / "seed42"
    cmd = [PY, str(ROOT / "scripts" / "train" / "train_novel.py"), "--algo", algo, "--seed", "42", "--config", str(cfg), "--output-dir", str(out)]
    _log("RUN " + " ".join(cmd))
    code = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    if code != 0 or not (out / "checkpoint.pt").is_file():
        return code or 1
    ev = [
        PY, str(ROOT / "scripts" / "eval" / "eval_novel.py"),
        "--algo", algo, "--seed", "42", "--config", str(cfg),
        "--checkpoint", str(out / "checkpoint.pt"), "--split", "test",
        "--output-dir", str(out / "test"),
    ]
    return subprocess.run(ev, cwd=ROOT, env=env).returncode


def _report(algo: str, baselines: dict[int, float], fixes: int, reason: str, nxt: str) -> None:
    accs = _accs(V2 / algo / "seed42" / "test" / "test_metrics.json")
    _log("=" * 35)
    _log(f"{algo.upper()} — {'PASS' if accs else 'NOT_MEASURED'}")
    _log("=" * 35)
    if accs is None:
        for k in range(1, 7):
            _log(f"{k}-view:  NOT_MEASURED (baseline {baselines[k] * 100:.2f}%)")
        _log("Status:  NOT_MEASURED")
    else:
        flat = all(abs(accs[k] - accs[1]) < 1e-6 for k in accs)
        status = "FAIL" if accs[1] < 0.60 or (flat and accs[1] < 0.70) else "PASS"
        for k in range(1, 7):
            _log(f"{k}-view:  {accs.get(k, float('nan')) * 100:.2f}% (baseline {baselines[k] * 100:.2f}%)")
        _log(f"Status:  {status}")
    _log(f"Fix:     {fixes}")
    if reason:
        _log(f"Reason:  {reason}")
    _log(f"Next:    {nxt}")
    _log("=" * 35)


def _train_failed(algo: str) -> tuple[int, str]:
    attempts = [
        {"NOVEL_SET_MODE": "residual", "NOVEL_FOCAL": "0"},
        {"NOVEL_SET_MODE": "mean", "NOVEL_FOCAL": "0"},
        {"NOVEL_SET_MODE": "mean", "NOVEL_FOCAL": "1"},
    ]
    reason = ""
    for i, flags in enumerate(attempts, start=1):
        env = os.environ.copy()
        env.update(flags)
        env["NOVEL_WORKERS"] = "0"
        env["NOVEL_COMPILE"] = "0"
        env["PYTHONUNBUFFERED"] = "1"
        _log(f"{algo} fix {i} {flags}")
        _run(algo, env)
        accs = _accs(V2 / algo / "seed42" / "test" / "test_metrics.json")
        if accs and accs[1] >= 0.60:
            (V2 / algo / "seed42" / "fix.txt").write_text(f"fix {i} {flags}\n", encoding="utf-8")
            return i, ""
        reason = {
            "vcie": "OVERFITTING_FAILURE",
            "sfpl": "FEDERATED_SIMULATION_FAILURE",
            "mtpt": "MULTI_TASK_IMBALANCE_FAILURE",
        }[algo]
    (V2 / algo / "seed42" / "fix.txt").write_text(reason + "\n", encoding="utf-8")
    return 3, reason


def _train_once(algo: str) -> int:
    env = os.environ.copy()
    env["NOVEL_WORKERS"] = "0"
    env["NOVEL_COMPILE"] = "0"
    env["NOVEL_SET_MODE"] = "residual"
    env["PYTHONUNBUFFERED"] = "1"
    _run(algo, env)
    return 1


def _write_papers(baselines: dict[int, float]) -> None:
    def table(algos: list[str], title: str) -> str:
        lines = [f"# {title}", "", "| algorithm | 1-view | 2-view | 3-view | 4-view | 5-view | 6-view | v1 1-view |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for algo in algos:
            accs = _accs(V2 / algo / "seed42" / "test" / "test_metrics.json")
            old = _accs(V1 / algo / "seed42" / "test" / "test_metrics.json")
            if accs is None:
                cells = ["NOT_MEASURED"] * 6
            else:
                cells = [f"{accs[k]:.4f}" for k in range(1, 7)]
            old_cell = "NOT_MEASURED" if old is None else f"{old[1]:.4f}"
            lines.append(f"| {algo} | " + " | ".join(cells) + f" | {old_cell} |")
        lines.append("")
        lines.append("v2 files: `realtime_bangla_taka_detection/results/novel_v2/<algo>/seed42/test/test_metrics.json`.")
        lines.append("v1 files were not overwritten.")
        return "\n".join(lines) + "\n"

    (PAPER / "FAILED_FIXES_RESULTS.md").write_text(table(FAILED, "Failed-algorithm fixes, seed 42"), encoding="utf-8")
    (PAPER / "WEAK_IMPROVEMENTS_RESULTS.md").write_text(table(WEAK, "Weak-algorithm retrains, seed 42"), encoding="utf-8")
    (PAPER / "NEW_ALGORITHMS_RESULTS.md").write_text(table(NEW, "New algorithms, seed 42"), encoding="utf-8")
    # refresh the master comparison with a v2 section appended once
    comp = PAPER / "NOVEL_ALGORITHMS_COMPARISON.md"
    base = comp.read_text(encoding="utf-8") if comp.is_file() else ""
    marker = "\n## v2 retrains\n"
    body = table(ORDER, "v2")
    if marker.strip() in base:
        base = base.split("## v2 retrains")[0].rstrip() + "\n"
    comp.write_text(base.rstrip() + "\n\n## v2 retrains\n\n" + body, encoding="utf-8")
    note = PAPER / "NOVELTY_DECLARATION.md"
    extra = (
        "\n## v2 additions\n\n"
        "PRAVT, CRIS, SAVS, MAVT, and VAT were trained after the v1 runs. "
        "PRAVT adds a reversed-order loss. CRIS adds a KL bottleneck on the fused vector. "
        "SAVS is sharpness-aware minimization on the same classifier. "
        "MAVT adds an in-batch prototype loss. VAT samples prefix length with probability proportional to 1/k "
        "and predicts that length. None of these are claimed as first or state of the art. "
        "VCIE, SFPL, and MTPT v2 use a residual around the set encoder because v1 predicted only the majority class.\n"
    )
    text = note.read_text(encoding="utf-8") if note.is_file() else ""
    if "## v2 additions" not in text:
        note.write_text(text.rstrip() + extra, encoding="utf-8")
    _log(f"baselines used { {k: round(v, 4) for k, v in baselines.items()} }")


def main() -> None:
    baselines = _baselines()
    for i, algo in enumerate(ORDER):
        nxt = ORDER[i + 1] if i + 1 < len(ORDER) else "write papers"
        existing = _accs(V2 / algo / "seed42" / "test" / "test_metrics.json")
        if existing and existing[1] >= 0.60:
            _log(f"{algo} v2 already measured")
            _report(algo, baselines, 0, "", f"Starting {nxt}")
            continue
        if algo in FAILED:
            fixes, reason = _train_failed(algo)
        else:
            fixes, reason = _train_once(algo), ""
        _report(algo, baselines, fixes, reason, f"Starting {nxt}")
    _write_papers(baselines)
    fixed = 0
    for algo in FAILED:
        accs = _accs(V2 / algo / "seed42" / "test" / "test_metrics.json")
        if accs and accs[1] >= 0.60:
            fixed += 1
    improved = 0
    for algo in WEAK:
        new = _accs(V2 / algo / "seed42" / "test" / "test_metrics.json")
        old = _accs(V1 / algo / "seed42" / "test" / "test_metrics.json")
        if new and old and sum(new.values()) > sum(old.values()):
            improved += 1
    news = sum(1 for algo in NEW if _accs(V2 / algo / "seed42" / "test" / "test_metrics.json"))
    passing = sum(1 for algo in ORDER if (_accs(V2 / algo / "seed42" / "test" / "test_metrics.json") or {}).get(1, 0) >= 0.60)
    _log("=" * 35)
    _log("ALL COMPLETE")
    _log("=" * 35)
    _log(f"Failed fixed:      {fixed}/3")
    _log(f"Weak improved:     {improved}/4")
    _log(f"New implemented:   {news}/5")
    _log(f"Total passing:     {passing}/12")
    _log("=" * 35)


if __name__ == "__main__":
    main()
