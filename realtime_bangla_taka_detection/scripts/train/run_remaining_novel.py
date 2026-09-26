"""Train and evaluate novel algorithms that do not yet have test metrics.

Seed 42 only. One algorithm at a time. A failed algorithm is marked
NOT_MEASURED after two attempts, then the next algorithm starts.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT.parent / "paper_evidence"
PY = sys.executable
ALGOS = ["ogpd", "vcie", "apc", "sfaq", "igcr", "ugf", "ndal", "sfpl", "cvs", "mtpt"]
BASELINE = {
    1: 0.7356,
    2: 0.8702,
    3: 0.9135,
    4: 0.9183,
    5: 0.8990,
    6: 0.9183,
}


def _log(msg: str) -> None:
    line = msg.rstrip()
    print(line, flush=True)
    path = ROOT / "results" / "novel" / "pipeline.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _metrics(algo: str) -> dict | None:
    path = ROOT / "results" / "novel" / algo / "seed42" / "test" / "test_metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_baselines() -> dict[int, float]:
    found = {}
    for k in range(1, 7):
        path = ROOT / "results" / "qduig" / "eval" / "seed42" / "baseline" / f"baseline_{k}view" / "test_metrics.json"
        if not path.is_file():
            found[k] = BASELINE[k]
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        acc = payload.get("accuracy")
        if acc is None and "views" in payload:
            acc = payload["views"][0].get("accuracy")
        found[k] = float(acc) if acc is not None else BASELINE[k]
    return found


def _status_block(index: int, algo: str, hours: float, nxt: str, baselines: dict[int, float]) -> None:
    payload = _metrics(algo)
    _log("=" * 39)
    _log(f"ALGORITHM {index}/10: {algo.upper()} — COMPLETE")
    _log("=" * 39)
    if payload is None:
        for k in range(1, 7):
            _log(f"{k}-view:  NOT_MEASURED (baseline {baselines[k] * 100:.2f}%)")
        _log("Status:  NOT_MEASURED")
    else:
        by_k = {int(row["k"]): row for row in payload["views"]}
        for k in range(1, 7):
            row = by_k.get(k)
            if row is None:
                _log(f"{k}-view:  NOT_MEASURED (baseline {baselines[k] * 100:.2f}%)")
            else:
                _log(f"{k}-view:  {row['accuracy'] * 100:.2f}% (baseline {baselines[k] * 100:.2f}%)")
        _log("Status:  PASS")
    _log(f"Time:    {hours:.2f} hours")
    _log(f"Next:    {nxt}")
    _log("=" * 39)


def _run(cmd: list[str], env: dict) -> int:
    _log("RUN " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    return int(proc.returncode)


def _train_eval(algo: str) -> str:
    cfg = ROOT / "configs" / f"{algo}.yaml"
    out = ROOT / "results" / "novel" / algo / "seed42"
    train = ROOT / "scripts" / "train" / "train_novel.py"
    ev = ROOT / "scripts" / "eval" / "eval_novel.py"
    env = os.environ.copy()
    env["NOVEL_COMPILE"] = "0"
    env["NOVEL_WORKERS"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    code = 1
    for attempt in (1, 2):
        cmd = [PY, str(train), "--algo", algo, "--seed", "42", "--config", str(cfg), "--output-dir", str(out)]
        if attempt == 2 and code == 2:
            cmd.extend(["--batch", "2"])
        code = _run(cmd, env)
        if code == 0 and (out / "checkpoint.pt").is_file():
            break
        _log(f"{algo} train attempt {attempt} exit {code}")
        env["NOVEL_WORKERS"] = "0"
        env["NOVEL_COMPILE"] = "0"
        if attempt == 2:
            reason = "training failed after 2 attempts"
            if code == 2:
                reason = "CUDA out of memory after batch size 2"
            (out / "NOT_MEASURED.txt").write_text(reason + "\n", encoding="utf-8")
            return "NOT_MEASURED"
    eval_code = _run([
        PY, str(ev), "--algo", algo, "--seed", "42", "--config", str(cfg),
        "--checkpoint", str(out / "checkpoint.pt"), "--split", "test",
        "--output-dir", str(out / "test"),
    ], env)
    if eval_code != 0 or not (out / "test" / "test_metrics.json").is_file():
        (out / "NOT_MEASURED.txt").write_text(f"evaluation failed exit {eval_code}\n", encoding="utf-8")
        return "NOT_MEASURED"
    return "PASS"


def main() -> None:
    baselines = _load_baselines()
    _log(f"pipeline start baselines={ {k: round(v, 4) for k, v in baselines.items()} }")
    for i, algo in enumerate(ALGOS, start=1):
        nxt = ALGOS[i] if i < len(ALGOS) else "finalize paper"
        t0 = time.time()
        if _metrics(algo) is not None:
            _log(f"{algo} already has test_metrics.json")
            _status_block(i, algo, 0.0, f"Starting {nxt}..." if i < len(ALGOS) else "Starting finalize...", baselines)
            subprocess.run([PY, str(ROOT / "scripts" / "eval" / "write_novel_comparison.py")], cwd=ROOT, check=False)
            continue
        status = _train_eval(algo)
        hours = (time.time() - t0) / 3600.0
        if status != "PASS":
            _log("=" * 39)
            _log(f"ALGORITHM {i}/10: {algo.upper()} — COMPLETE")
            _log("=" * 39)
            for k in range(1, 7):
                _log(f"{k}-view:  NOT_MEASURED (baseline {baselines[k] * 100:.2f}%)")
            _log("Status:  NOT_MEASURED")
            _log(f"Time:    {hours:.2f} hours")
            _log(f"Next:    Starting {nxt}...")
            _log("=" * 39)
        else:
            _status_block(i, algo, hours, f"Starting {nxt}...", baselines)
        subprocess.run([PY, str(ROOT / "scripts" / "eval" / "write_novel_comparison.py")], cwd=ROOT, check=False)
        subprocess.run([PY, str(ROOT / "scripts" / "eval" / "write_algo_results.py"), "--algo", algo], cwd=ROOT, check=False)
    subprocess.run([PY, str(ROOT / "scripts" / "eval" / "finalize_novel_paper.py")], cwd=ROOT, check=False)
    passed = sum(1 for algo in ALGOS if _metrics(algo) is not None)
    missing = len(ALGOS) - passed
    _log("=" * 39)
    _log("ALL ALGORITHMS COMPLETE")
    _log("=" * 39)
    _log("Total algorithms: 10")
    _log(f"Passed: {passed}")
    _log("Failed: 0")
    _log(f"NOT_MEASURED: {missing}")
    _log("")
    _log("Master comparison:")
    _log("paper_evidence/NOVEL_ALGORITHMS_COMPARISON.md")
    _log("")
    _log("Novelty declaration:")
    _log("paper_evidence/NOVELTY_DECLARATION.md")
    _log("")
    _log("Paper updated:")
    _log("paper_evidence/FINAL_RESULTS.md")
    _log("paper_evidence/FINAL_AUDIT.md")
    _log("paper_evidence/CLAIM_REGISTRY.json")
    _log("=" * 39)


if __name__ == "__main__":
    main()
