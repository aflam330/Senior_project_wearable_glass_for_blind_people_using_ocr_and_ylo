"""User-study analysis. Writes NOT MEASURED if no real rows exist.

Design (study_protocol.md): each participant does every task under both conditions
(baseline vs glass), so the participant is the independent unit and all tests are
paired. For every task x measure:
  1. average each participant's trials per condition,
  2. take the per-participant difference (glass - baseline),
  3. Wilcoxon signed-rank test (exact for n <= 25, zeros dropped), matched-pairs
     rank-biserial r as effect size,
  4. Holm correction over all task x measure tests.
Study-level SUS / NASA-TLX / satisfaction are summarised per condition and tested the
same way.

Usage:
  python analysis_script.py                                   # participant_data_template.csv
  python analysis_script.py --data my_trials.csv --out report.json
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "participant_data_template.csv"
OUT = HERE / "statistical_report.json"

TRIAL_MEASURES = ("task_time_s", "correct", "errors", "assistance_requests")
STUDY_MEASURES = ("nasa_tlx", "sus", "satisfaction")
# direction in which the glass is better, for the summary line
BETTER = {"task_time_s": "lower", "correct": "higher", "errors": "lower", "assistance_requests": "lower",
          "nasa_tlx": "lower", "sus": "higher", "satisfaction": "higher"}
BASELINE = "baseline"


def _num(x: str | None) -> float | None:
    x = (x or "").strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def wilcoxon_signed_rank(diffs: list[float]) -> dict:
    """Two-sided Wilcoxon signed-rank test. Exact null for n <= 25, normal approximation above."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return {"n_nonzero": 0, "W_plus": 0.0, "p": 1.0, "method": "all differences zero", "rank_biserial": 0.0}
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:  # average ranks for ties in |d|
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    total = n * (n + 1) / 2.0
    rank_biserial = (2.0 * w_plus - total) / total
    if n <= 25:
        # exact: enumerate all sign assignments of the (possibly tied) ranks
        stat = min(w_plus, total - w_plus)
        count = sum(1 for signs in itertools.product((0, 1), repeat=n)
                    if sum(r for r, s in zip(ranks, signs) if s) <= stat + 1e-9)
        p = min(1.0, 2.0 * count / 2 ** n)
        method = "exact"
    else:
        mean = total / 2.0
        var = n * (n + 1) * (2 * n + 1) / 24.0
        z = (w_plus - mean) / math.sqrt(var)
        p = math.erfc(abs(z) / math.sqrt(2.0))
        method = "normal approximation"
    return {"n_nonzero": n, "W_plus": w_plus, "p": p, "method": method, "rank_biserial": rank_biserial}


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def paired_test(per_participant: dict[str, dict[str, float]], treatment: str, measure: str) -> dict | None:
    pairs = [(v[BASELINE], v[treatment]) for v in per_participant.values() if BASELINE in v and treatment in v]
    if not pairs:
        return None
    base, glass = [b for b, _ in pairs], [g for _, g in pairs]
    diffs = [g - b for b, g in pairs]
    test = wilcoxon_signed_rank(diffs)
    return {
        "measure": measure,
        "n_participants": len(pairs),
        "baseline_median": statistics.median(base),
        "glass_median": statistics.median(glass),
        "median_difference": statistics.median(diffs),
        "better_when": BETTER.get(measure),
        **test,
    }


def analyse(rows: list[dict]) -> dict:
    conditions = sorted({r["condition"].strip() for r in rows})
    treatments = [c for c in conditions if c != BASELINE]
    if BASELINE not in conditions or len(treatments) != 1:
        raise SystemExit(f"need exactly two conditions, one named '{BASELINE}'; got {conditions}")
    treatment = treatments[0]
    participants = sorted({r["participant_id"].strip() for r in rows})

    tests = []
    # trial-level measures, per task
    for task in sorted({r["task"].strip() for r in rows}):
        for measure in TRIAL_MEASURES:
            acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for r in rows:
                v = _num(r.get(measure))
                if r["task"].strip() == task and v is not None:
                    acc[r["participant_id"].strip()][r["condition"].strip()].append(v)
            per = {p: {c: statistics.fmean(vs) for c, vs in cs.items()} for p, cs in acc.items()}
            t = paired_test(per, treatment, measure)
            if t:
                t["task"] = task
                tests.append(t)
    # study-level questionnaires (one value per participant x condition; averaged if repeated)
    for measure in STUDY_MEASURES:
        acc = defaultdict(lambda: defaultdict(list))
        for r in rows:
            v = _num(r.get(measure))
            if v is not None:
                acc[r["participant_id"].strip()][r["condition"].strip()].append(v)
        per = {p: {c: statistics.fmean(vs) for c, vs in cs.items()} for p, cs in acc.items()}
        t = paired_test(per, treatment, measure)
        if t:
            t["task"] = "questionnaire"
            tests.append(t)

    for t, p_adj in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = p_adj
        t["significant_holm_0.05"] = p_adj < 0.05
    return {
        "status": "MEASURED",
        "design": "within-subject, participant = unit, paired Wilcoxon signed-rank, Holm-corrected",
        "conditions": {"baseline": BASELINE, "treatment": treatment},
        "n_participants": len(participants),
        "n_trial_rows": len(rows),
        "tests": tests,
        "note": "With 8-12 participants only large effects can reach significance; report effect sizes too.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(TEMPLATE))
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    data, out = Path(args.data), Path(args.out)
    rows = list(csv.DictReader(data.open(encoding="utf-8")))
    filled = [r for r in rows if (r.get("task_time_s") or "").strip()]
    if not filled:
        out.write_text(json.dumps({"status": "NOT MEASURED",
                                   "reason": f"{data.name} has no completed trials"}, indent=2), encoding="utf-8")
        print("NOT MEASURED: no participant data")
        sys.exit(0)
    report = analyse(filled)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['n_participants']} participants, {len(report['tests'])} paired tests -> {out}")
    for t in report["tests"]:
        print(f"  {t['task']:14s} {t['measure']:20s} n={t['n_participants']:2d} "
              f"base={t['baseline_median']:.2f} glass={t['glass_median']:.2f} "
              f"p={t['p']:.4f} p_holm={t['p_holm']:.4f} r={t['rank_biserial']:+.2f}")


if __name__ == "__main__":
    main()
