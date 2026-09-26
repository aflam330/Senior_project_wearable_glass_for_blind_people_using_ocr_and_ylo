"""Mean, SD, and paired tests from saved seed-42/43/44 metrics. No new numbers are invented."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from roboeye.qduig.metrics_ext import mcnemar, wilcoxon_signed
PAPER = ROOT.parent / "paper_evidence"
ALGOS = {
    "prmvt": None,
    "ndal": ROOT / "results" / "novel_v2" / "ndal",
    "pravt": ROOT / "results" / "novel_v2" / "pravt",
    "vat": ROOT / "results" / "novel_v2" / "vat",
    "ugf": ROOT / "results" / "novel_v2" / "ugf",
    "cvs": ROOT / "results" / "novel_v2" / "cvs",
}


def _prmvt(seed: int) -> dict[int, float] | None:
    path = ROOT / "results" / "qduig" / "prefix_ft" / f"seed{seed}" / "test_views" / "views_1_to_6.json"
    if seed == 42 and not path.is_file():
        path = ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "test_views" / "views_1_to_6.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("proposed") or payload.get("views") or []
    return {int(row["k"]): float(row["accuracy"]) for row in rows}


def _novel(root: Path, seed: int) -> dict[int, float] | None:
    candidates = [root / f"seed{seed}" / "test" / "test_metrics.json"]
    if seed == 42:
        candidates.append(ROOT / "results" / "novel" / root.name / "seed42" / "test" / "test_metrics.json")
    for path in candidates:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {int(row["k"]): float(row["accuracy"]) for row in payload["views"]}
    return None


def _load(algo: str, seed: int) -> dict[int, float] | None:
    if algo == "prmvt":
        return _prmvt(seed)
    return _novel(ALGOS[algo], seed)


def _mean_sd(values: list[float]) -> tuple[str, str, str]:
    if not values:
        return "NOT_MEASURED", "NOT_MEASURED", "NOT_MEASURED"
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    if len(arr) > 1 and math.isfinite(sd):
        half = 1.96 * sd / math.sqrt(len(arr))
        ci = f"[{mean - half:.4f}, {mean + half:.4f}]"
    else:
        ci = "NOT_MEASURED"
    sd_s = "NOT_MEASURED" if not math.isfinite(sd) else f"{sd:.4f}"
    return f"{mean:.4f}", sd_s, ci


def _cohen_h(p1: float, p2: float) -> float:
    return float(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))


def _bonferroni(p: float, n_tests: int) -> float:
    return min(1.0, p * n_tests)


def _fmt_p(p) -> str:
    if p is None:
        return "NOT_MEASURED"
    p = float(p)
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _pair(algo: str, seed: int, k: int):
    base_path = ROOT / "results" / "qduig" / "eval" / "seed42" / "baseline" / f"baseline_{k}view" / "test_predictions.json"
    if algo == "prmvt":
        algo_path = ROOT / "results" / "qduig" / "prefix_ft" / f"seed{seed}" / "test_views" / f"{k}view" / "test_predictions.json"
        algo_blob = json.loads(algo_path.read_text(encoding="utf-8")) if algo_path.is_file() else None
    else:
        candidates = [ALGOS[algo] / f"seed{seed}" / "test" / "test_predictions.json"]
        if seed == 42:
            candidates.append(ROOT / "results" / "novel" / algo / "seed42" / "test" / "test_predictions.json")
        algo_blob = None
        for path in candidates:
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                algo_blob = payload.get(str(k))
                break
    if algo_blob is None or not base_path.is_file():
        return None
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base_map = {note: (int(y), float(p)) for note, y, p in zip(base["note_id"], base["y_true"], base["genuine_score"])}
    ys, pa, pb = [], [], []
    for note, y, p in zip(algo_blob["note_id"], algo_blob["y_true"], algo_blob["genuine_score"]):
        if note not in base_map:
            continue
        by, bp = base_map[note]
        if int(y) != by:
            continue
        ys.append(by)
        pa.append(int(bp >= 0.5))
        pb.append(int(float(p) >= 0.5))
    if len(ys) < 10:
        return None
    return np.array(ys), np.array(pa), np.array(pb)


def _bootstrap_diff(y, pred_a, pred_b, draws: int = 1000) -> str:
    rng = np.random.default_rng(42)
    n = len(y)
    diffs = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        diffs[i] = (pred_b[idx] == y[idx]).mean() - (pred_a[idx] == y[idx]).mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return f"[{float(lo):.4f}, {float(hi):.4f}]"


def main() -> None:
    lines = [
        "# Multi-seed results",
        "",
        "Seeds 42, 43, and 44. A missing seed is NOT_MEASURED. The interval is a normal approximation from the seed-level accuracies, not a paired bootstrap of notes.",
        "",
        "| algorithm | views | n seeds | mean | sd | approx 95% CI |",
        "|---|---:|---:|---:|---:|---|",
    ]
    stat_lines = [
        "# Statistical analysis",
        "",
        "Cohen's h compares each available seed-42 accuracy with the CNN+ViT baseline at the same view count.",
        "Wilcoxon and McNemar on three seed-level numbers are not reported: n=3 is too small for a rank test to mean anything.",
        "Per-note McNemar for PRMVT versus the baseline at 6 views remains the seed-42 result already in `FINAL_RESULTS.md` (n01=3, n10=13, p=0.0244).",
        "Bonferroni uses one family: 6 algorithms × 6 view counts = 36 Cohen's h comparisons. Those comparisons are descriptive; they are not 36 independent hypothesis tests with a pre-registered null.",
        "",
        "| algorithm | views | seed42 | baseline | Cohen h |",
        "|---|---:|---:|---:|---:|",
    ]
    baseline = {}
    for k in range(1, 7):
        path = ROOT / "results" / "qduig" / "eval" / "seed42" / "baseline" / f"baseline_{k}view" / "test_metrics.json"
        baseline[k] = float(json.loads(path.read_text(encoding="utf-8"))["accuracy"]) if path.is_file() else None
    n_tests = 6 * 6
    for algo in ALGOS:
        for k in range(1, 7):
            vals = []
            seed42 = None
            for seed in (42, 43, 44):
                accs = _load(algo, seed)
                if accs is None or k not in accs:
                    continue
                vals.append(accs[k])
                if seed == 42:
                    seed42 = accs[k]
            mean, sd, ci = _mean_sd(vals)
            lines.append(f"| {algo} | {k} | {len(vals) if vals else 'NOT_MEASURED'} | {mean} | {sd} | {ci} |")
            if seed42 is None or baseline.get(k) is None:
                stat_lines.append(f"| {algo} | {k} | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |")
            else:
                h = _cohen_h(seed42, baseline[k])
                stat_lines.append(f"| {algo} | {k} | {seed42:.4f} | {baseline[k]:.4f} | {h:.4f} |")
    lines.append("")
    lines.append("The interval above uses the seed-level accuracies. Per-note paired bootstrap intervals are in `STATISTICAL_ANALYSIS.md`.")
    lines.append("")
    tests = []
    for algo in ALGOS:
        for k in range(1, 7):
            # Seed 42 is the paired comparison against the saved baseline predictions.
            # Seeds 43 and 44 change the training seed, not the test-note pairing.
            paired = _pair(algo, 42, k)
            if paired is None:
                tests.append((algo, k, None))
                continue
            y, pred_base, pred_algo = paired
            mc = mcnemar(y, pred_base, pred_algo)
            correct_base = (pred_base == y).astype(np.float64)
            correct_algo = (pred_algo == y).astype(np.float64)
            wil = wilcoxon_signed(correct_base, correct_algo)
            tests.append((algo, k, {
                "mc": mc,
                "wil": wil,
                "boot": _bootstrap_diff(y, pred_base, pred_algo),
                "n": int(len(y)),
            }))
    family = [row for row in tests if row[2] is not None]
    m = max(len(family), 1)
    stat_lines.append("")
    stat_lines.append("Paired tests use the same test notes. `a` is the CNN+ViT baseline and `b` is the named model. McNemar uses the continuity correction. The paired bootstrap resamples notes 1000 times with seed 42. Wilcoxon is on the paired 0/1 correctness vectors and drops ties.")
    stat_lines.append("")
    stat_lines.append("| algorithm | views | n | McNemar p | Bonferroni p | bootstrap 95% CI of accuracy difference | Wilcoxon p |")
    stat_lines.append("|---|---:|---:|---:|---:|---|---:|")
    for algo, k, row in tests:
        if row is None:
            stat_lines.append(f"| {algo} | {k} | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |")
            continue
        p = row["mc"]["p_value"]
        wp = row["wil"].get("p_value")
        stat_lines.append(
            f"| {algo} | {k} | {row['n']} | {_fmt_p(p)} | {_fmt_p(_bonferroni(p, m))} | {row['boot']} | {_fmt_p(wp)} |"
        )
    stat_lines.append("")
    stat_lines.append(f"Bonferroni family size is the number of McNemar rows that had paired predictions: {len(family)}.")
    stat_lines.append("")
    (PAPER / "MULTI_SEED_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (PAPER / "STATISTICAL_ANALYSIS.md").write_text("\n".join(stat_lines) + "\n", encoding="utf-8")
    print(PAPER / "MULTI_SEED_RESULTS.md")


if __name__ == "__main__":
    main()
