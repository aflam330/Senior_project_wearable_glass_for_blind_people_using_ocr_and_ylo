"""Score System Usability Scale (SUS) and NASA-TLX questionnaires from the user study.

Input CSV (one row per participant x condition):
  participant, condition, sus1..sus10 (1-5), tlx_mental, tlx_physical, tlx_temporal, tlx_performance,
  tlx_effort, tlx_frustration (0-100), and optionally w_<subscale> pairwise-comparison weights (0-5, summing to 15).
SUS  = 2.5 * (sum over odd items of (x - 1) + sum over even items of (5 - x))           (Brooke 1996)
RTLX = mean of the six subscales; weighted TLX = sum(rating * weight) / 15 when weights are given (Hart & Staveland 1988)
Per condition: mean, sd, n. With two conditions and paired participants: Wilcoxon signed-rank test.
Run with no file: prints DATA_NOT_COLLECTED and runs the self-checks.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from statistics import mean, stdev

TLX = ["mental", "physical", "temporal", "performance", "effort", "frustration"]


def sus(items: list[float]) -> float:
    if len(items) != 10 or not all(1 <= x <= 5 for x in items):
        raise ValueError("SUS needs 10 answers on 1-5")
    return 2.5 * sum((x - 1) if i % 2 == 0 else (5 - x) for i, x in enumerate(items))


def tlx(r: dict[str, float], w: dict[str, float] | None = None) -> tuple[float, float | None]:
    raw = mean(r[k] for k in TLX)
    weighted = sum(r[k] * w[k] for k in TLX) / 15 if w and abs(sum(w.values()) - 15) < 1e-9 else None
    return raw, weighted


def self_check() -> None:
    assert sus([3] * 10) == 50.0                    # all neutral
    assert sus([5, 1] * 5) == 100.0                 # best possible
    assert sus([1, 5] * 5) == 0.0                   # worst possible
    raw, wt = tlx({k: 50 for k in TLX}, {"mental": 5, "physical": 0, "temporal": 3, "performance": 2, "effort": 4, "frustration": 1})
    assert raw == 50 and wt == 50
    print("self-checks passed")


def main() -> None:
    self_check()
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("user_study_questionnaires.csv")
    if not path.is_file():
        print("DATA_NOT_COLLECTED", path)
        return
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by = {}
    for r in rows:
        s = sus([float(r[f"sus{i}"]) for i in range(1, 11)])
        w = {k: float(r[f"w_{k}"]) for k in TLX} if all(f"w_{k}" in r and r[f"w_{k}"] for k in TLX) else None
        raw, wt = tlx({k: float(r[f"tlx_{k}"]) for k in TLX}, w)
        by.setdefault(r["condition"], {})[r["participant"]] = (s, raw, wt)
    for c, d in by.items():
        s = [v[0] for v in d.values()]
        t = [v[1] for v in d.values()]
        print(f"{c}: n={len(s)} SUS {mean(s):.1f} ± {stdev(s) if len(s) > 1 else 0:.1f}  RTLX {mean(t):.1f} ± {stdev(t) if len(t) > 1 else 0:.1f}")
    if len(by) == 2:
        from scipy.stats import ttest_rel, wilcoxon
        a, b = list(by)
        common = sorted(set(by[a]) & set(by[b]))
        if len(common) >= 6:
            for i, name in ((0, "SUS"), (1, "RTLX")):
                xa = [by[a][p][i] for p in common]
                xb = [by[b][p][i] for p in common]
                st = wilcoxon(xa, xb)
                tt = ttest_rel(xa, xb)
                d = [u - v for u, v in zip(xa, xb)]
                dz = mean(d) / stdev(d) if stdev(d) > 0 else float("nan")
                print(f"{name} {a} vs {b}: n={len(common)} Wilcoxon p={st.pvalue:.4g}  paired t p={tt.pvalue:.4g}  dz={dz:.2f}")


if __name__ == "__main__":
    main()
