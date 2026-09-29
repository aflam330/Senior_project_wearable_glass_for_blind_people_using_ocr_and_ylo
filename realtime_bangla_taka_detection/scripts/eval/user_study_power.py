"""Power analysis for the paired (within-subject) user study, exact noncentral-t computation.

Paired t-test, two-sided, alpha = 0.05. For n participants, the smallest effect dz (mean difference / sd
of differences) detectable with 80 % and 90 % power, and the power for dz = 0.2, 0.3, 0.5.
A Wilcoxon signed-rank test needs about n / 0.955 participants for the same power under normality (ARE).
Output: results/user_study/power.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[2]


def power(n: int, dz: float, alpha: float = 0.05) -> float:
    df = n - 1
    tc = stats.t.ppf(1 - alpha / 2, df)
    nc = dz * np.sqrt(n)
    upper = stats.nct.sf(tc, df, nc)
    lower = stats.nct.cdf(-tc, df, nc)
    # scipy returns NaN far in the tails: there the upper tail is ~1 and the lower tail ~0
    upper = 1.0 if np.isnan(upper) else upper
    lower = 0.0 if np.isnan(lower) else lower
    return float(upper + lower)


def main() -> None:
    out = {}
    for n in (20, 50, 105):
        out[str(n)] = {"min_dz_80": float(optimize.brentq(lambda d: power(n, d) - 0.8, 0.01, 1.5)),
                       "min_dz_90": float(optimize.brentq(lambda d: power(n, d) - 0.9, 0.01, 1.5)),
                       "power_at_dz": {str(d): power(n, d) for d in (0.2, 0.3, 0.5)}}
    out["n_for_80pct_at_dz_0.3"] = int(next(n for n in range(5, 1000) if power(n, 0.3) >= 0.8))
    out["wilcoxon_note"] = "Wilcoxon needs about n / 0.955 for the same power under normality (asymptotic relative efficiency)."
    dst = ROOT / "results" / "user_study"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "power.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
