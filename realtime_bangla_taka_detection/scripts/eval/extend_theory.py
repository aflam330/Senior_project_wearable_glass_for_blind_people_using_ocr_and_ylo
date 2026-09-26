"""Append numerical bounds that can be computed from saved counts. No fitted KL."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT.parent / "paper_evidence"
MARK = "## 5. Numerical bounds computed from the split"


def main() -> None:
    meta = json.loads((ROOT / "results" / "camva" / "splits" / "split_metadata.json").read_text(encoding="utf-8"))
    n_train = int(meta["train"]["n_notes"])
    text = (PAPER / "THEORETICAL_ANALYSIS.md").read_text(encoding="utf-8")
    if MARK in text:
        text = text.split(MARK)[0].rstrip() + "\n"
    eps, delta = 0.05, 0.05
    n_need = math.ceil(math.log(2 / delta) / (2 * eps * eps))
    # Measured fixed-count gap, read from the paper file only if the json exists.
    fixed = ROOT / "results" / "qduig" / "eval" / "seed42"
    one = six = None
    # The pre-prefix 1-view collapse is stored in FINAL_RESULTS as 0.5865 / 0.9663.
    # Prefer a json if one is present under proposed seed42.
    proposed = ROOT / "results" / "qduig" / "proposed" / "seed42" / "test_views" / "views_1_to_6.json"
    if proposed.is_file():
        rows = {int(r["k"]): float(r["accuracy"]) for r in json.loads(proposed.read_text(encoding="utf-8")).get("proposed", [])}
        one, six = rows.get(1), rows.get(6)
    block = f"""
{MARK}

These numbers are consequences of the stated inequalities and the split size. They are not a fit to the test accuracy.

### Sample complexity for a fixed predictor

Assume notes are i.i.d. and the predictor is chosen before seeing them. Hoeffding's inequality gives
\\[
n \\ge \\frac{{\\ln(2/\\delta)}}{{2\\varepsilon^2}}
\\]
for an additive accuracy deviation of \\(\\varepsilon\\) with probability at least \\(1-\\delta\\).

For \\(\\varepsilon=0.05\\) and \\(\\delta=0.05\\), that expression equals **{n_need}** notes. The training split has **{n_train}** notes (`split_metadata.json`). {n_train} is {"at least" if n_train >= n_need else "below"} this fixed-predictor count. This does not bound prefix-robust SGD. The hypothesis class of the network was not measured, so a uniform-convergence sample size for the training algorithm is NOT_MEASURED.

Fixed-view training and prefix-robust training use the same notes. The difference is the support of \\(K\\), not a smaller \\(n\\). No claim is made that prefix-robust training needs fewer notes.

### PAC-Bayes

The McAllester form in section 4 needs \\(\\mathrm{{KL}}(\\rho\\|\\pi)\\). That KL was not estimated for any posterior over these weights. A numerical PAC-Bayes value is therefore NOT_MEASURED. Substituting \\(n={n_train}\\) without a KL would not be a bound.

### Oracle gap, measured

"""
    if one is not None and six is not None:
        block += (
            f"The fixed-count checkpoint file `{proposed.relative_to(ROOT)}` has 1-view accuracy {one:.4f} "
            f"and 6-view accuracy {six:.4f}. The difference is {six - one:.4f}. "
            "That is an empirical gap on this split, not a proved lower bound.\n"
        )
    else:
        block += (
            "A separate fixed-count `views_1_to_6.json` was not found under `results/qduig/proposed/seed42`. "
            "The 1-view collapse cited in `FINAL_RESULTS.md` is not recomputed here. "
            "A numerical lower bound tighter than the proposition in section 3 is NOT_MEASURED.\n"
        )
    (PAPER / "THEORETICAL_ANALYSIS.md").write_text(text + block, encoding="utf-8")
    print(n_train, n_need, one, six)


if __name__ == "__main__":
    main()
