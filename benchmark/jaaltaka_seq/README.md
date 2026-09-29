# JaalTaka-Seq: sequential multi-view banknote authentication benchmark

**Task.** Each test note has 6 ordered close-up photographs. A policy looks at views one at a time, chooses which view to look at next, and decides when to stop. It then outputs P(genuine), or abstains.

The goal is to be right while looking at few views.

## Contents

| File | Purpose |
|---|---|
| `split.json` | Note-disjoint split (seed 42): train 974 / val 208 / test 208 notes, with verified zero overlap. The note ID prefix (`genuine:` / `counterfeit:`) is the label. |
| `evaluate.py` | Validates a submission and scores it: accuracy (with Wilson 95 % CI), coverage, selective accuracy, wrong-verdict rate, mean views, cost-adjusted accuracy (accuracy − 0.02 × mean views), ECE. |
| `make_baselines.py` | Rebuilds `split.json`, the baseline submissions in `baselines/` and the leaderboard from the project's saved predictions. |
| `LEADERBOARD.md`, `leaderboard.json` | 9 policies × {adaptive, forced 1 view, forced 6 views}, 3 seeds each |

## Submission format

See the `evaluate.py` docstring. It requires one entry per test note, with `views_used` in acquisition order and `genuine_prob` in [0, 1] or `null` to abstain.

**Rules**
- Use `train` to fit and `val` to choose everything (checkpoints, thresholds, costs).
- Score `test` once.
- Report mean and SD over at least 3 training seeds.

## Baselines

The 9 policies from the project, run on the prefix-robust PRMVT checkpoint with the fixed evaluation code:

1. fixed order
2. random
3. quality
4. confidence
5. diversity
6. quality + diversity
7. uncertainty + quality
8. uncertainty + diversity
9. the full CRIQP policy

**Current best** by cost-adjusted accuracy: always take the first view (fixed_order_1view), 0.9647 ± 0.0155 accuracy at 1.00 view. No learned selection policy beats the fixed order. This benchmark is open for methods that do.

## Data access

The JaalTaka images are **not included**. Their source and license are not documented in this repository (`paper_evidence/OPEN_SOURCE.md`), so they cannot be redistributed here. The split and the scorer are released so that anyone holding the images can reproduce the leaderboard.
