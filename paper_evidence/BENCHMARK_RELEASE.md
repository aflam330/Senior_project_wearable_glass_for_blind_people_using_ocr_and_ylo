# JaalTaka-Seq benchmark package (2026-09-29)

**Location:** `benchmark/jaaltaka_seq/`. It supersedes the table-only `JAALTAKA_SEQ_BENCHMARK.md`.

| Part | Status | Evidence |
|---|---|---|
| Dataset JSON, note-disjoint | Done | `split.json`: 974 / 208 / 208 notes, overlap asserted zero by `make_baselines.py` |
| 9 policy baselines | Done | `baselines/*.json`: 9 policies × 3 settings × seeds 42–44, from `results/qduig/eval_prefix_ft/seed*/policies/*/test_predictions.json` (post-fix code) |
| Metrics per policy | Done | `leaderboard.json`, `LEADERBOARD.md` |
| Submission + evaluation | Done | `evaluate.py` validates coverage, view indices and probabilities |
| Documentation | Done | `benchmark/jaaltaka_seq/README.md` |
| Public hosting | **Not done** | JaalTaka images have no documented source or license; nothing was uploaded |

## Headline

- The fixed-order single view is best by cost-adjusted accuracy: **0.9647 (SD 0.0155)** at 1.00 view.
- Taking all six views gives **0.9808 (SD 0.0096)**.
- The learned full policy stops at 1.00 view with **0.8926 (SD 0.0474)**.
- All other "adaptive" policies never stop early. They take 6.00 views.

These match `WEAK_RESULTS_FIX.md`, computed independently from the same prediction files.
