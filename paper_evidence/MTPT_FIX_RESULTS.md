# MTPT

Target was 1-view accuracy at least 0.9711538461538461. The standing v2 1-view accuracy is 0.9519230769230769 (`results/novel_v2/mtpt/seed42/test/test_metrics.json`). Denomination and emotion labels are still not in JaalTaka, so those tasks remain NOT_MEASURED.

## Authenticity loss only

The proxy-quality weight was set to 0 for 6 epochs. The checkpoint is the best validation mean over 1–6 views (epoch 6, validation 1-view 0.9567307692307693).

Test: `results/novel_v2/mtpt_auth/seed42/test/test_metrics.json`

| views | accuracy |
|---:|---:|
| 1 | 0.9471153846153846 |
| 2 | 0.9326923076923077 |
| 3 | 0.9471153846153846 |
| 4 | 0.9375 |
| 5 | 0.9182692307692307 |
| 6 | 0.9278846153846154 |

1-view is below v2 and below 0.9711538461538461. This run does not replace v2. PCGrad, Kendall weighting, and per-task layer norms were not trained after this run missed the target.

Verdict: FAIL. v2 remains the MTPT result.
