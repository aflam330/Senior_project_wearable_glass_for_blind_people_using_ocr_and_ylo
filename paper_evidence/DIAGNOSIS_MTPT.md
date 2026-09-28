# Diagnosis: MTPT

Target is 1-view test accuracy at least 0.9711538461538461, the PRMVT prefix result. Standing v2 is 0.9519230769230769.

## Checkpoint

`results/novel_v2/mtpt/seed42/test/test_metrics.json`, seed 42, n=208.

| views | accuracy |
| ---: | ---: |
| 1 | 0.9519230769230769 |
| 2 | 0.9230769230769231 |
| 3 | 0.9375 |
| 4 | 0.9326923076923077 |
| 5 | 0.9230769230769231 |
| 6 | 0.9134615384615384 |

## What was already changed

Setting the auxiliary-task weight to 0 for 6 epochs and selecting by mean validation accuracy produced test 1-view 0.9471153846153846 (`results/novel_v2/mtpt_auth/seed42/test/test_metrics.json`). That is worse than v2. The auxiliary loss is not the limiter.

PRMVT's published schedule is 6 epochs at learning rate 0.001, then 3 epochs at 0.0003, with mixed prefix dropout, and the checkpoint is the best mean over views. MTPT v2 was not trained with that 6+3 schedule. Denomination and emotion labels are not in JaalTaka, so those heads are not a measured task.

## Root cause

This is a training-schedule gap relative to PRMVT, plus mean-based checkpoint selection. It is not class imbalance and not an auxiliary-weight failure. Gradient norms and feature drift were not logged for MTPT. A 1-view-selected 6+3 run was not in the saved logs.
