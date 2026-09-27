# Fix 6: separate auxiliary encoder

Seed 42. The start checkpoint is `her_base`. A three-head linear tower (quality, diversity, info-gain) trains on detached pooled features. The authenticator parameters are frozen. Inference uses the authenticator only. The tower weights are not saved in the checkpoint.

## First run: batch-norm buffers still updated

`results/qduig/auxfix_separate/seed42/test_views/views_1_to_6.json`

| views | accuracy |
|---:|---:|
| 1 | 0.9519230769230769 |
| 2 | 0.9711538461538461 |
| 3 | 0.9134615384615384 |
| 4 | 0.9086538461538461 |
| 5 | 0.9471153846153846 |
| 6 | 0.9134615384615384 |

1-view is below 0.9711538461538461. The view-count batch-norm was left in train mode, so its running statistics moved even though the weights were frozen.

## Corrected run: batch-norm frozen

`results/qduig/auxfix_separate_frozen/seed42/test_views/views_1_to_6.json`

| views | accuracy |
|---:|---:|
| 1 | 0.9903846153846154 |
| 2 | 0.9711538461538461 |
| 3 | 0.9471153846153846 |
| 4 | 0.9086538461538461 |
| 5 | 0.9759615384615384 |
| 6 | 0.9182692307692307 |

These six numbers match `her_base` (`results/qduig/ablation_prefix/her_base/seed42/test_views/views_1_to_6.json`).

Last-epoch auxiliary terms on the corrected run (`results/qduig/auxfix_separate_frozen/seed42/val_metrics.json`): quality 0.08368236527183462, diversity 0.009134438387988407, info-gain 0.014033642198768057. Uncertainty and cost are 0. The authentication loss is logged (0.10758475605381683) and does not update the authenticator.

Verdict on the corrected run: SUCCESS at 1 view, and the other view counts match `her_base`.
