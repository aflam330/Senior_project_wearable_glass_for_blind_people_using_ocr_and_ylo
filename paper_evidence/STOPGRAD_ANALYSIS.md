# Stop-gradient 6-view drop

`her_base` test accuracy is 0.9903846153846154 at 1 view and 0.9182692307692307 at 6 views (`results/qduig/ablation_prefix/her_base/seed42/test_views/views_1_to_6.json`).

The stop-gradient fine-tune started from that checkpoint and kept training the authenticator with the authentication loss for 2 epochs. Auxiliary quality and uncertainty losses were nonzero. The epoch was chosen by validation mean accuracy over 1–6 views.

`results/qduig/auxfix_stopgrad/seed42/test_views/views_1_to_6.json`

| views | accuracy |
|---:|---:|
| 1 | 0.9903846153846154 |
| 2 | 0.9375 |
| 3 | 0.9182692307692307 |
| 4 | 0.9086538461538461 |
| 5 | 0.9326923076923077 |
| 6 | 0.9086538461538461 |

1-view stayed at 0.9903846153846154. 6-view fell by 0.009615384615384582. Both saved epochs had validation 1-view 0.9807692307692307. Epoch 2 had the higher validation mean (0.923076923076923 versus 0.889423076923077), so mean-over-views selection kept the epoch that was trained longer. The extra authentication updates are the change between `her_base` and this checkpoint.

Leaving the authenticator fixed removes that update. The separate tower run with batch-norm frozen did not change the authenticator. Its test accuracies match `her_base` at every view count, including 6-view 0.9182692307692307, while the tower losses stay nonzero (`results/qduig/auxfix_separate_frozen/seed42/test_views/views_1_to_6.json`).

That meets 1-view ≥ 0.9903846153846154 and 6-view ≥ 0.9182692307692307. The 2-epoch stop-gradient continuation does not meet the 6-view bar.

Verdict: PASS by not updating the authenticator. The continuation itself is a negative result at 6 views.
