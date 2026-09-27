# Normalized entropy cost

The unnormalized entropy cost, trained from scratch, scored 0.9519230769230769 at 1 view. The matched run with that term removed scored 0.9615384615384616. Source: `ABLATION_RESULTS.md`.

This attempt resumes the matched checkpoint for 2 epochs. Predictive entropy is divided by ln 2, and λ_c is 0.01. The saved PRMVT checkpoint is not overwritten. Test, seed 42, n=208. Source: `results/qduig/cost_norm_ft/seed42/test_views/views_1_to_6.json`.

| views | accuracy |
|---:|---:|
| 1 | 0.9615384615384616 |
| 2 | 0.9711538461538461 |
| 3 | 0.9807692307692307 |
| 4 | 0.9855769230769231 |
| 5 | 0.9807692307692307 |
| 6 | 0.9903846153846154 |

1-view accuracy equals the matched run without the cost term. The normalized term does not fall below 0.9615384615384616 on this split.

Checkpoint selection used validation mean accuracy over 1–6 views. The test split was not used to choose the epoch.
