# Ablation results

Seed 42, test split, n=208. Forced first-k views. Full is the saved prefix-robust checkpoint.
The earlier 4-epoch runs in `results/qduig/ablations/` are unchanged and are not this table.
no_cost and no_calibration matched that 4-epoch full run because those switches do not change forced-view training. That cause is recorded here instead of training a duplicate. A later matched pair, `cost_entropy` and `no_cost_matched`, adds mean predictive entropy to the cost so the term has a gradient. Those two rows are new trainings. They do not replace `full_prmvt`.

| config | 1 | 2 | 3 | 4 | 5 | 6 | note |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.7356 | 0.8702 | 0.9135 | 0.9183 | 0.8990 | 0.9183 | existing CNN+ViT, not retrained |
| rsqa | 0.9327 | 0.9038 | 0.8365 | 0.7356 | 0.7548 | 0.8029 | quality loss only, prefix 6+3 1-view minus full: -0.0385 |
| cvr | 0.9808 | 0.9808 | 0.9808 | 0.9808 | 0.9808 | 0.9808 | diversity and redundancy only, prefix 6+3 1-view minus full: +0.0096 |
| her_base | 0.9904 | 0.9712 | 0.9471 | 0.9087 | 0.9760 | 0.9183 | prefix encoder, auxiliary losses off; HER is an eval map, see CALIBRATION_RESULTS.md 1-view minus full: +0.0192 |
| pcr_ig | 0.9712 | 0.9663 | 0.9808 | 0.9808 | 0.9904 | 0.9856 | information-gain loss only, prefix 6+3 1-view minus full: +0.0000 |
| qd | 0.9712 | 0.9760 | 0.9712 | 0.9423 | 0.9663 | 0.9663 | quality plus diversity, prefix 6+3 1-view minus full: +0.0000 |
| ud | 0.9808 | 0.9856 | 0.9856 | 0.9808 | 0.9808 | 0.9856 | uncertainty plus diversity, prefix 6+3 1-view minus full: +0.0096 |
| full_prmvt | 0.9712 | 0.9760 | 0.9760 | 0.9760 | 0.9808 | 0.9760 | existing prefix_ft seed 42, 6+3 epochs, not retrained |
| no_cost | 0.9712 | 0.9760 | 0.9760 | 0.9760 | 0.9808 | 0.9760 | same forced-view weights as full. The policy eval is separate: with lambda 0.02, accuracy 0.4135 at 1.00 view; with lambda 0, accuracy 0.4231 at 2.22 views. Source: `results/qduig/prefix_ft/seed42/cost_policy_test.json`. |
| no_calibration | 0.9712 | 0.9760 | 0.9760 | 0.9760 | 0.9808 | 0.9760 | same uncalibrated weights as full, so forced-view accuracy matches. At 6 views, threshold 0.5, HER fit on validation changes test accuracy from 0.9759615384615384 to 0.9711538461538461. Temperature scaling and the entropy map stay at 0.9759615384615384. Source: `results/calibration/accuracy_seed42.json`. |
| no_redundancy | 0.9327 | 0.9375 | 0.9423 | 0.9135 | 0.9375 | 0.9135 | PRMVT recipe with redundancy off 1-view minus full: -0.0385 |
| no_infogain | 0.9327 | 0.9615 | 0.9615 | 0.9375 | 0.9279 | 0.9375 | PRMVT recipe with information-gain loss off 1-view minus full: -0.0385 |
| cost_entropy | 0.9519 | 0.9712 | 0.9808 | 0.9808 | 0.9808 | 0.9808 | New seed-42 prefix 6+3 run. Cost loss adds mean predictive entropy, so the term has a gradient. Not a replacement for full_prmvt. 1-view minus no_cost_matched: -0.0096 |
| no_cost_matched | 0.9615 | 0.9712 | 0.9856 | 0.9856 | 0.9856 | 0.9808 | Same recipe as cost_entropy with the cost term removed. 1-view minus cost_entropy: +0.0096 |
