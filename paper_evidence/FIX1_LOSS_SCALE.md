# Fix 1: loss scale

Seed 42. Each run fine-tunes the saved full PRMVT checkpoint for 2 epochs. The checkpoint is chosen by validation mean accuracy over 1–6 views. Test labels are not used for that choice. Multitask weight stays 0.5 and contrastive weight stays 0.05. Only the auxiliary loss weights change.

Full PRMVT 1-view is 0.9711538461538461. `her_base` 1-view is 0.9903846153846154.

| config | λ scale | 1-view | 6-view | verdict |
| --- | --- | ---: | ---: | --- |
| auxfix_scale_0p1 | 0.1 | 0.9663461538461539 | 0.9759615384615384 | FAIL |
| auxfix_scale_0p01 | 0.01 | 0.9567307692307693 | 0.9663461538461539 | FAIL |
| auxfix_scale_0p001 | 0.001 | 0.9663461538461539 | 0.9759615384615384 | FAIL |
| auxfix_scale_0p0001 | 0.0001 | 0.9519230769230769 | 0.9663461538461539 | FAIL |
| auxfix_scale_divcost | diversity 0.001, cost 0.001, other auxiliary weights unchanged | 0.9519230769230769 | 0.9663461538461539 | FAIL |

An earlier 2-epoch fine-tune that also divided the multitask and contrastive weights by 10 scored 0.9663461538461539 at 1 view (`results/qduig/aux_small_ft/seed42/test_views/views_1_to_6.json`).

None of these 1-view scores reach 0.9711538461538461. Shrinking the auxiliary weights, including a diversity-and-cost-only reduction, does not stop the drop.

Artifacts: `results/qduig/auxfix_scale_*/seed42/test_views/views_1_to_6.json`.
