# Rejection in the authenticator

Two measured rules. They are not the same checkpoint.

## Single occlusion fine-tune, entropy quantile

Checkpoint `results/qduig/occlusion_ft/seed42/checkpoint.pt`. Median fill after occlusion 0.55. Threshold 0.4246242105960845, the 0.70 quantile of occluded validation entropy. Source: `results/robustness/occlusion_rejection_seed42.json`.

| quantity | value |
| --- | ---: |
| test accuracy, no rejection | 0.875 |
| test occluded rejection rate | 0.3221153846153846 |
| test occluded accepted accuracy | 0.9361702127659575 |
| test clean rejection rate | 0.125 |

Accepted error on occluded notes is 0.06382978723404253. This rule does not reach a 0.5% wrong-verdict share.

## Three-seed ensemble, confidence 0.99

Source: `results/robustness/occlusion_decision.json`. The threshold 0.99 is the largest value on a 0.50–0.99 grid that still answers at least 95% of clean validation notes. Test was not used to choose it.

| split | answered | accuracy on answered | wrong verdicts, share of all notes |
| --- | ---: | ---: | ---: |
| test clean | 0.9615384615384616 | 0.99 | 0.009615384615384616 |
| test occlusion 0.55 | 0.46634615384615385 | 0.9896907216494846 | 0.004807692307692308 |

On occluded test notes the ensemble is wrong on 0.004807692307692308 of all notes, which is 1 of 208, and it declines to answer the rest. Without rejection the same ensemble's occluded test accuracy is 0.8894230769230769, so the unanswered share is the difference between a verdict and a request to move the note.
