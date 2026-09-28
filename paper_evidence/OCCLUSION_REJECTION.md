# Occlusion uncertainty rejection

Checkpoint `results/qduig/occlusion_ft/seed42/checkpoint.pt`. Occlusion 0.55 with median fill. A note is rejected when the predicted entropy is above a threshold. The threshold is the 0.70 quantile of occluded validation entropy, because that quantile was the one that kept the most validation notes while accepted validation accuracy stayed at least 0.90. Test labels were not used to choose it. Source: `results/robustness/occlusion_rejection_seed42.json`.

| quantity | value |
| --- | ---: |
| validation threshold | 0.4246242105960845 |
| validation coverage | 0.6971153846153846 |
| validation accepted accuracy | 0.9241379310344827 |
| test accuracy with no rejection | 0.875 |
| test occluded coverage | 0.6778846153846154 |
| test occluded rejection rate | 0.3221153846153846 |
| test occluded accepted accuracy | 0.9361702127659575 |
| test clean rejection rate | 0.125 |
| test clean accepted accuracy | 0.9285714285714286 |

Accepted occluded accuracy is above 0.90. Accuracy on all 208 test notes is still 0.875. The gain comes from refusing 0.3221153846153846 of the occluded notes, and the same rule also refuses 0.125 of the clean notes. This is not a 0.90 classifier on the full test set.

---

## Update 2026-09-28: rejection on the occlusion-robust ensemble

Model: the mean of the three occlusion-robust seeds (see `OCCLUSION_FINETUNE.md` and `OCCLUSION_ENSEMBLE.md`). Rule fixed before looking at test: answer only when confidence max(p, 1−p) ≥ t, where t is the largest value on a 0.50–0.99 grid that still answers ≥ 95% of clean validation notes. Otherwise say "please reposition the note". The chosen threshold is **t = 0.99**. Raw: `results/robustness/occlusion_decision.json`.

| Split / condition | Notes answered | Accuracy on answered | Wrong verdicts, share of all notes |
|---|---:|---:|---:|
| val clean | 99.0% | 99.0% | 1.0% |
| val 55% occlusion | 46.2% | 97.9% | 1.0% |
| **test clean** | **96.2%** | **99.0%** | **1.0%** |
| **test 55% occlusion** | **46.6%** | **99.0%** | **0.5%** |

Without rejection the same ensemble gives a wrong verdict on 11.1% of occluded test notes. With rejection it gives a wrong verdict on 0.5% and asks for a better view on the other half. For a blind user, "move the note" is recoverable; a wrong "genuine" is not.
