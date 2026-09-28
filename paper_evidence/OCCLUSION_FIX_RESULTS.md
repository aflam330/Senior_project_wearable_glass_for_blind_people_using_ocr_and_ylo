# Occlusion 0.55

Target was 0.90. It was not reached.

The occlusion draw matches the earlier severity sweep: `np.random.seed(1000 + start + i)` once per test note, then a black box of area fraction 0.55 on each view. Identity on the saved full PRMVT checkpoint scores 0.5625 at 6 views, the same number as `results/robustness/severity_seed42.json`.

## Already trained checkpoints

`results/robustness/occlusion_ft_seed42.json`: occlusion fine-tune, 6-view occlusion 0.6682692307692307. Clean 1-view 0.8461538461538461. Clean 6-view 0.8990384615384616.

## Inference repairs

No training. Source: `results/robustness/occlusion_repairs_seed42.json`. Lists are views 1 through 6.

| model | repair | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix_ft | identity | 0.5096153846153846 | 0.5625 | 0.6586538461538461 | 0.5528846153846154 | 0.5576923076923077 | 0.5625 |
| prefix_ft | inpaint | 0.6730769230769231 | 0.6538461538461539 | 0.6778846153846154 | 0.5673076923076923 | 0.6057692307692307 | 0.6634615384615384 |
| prefix_ft | median fill | 0.5769230769230769 | 0.5721153846153846 | 0.5817307692307693 | 0.5528846153846154 | 0.5240384615384616 | 0.5384615384615384 |
| occlusion_ft | identity | 0.6971153846153846 | 0.6586538461538461 | 0.6682692307692307 | 0.6538461538461539 | 0.6105769230769231 | 0.6682692307692307 |
| occlusion_ft | inpaint | 0.6298076923076923 | 0.6009615384615384 | 0.6634615384615384 | 0.6586538461538461 | 0.5961538461538461 | 0.6057692307692307 |
| occlusion_ft | median fill | 0.7596153846153846 | 0.8173076923076923 | 0.8365384615384616 | 0.8269230769230769 | 0.8317307692307693 | 0.875 |

The highest 6-view score is 0.875, from median-fill on the occlusion fine-tune. That is below 0.90. Inpaint raises full PRMVT from 0.5625 to 0.6634615384615384 and lowers the occlusion fine-tune from 0.6682692307692307 to 0.6057692307692307.

Median fill is applied only to the occluded test images. Clean accuracy of the occlusion fine-tune stays 0.8990384615384616 at 6 views.

## Test-time crops

No training. Median fill, then the full frame and four 70% corner crops. The crop is chosen without labels. Source: `results/robustness/occlusion_tta_seed42.json`. Test n=208, 6 views.

| rule | accuracy |
| --- | ---: |
| full frame | 0.875 |
| mean of the five crops | 0.8653846153846154 |
| most confident crop | 0.8557692307692307 |

Crops are below the full frame.

## Matched fine-tune

Training occludes a different box on each view, fills it with the median of the remaining pixels, and resumes the occlusion fine-tune. Batch-norm and dropout stay in eval mode. An epoch is kept only when clean 6-view validation does not fall below the starting value 0.9375 and occluded validation improves. The test split was not used to choose the epoch.

The starting checkpoint on validation is clean 0.9375 and median-fill occlusion 0.8413461538461539.

| run | what happened | clean val | occluded val | kept |
| --- | --- | ---: | ---: | --- |
| learning rate 0.0003, full pass | collapsed | 0.5769230769230769 | 0.5769230769230769 | no |
| learning rate 0.00001, full pass | collapsed | 0.5769230769230769 | 0.5769230769230769 | no |
| learning rate 0.00001, frozen batch-norm, full pass | collapsed | 0.5769230769230769 | 0.5769230769230769 | no |
| 40 notes, same settings | one batch had a non-finite gradient and was skipped | 0.9423076923076923 | 0.8413461538461539 | no |

0.5769230769230769 is 120/208, the number of genuine validation notes. The 40-note run reached that number because the tenth batch wrote non-finite ViT weights. Skipping that batch leaves clean validation at 0.9423076923076923, one note above the start, and leaves occluded validation unchanged. The saved checkpoint is byte-identical to `results/qduig/occlusion_ft/seed42/checkpoint.pt`.

A masked autoencoder was not trained. Security-feature labels are not in JaalTaka, so that prior is NOT_MEASURED.

Verdict: FAIL. Best measured 6-view test accuracy at occlusion 0.55 remains 0.875.
