# Occlusion 0.55 limitation

Target: 6-view test accuracy 0.90 on all 208 notes. It was not reached.

| attempt | 6-view result on all test notes | file |
| --- | ---: | --- |
| black box | 0.6682692307692307 | `OCCLUSION_FIX_RESULTS.md` |
| median fill | 0.875 | `OCCLUSION_FIX_RESULTS.md` |
| learned reconstructor | 0.7259615384615384 | `OCCLUSION_INPAINT.md` |
| different box per view, early stop | no epoch kept; test stays 0.875 | `OCCLUSION_MULTIVIEW.md`, `OCCLUSION_EARLYSTOP.md` |
| five-crop test-time augmentation | 0.8653846153846154 | `OCCLUSION_TTA.md` |

Entropy rejection raises accuracy to 0.9361702127659575 on the notes it keeps, and rejects 0.3221153846153846 of occluded test notes (`OCCLUSION_REJECTION.md`). The score on every test note remains 0.875.

The missing region is 55% of every view. Filling it with a flat color, a classical inpaint, a learned blur, or another crop does not restore the ink. Fine-tuning on that input either leaves occluded validation unchanged or collapses the classifier to the genuine class after a non-finite ViT step. Security-feature maps are not in the labels, so that prior is NOT_MEASURED. Pi 5 latency for these repairs is NOT_MEASURED.

Verdict: FAIL on the full-test target 0.90. Best full-test 6-view accuracy is 0.875.

---

## Update 2026-09-28

The cause was found (`DIAGNOSIS_OCCLUSION.md`): the model called covered genuine notes counterfeit, and the earlier fine-tunes never trained properly. The frozen CNN, very short runs and the NaN-gradient bug were to blame.

A proper occlusion fine-tune (`OCCLUSION_FINETUNE.md`) raises 55%-occlusion test accuracy from 51.9% to 88.5% (seed 42), 87.0 ± 2.2% over three seeds, and 88.9% for the three-seed ensemble. Clean accuracy stays at 98–99%.

**The 90% target is still not met**, so occlusion remains a stated limitation, now a much smaller one. Adding rejection (`OCCLUSION_REJECTION.md`) cuts wrong verdicts on occluded test notes to 0.5%, at the cost of asking the user to reposition about half of them. All of this uses synthetic black boxes; real occlusion by fingers is NOT_MEASURED.
