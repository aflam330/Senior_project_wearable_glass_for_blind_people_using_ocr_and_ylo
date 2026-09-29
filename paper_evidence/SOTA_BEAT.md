# Same-split comparisons and published numbers, 2026-09-29

A published accuracy is a win for this project only when both numbers are authenticity on the same notes. Denomination recognition is a different task. A number from another dataset is cited, and it is not treated as a score on JaalTaka.

## Same JaalTaka test, 208 notes, seed 42

Counts were checked by `scripts/eval/final_scientific_pass.py`. Each accuracy is an integer over 208. Wilson 95% intervals use z = 1.959963984540054.

| system | correct | accuracy | Wilson low | Wilson high | source |
| --- | ---: | ---: | ---: | ---: | --- |
| CNN+ViT baseline, 1 view | 153 | 0.7355769230769231 | 0.6717625691851389 | 0.7908475381354906 | `results/qduig/eval/seed42/baseline/baseline_1view/test_metrics.json` |
| PRMVT prefix fine-tune, 1 view | 202 | 0.9711538461538461 | 0.9385063212370869 | 0.986713893404172 | `results/qduig/prefix_ft/seed42/test_views_20260928/views_1_to_6.json` (after the NaN fix) |
| PRMVT prefix fine-tune, 6 views | 204 | 0.9807692307692307 | 0.9516055170639194 | 0.9924967427740997 | same file |
| MTPT 6+3, 1 view | 204 | 0.9807692307692307 | 0.9516055170639194 | 0.9924967427740997 | `results/novel_v2/mtpt_prefix_ft/seed42/test/test_metrics.json` |
| VCIE k1 checkpoint, 1 view and 6 views | 193 | 0.9278846153846154 | 0.8844377972066675 | 0.9558132140491697 | `results/novel_v2/vcie_k1/seed42/test/test_metrics.json` |
| Occlusion ensemble, clean | 206 | 0.9903846153846154 | 0.9656251630425841 | 0.9973591419921957 | `results/robustness/occlusion_decision.json` |
| Occlusion ensemble, 55% box, 6 views | 185 | 0.8894230769230769 | 0.8395442985060947 | 0.9251785319627007 | same file |
| CNN+ViT baseline, 6 views | 191 | 0.9182692307692307 | 0.8730218458710761 | 0.9483471201880005 | `results/qduig/eval/seed42/baseline/baseline_6view/test_metrics.json` |

153/208 = 0.7355769230769231 and 191/208 = 0.9182692307692307. On this split, the prefix fine-tune is ahead of the CNN+ViT baseline at 1 view (202 versus 153) and at 6 views (204 versus 191). The paired McNemar test already stored for an earlier Q-DUIG checkpoint is a different comparison (`results/qduig/eval/seed42/statistics.json`) and is not reused here.

Reaching 0.90 occluded accuracy on all 208 notes would require 188 correct notes. The ensemble has 185. The Wilson interval includes 0.90. The point count does not.

## Published Bangladeshi numbers on other data

| paper | task | their number | scored on our test? |
| --- | --- | --- | --- |
| Chowdhury et al., NoteShieldBD transfer learning (ResNet101) | authenticity on NoteShieldBD | 99% | no |
| ICCIT 2025, EfficientNetB0, full and partial notes | denomination | validation 99.03%, test 97.90% | no |
| arXiv:2101.05081, MobileNet on 8,000 images | denomination | test 98.88% | no |
| arXiv:2101.05081, NASNetMobile on 1,970 images | denomination | test 100% | no |

Those papers do not publish a score on the JaalTaka seed-42 note split. This project does not publish a score on NoteShieldBD. The download attempt is in `CROSS_DATASET_DOWNLOAD_LOG.md`.

## Detector, same saved file

Synthetic notes, n = 2052: mean IoU 0.9083, share with IoU ≥ 0.5 equal to 1.0, false alarms 0 on 230 empty images at confidence 0.25, 0.35, and 0.5 (`results/bbox/bbox_eval.json`). Class-correct 0.9966 is not mAP.

---

## Correction 2026-09-29: PRMVT numbers after the NaN fix

The PRMVT rows above read `results/qduig/prefix_ft/seed42/test_views/`, written before the
NaN-entropy fix of 2026-09-28. The re-evaluated file `test_views_20260928/views_1_to_6.json` gives,
on the same 208 test notes: 1 view 202/208 = 0.9712 (unchanged), 2-4 views 204/208 = 0.9808,
5 views 205/208 = 0.9856, 6 views 204/208 = 0.9808 (was 203). The 1-to-6-view drop is therefore
-2 notes (6 views is two notes better), not one note. Over seeds 42-44 the means are 96.47 / 98.40 /
98.08 / 97.60 / 98.56 / 98.08 % (`WEAK_RESULTS_FIX.md`).

---

## Update 2026-09-29 (evening): same-split backbones

The five ImageNet backbones used in the JaalTaka data article were run on this split as frozen linear probes (`SOTA_COMPARISON.md`):

| Backbone | 1 view | 6 views |
|---|---:|---:|
| ResNet-50 | 98.1 % | 99.0 % |
| DenseNet-121 | 97.6 % | 99.0 % |
| VGG-16 | 96.6 % | 99.0 % |
| MobileNet-V2 | 96.6 % | 98.6 % |
| Inception-V3 | 96.6 % | 98.6 % |

Against PRMVT and the shared-BN prefix network (3 seeds each), 60 paired exact McNemar tests found no significant difference.

**Verdict: the project does not beat the state of the art on JaalTaka authentication accuracy, and does not claim to.** What it adds:
1. **VCDS in joint fusion networks, and its fix.** Fixed-view training collapses at 1 view (71.2 ± 11.4 %); prefix training does not (98.2 ± 0.7 %). Measured in `SAME_ARCH_RESULTS.md`.
2. **A measured deployment of a JaalTaka-trained checker on whole-note photographs,** with a policy that passed no counterfeit (`JAAL_VERDICT_FIXED.md`).
3. **Same-split baselines,** which earlier Bangladeshi work does not report.

---

## Literature check, 2026-09-30 (web search; only claims visible in the search results are used)

| Area | What the literature has | Where this project stands | Source |
|---|---|---|---|
| Bangladeshi counterfeit detection | CNN classifiers on single splits, e.g. modified AlexNet + SVM; one model reported 85.4 % → 90.03 % after retraining. Security features (watermark, thread) motivate these models but are not evaluated separately | Counterfeit prints unseen in training (serial-disjoint split): prefix network 89.9 %; + watermark window 94.4–95.5 %. No prior work found that reports an unseen-print evaluation | [academia.edu](https://www.academia.edu/116590712/Enhanced_Counterfeit_Detection_of_Bangladesh_Currency_through_Convolutional_Neural_Networks_A_Deep_Learning_Approach), [ResearchGate](https://www.researchgate.net/publication/364609677_A_Deep_Learning_Approach_for_Detecting_Bangladeshi_Counterfeit_Currency), [Springer](https://link.springer.com/chapter/10.1007/978-3-031-19958-5_51) |
| JaalTaka benchmark | 1,390 notes (802 genuine, 588 counterfeit), six region images per note; counterfeits from the Rapid Action Battalion; Data in Brief, 2025 | The serial audit shows its counterfeits share a few printed serials, so random splits are not print-disjoint (`JAALTAKA_SERIAL_AUDIT.md`) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12774690/), [Mendeley Data](https://data.mendeley.com/datasets/2m7wk5cy4c/2), [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S235234092501090X) |
| Taka security features | Portrait watermark with a bright electrotype denomination; a 4 mm security thread | Used: the watermark window is the most transferable cue measured here | [BookMyForex guide](https://www.bookmyforex.com/currency-exchange/counterfeit-detection-guide/bangladesh-taka/) |
| Missing or variable views in multi-view learning | View dropout is known to make models robust to missing views, and to be able to hurt full-view accuracy; incomplete multi-view learning is an active field | Prefix training is a form of view dropout, so it is **not new as a technique**. New here: the non-identifiability explanation, the finding that the benefit depends on pooling vs concatenation fusion, and the application to authentication | [arXiv 2501.01132](https://arxiv.org/pdf/2501.01132), [arXiv 2303.17117](https://arxiv.org/html/2303.17117v4) |

**Verdict.**
- **Standard JaalTaka split:** the project ties the best simple baselines.
- **Unseen counterfeit prints:** it is the strongest measured result found (watermark-aware hybrid). No published number exists on that protocol to beat.
- **The training fix itself is not a new technique.** Its analysis and the fusion-dependence finding are.
