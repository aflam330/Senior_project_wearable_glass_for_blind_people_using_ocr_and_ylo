# Auxiliary stack analysis

> **Correction 2026-09-28.** Q-DUIG accuracies in this file were computed before three evaluation bugs were fixed: NaN entropy for fully confident notes (scored as p = 0.5), a volume feature that changed when views were masked, and a missing view self-gate in the policy path. Every checkpoint was re-evaluated with the fixed code; see `WEAK_RESULTS_FIX.md`. Where those numbers differ from the ones below, they supersede them, and verdicts based on the old numbers should be re-read. The original text is kept unchanged below.

Seed 42. Numbers below are copied from saved logs. They are training-time validation logs, not test accuracy.

## Compared runs

- Full PRMVT, stage 1: `realtime_bangla_taka_detection/results/qduig/prefix/seed42/val_metrics.json`
- Full PRMVT, fine-tune: `realtime_bangla_taka_detection/results/qduig/prefix_ft/seed42/val_metrics.json`
- her_base (auxiliary losses off), fine-tune: `realtime_bangla_taka_detection/results/qduig/ablation_prefix/her_base/seed42/val_metrics.json`
- Test 1-view, full PRMVT: 0.9711538461538461 (`results/qduig/prefix_ft/seed42/test_views/views_1_to_6.json`)
- Test 1-view, her_base: 0.9903846153846154 (`results/qduig/ablation_prefix/her_base/seed42/test_views/views_1_to_6.json`)

Loss weights in `configs/proposed_prefix_ft.yaml`: auth 1.0, quality 0.25, uncertainty 0.15, diversity 0.10, info-gain 0.20, cost 0.05.

## Raw losses, full PRMVT stage 1, last epoch (epoch 6)

| term | raw | times weight | weighted |
| --- | ---: | ---: | ---: |
| auth | 0.6040661315408821 | 1.0 | 0.6040661315408821 |
| quality | 0.18274813866361334 | 0.25 | 0.045687034665903335 |
| uncertainty | 0.09979408691795945 | 0.15 | 0.014969113037693918 |
| diversity | 0.3674302036267776 | 0.10 | 0.03674302036267776 |
| info_gain | 0.00979873927637759 | 0.20 | 0.001959747855275518 |
| cost | 0.5893223619803756 | 0.05 | 0.02946611809901878 |

Weighted auxiliary sum is 0.1288250340205693. Authentication is 4.689043058544287 times that sum.

## Raw losses, full PRMVT fine-tune, last epoch (epoch 3)

| term | raw | weighted |
| --- | ---: | ---: |
| auth | 0.4106585505500711 | 0.4106585505500711 |
| quality | 0.1024460525711371 | 0.025611513142784275 |
| uncertainty | 0.05513868529491917 | 0.008270802794237876 |
| diversity | 0.34788517571327865 | 0.034788517571327865 |
| info_gain | 0.010268377376784736 | 0.002053675475356947 |
| cost | 0.765229283906596 | 0.0382614641953298 |

Validation 1-view at this epoch was 0.9807692307692307. The published test 1-view of the selected checkpoint is 0.9711538461538461.

## her_base fine-tune

Auxiliary terms are exactly 0 in every logged epoch (`quality`, `uncertainty`, `diversity`, `info_gain`, `cost`). Epoch 3 authentication loss is 0.194639. Validation 1-view at that epoch is 0.9807692307692307. The published test 1-view is 0.9903846153846154.

## What the logs do and do not show

The weighted auxiliary terms are smaller than the authentication term in both the 6-epoch stage and the fine-tune. Raw diversity is the same order of magnitude as authentication before the 0.10 weight. Raw cost is larger than authentication before the 0.05 weight. The mask-count cost has no parameter gradient unless `cost_from_entropy` is on. That flag is off in the full PRMVT config.

her_base also turns off the quality fusion path. With `use_quality: false` the model mean-pools views. With `use_quality: true` it uses RSQA. her_base also sets multitask and contrastive weights to 0. Full PRMVT sets them to 0.5 and 0.05. The 0.019230769230769273 test 1-view gap is therefore not explained by the weighted loss magnitudes alone. The forward path and the multitask/contrastive terms differ as well.

Gradient norm per loss: NOT_MEASURED. The training logs do not store gradient norms.

Feature drift in the shared encoder: NOT_MEASURED. No encoder-feature snapshot was saved.

Overfitting per task: the validation 1-view of full PRMVT fine-tune epoch 3 is 0.9807692307692307, while the test 1-view of the selected checkpoint is 0.9711538461538461. her_base validation 1-view at epoch 3 is also 0.9807692307692307, while its test 1-view is 0.9903846153846154. A per-task overfitting curve was not logged.
