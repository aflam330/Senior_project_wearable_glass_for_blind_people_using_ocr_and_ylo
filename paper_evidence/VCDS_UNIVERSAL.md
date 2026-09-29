# View-count drop

Definition. For a model scored at forced view counts k = 1 and k = 6 on the same notes, the view-count drop is Acc(6) - Acc(1). A positive drop means the 6-view score is higher. This is an empirical gap on JaalTaka, seed 42, n = 208. It is not a theorem that every multi-view dataset has the same gap.

## JaalTaka, measured

| model | Acc(1) | Acc(6) | Acc(6) - Acc(1) | source |
| --- | ---: | ---: | ---: | --- |
| CNN+ViT baseline | 0.7355769230769231 | 0.9182692307692307 | 0.1826923076923076 | prefix view table in `FINAL_RESULTS.md` |
| PRMVT prefix fine-tune | 0.9711538461538461 | 0.9759615384615384 | 0.0048076923076923 | `results/qduig/prefix_ft/seed42/test_views/views_1_to_6.json` |
| VCIE, 1-view checkpoint | 0.9278846153846154 | 0.9278846153846154 | 0.0 | `results/novel_v2/vcie_k1/seed42/test/test_metrics.json` |
| MTPT, 6+3 schedule | 0.9807692307692307 | 0.9663461538461539 | -0.0144230769230768 | `results/novel_v2/mtpt_prefix_ft/seed42/test/test_metrics.json` |

The baseline loses 0.1826923076923076 when only the first view is kept. PRMVT's drop is one note. MTPT's 1-view score is higher than its 6-view score, so the drop is negative.

## ModelNet40 renders, standing run

Point clouds were drawn as six turntable images. This is not a photograph protocol and not the full 40-class benchmark. The standing run is 40 epochs, initialization pinned, 400 / 100 / 100 objects, seed 42. Validation selected the checkpoint: fixed-view best validation accuracy at 6 views is 0.87; prefix best mean validation accuracy is 0.8066666666666666. Source: `results/vcds_modelnet/subset10_seed42_e40.json`.

| training | test k=1 | test k=2 | test k=3 | test k=6 |
| --- | ---: | ---: | ---: | ---: |
| fixed, 6 views only | 0.64 | 0.70 | 0.73 | 0.77 |
| prefix, k drawn from 1..6 | 0.64 | 0.73 | 0.77 | 0.77 |

On this test the two trainings match at 1 view and at 6 views. Prefix training is higher at 2 and 3 views. The label-using prefix oracle, which picks any k that is correct, scores 0.85. It is not a policy.

Earlier files from shorter runs are `subset10_seed42.json` (5 epochs) and `subset10_seed42_e15.json` (15 epochs). Their initialization was not pinned, so they are separate runs, not prefixes of the 40-epoch trajectory. Prefix validation on the 40-epoch run peaked at epoch 34 and was lower at epoch 40, so the selection score had stopped rising.

## Datasets without aligned views

The Bangladeshi folders in `data set for comparison` are denomination or coin labels. NSTU recognition filenames are augmentation copies. MVP-N is not in the folder. No view-count drop was computed for them. Detail: `CROSS_DATASET_DOWNLOAD_LOG.md`.

---

## Correction 2026-09-29: PRMVT numbers after the NaN fix

The PRMVT rows above read `results/qduig/prefix_ft/seed42/test_views/`, written before the
NaN-entropy fix of 2026-09-28. The re-evaluated file `test_views_20260928/views_1_to_6.json` gives,
on the same 208 test notes: 1 view 202/208 = 0.9712 (unchanged), 2-4 views 204/208 = 0.9808,
5 views 205/208 = 0.9856, 6 views 204/208 = 0.9808 (was 203). The 1-to-6-view drop is therefore
-2 notes (6 views is two notes better), not one note. Over seeds 42-44 the means are 96.47 / 98.40 /
98.08 / 97.60 / 98.56 / 98.08 % (`WEAK_RESULTS_FIX.md`).

---

## Update 2026-09-29: three seeds, larger ModelNet test, and a decomposition of the drop

Script: `realtime_bangla_taka_detection/scripts/eval/vcds_mechanism.py`; raw: `results/vcds_mechanism.json`.

ModelNet setup:
- Same renders and network as `train_modelnet_vcds.py`.
- 10 classes, 64 train / 16 validation / 20 test objects per class, except bowl, which has only 84 files and so 4 test objects. That makes **184 test objects**.
- Seeds 42, 43 and 44. Each seed trains a fixed-view model (6 views only) and a prefix model (k drawn from 1..6).
- Checkpoints are chosen on validation as before.

JaalTaka setup: CNN+ViT baseline seeds 42–44 against PRMVT seeds 42–44, re-evaluated after the NaN fix. PRMVT has a different architecture from the baseline, so its gain is not purely the training schedule.

The drop of a fixed-view model from 6 to k views mixes two things:
- **information loss:** fewer views show less, and no model can recover it
- **view-count shift:** the model was never trained on k views

A prefix-trained model at k estimates what is achievable at k. So **excess loss = Acc_prefix(k) − Acc_fixed(k)** isolates the part that training can fix.

| k | JaalTaka fixed | JaalTaka prefix | JaalTaka excess (per seed) | ModelNet fixed | ModelNet prefix | ModelNet excess (per seed) |
|---:|---:|---:|---|---:|---:|---|
| 1 | 76.0 | 96.5 | **+20.5** (+23.6, +14.4, +23.6) | 54.7 | 65.2 | **+10.5** (+9.8, +6.0, +15.8) |
| 2 | 89.1 | 98.4 | +9.3 | 63.8 | 74.3 | +10.5 |
| 3 | 91.8 | 98.1 | +6.3 | 71.7 | 75.5 | +3.8 |

Findings:
- At 1 view, prefix training beats fixed training in all 6 dataset × seed runs, so view-count shift is present in a non-currency multi-view task too. This **replaces the single-seed ModelNet conclusion above** ("both trainings score 0.64 at 1 view", 100 test objects). With three seeds and 184 test objects, the fixed model loses 10.5 points at 1 view that prefix training recovers.
- On ModelNet, about half of the 23-point 6→1 drop is excess loss. The other half remains for the prefix model too, and is information loss.
- The size of the feature shift does not predict the size of the drop across datasets. Mean shift at 1 view, relative to the distance between class centroids, is 3.41 on JaalTaka and 0.55 on ModelNet, while the drops are 16.5 and 23.0 points. Within each dataset the correlation is positive (Spearman 0.53 and 0.89), because both fall as k grows.

Scope: two domains (Taka photos and rendered shapes), one architecture per domain. This is evidence that view-count shift is not specific to JaalTaka. It is not a proof that it occurs in every multi-view task. Seed-42 numbers differ slightly between reruns because of GPU non-determinism.
