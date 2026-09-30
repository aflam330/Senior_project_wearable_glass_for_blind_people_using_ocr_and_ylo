# Quality loop, final (2026-09-29)

Each row scores a component with a measured number, names the improvement tried, and says whether it helped. An improvement was kept only if it was chosen on training or validation data, or fixed before the test was read. The loop stopped where the next step needed data or hardware that is not on disk, or where it would have meant choosing on test data already seen.

## Iterations in this pass

| # | Component | Before | Change | After | Kept |
|---|---|---|---|---|---|
| 1 | Main JaalTaka model | PRMVT, per-count BN: 96.5 ± 1.5 % at 1 view | Shared-BN prefix network (same-architecture run) | 98.2 ± 0.7 % at 1 view; preferred on validation (0.9869 vs 0.9843) | yes, as the recommended model |
| 2 | Whole-note ranking (counterfeit-set originals) | AUC 0.640 (whole crop as 1 view) | Cut views 1–4 at positions measured on TRAIN notes | AUC 0.829 (PRMVT), 0.966 (ResNet-50 probe) | yes |
| 3 | Deployed jaal output | Off | Policy E: "likely genuine" above the max VAL counterfeit score, else "check by hand"; never "counterfeit" | 0 / 88 test and 0 / 25 whole-note counterfeits passed; 0 "counterfeit" spoken in 1,889 app runs | yes, on in the app |
| 4 | Looser jaal threshold | — | 99th percentile of VAL counterfeit scores | 16 / 25 whole-note counterfeits passed | **no** (unsafe) |
| 5 | Wrong verdicts under bad light, 1 view | 59–86 of 208 | Image-quality gate fixed on clean VAL | 0 of 208 in each tested condition; clean answered 193 / 208 | evaluated; the glass needs its own calibration first |
| 6 | Claim checking | File existence only | Value check via `json_key` | 166 / 195 value-checked, all equal | yes |
| 7 | Reproducibility of stored test metrics | Not checked | Recompute from saved predictions | 970 / 970 reproduce | yes |
| 8 | Stale pre-fix numbers | 5 LaTeX tables, 1 claim, `SOTA_BEAT.md`, 2 sentences | Generators pointed at post-fix files; regenerated | 0 known stale numbers in the current tables | yes |
| 9 | Resumable same-architecture runner | Treated a half-trained checkpoint as done | Waits for `train_summary.json` | 9 / 9 runs complete | yes |
| 10 | Prototype authenticator | Could include test notes | TRAIN notes only | Leak removed; still uninformative (0.915 vs 0.907 similarity), stays off | yes |

## Where the loop stopped, and why

| Component | Current | Why no further step here |
|---|---|---|
| Jaal on whole notes | Safe, but confirms 0.4–2 % of independent genuine notes | Any new checker or threshold chosen now would be chosen on whole-note photos already scored. Needed: grouped whole-note counterfeit photos from the glass camera |
| Occlusion, real fingers | Gate tested only on black synthetic boxes | Needs real hand-held photos with labels |
| JaalTaka accuracy | Prefix network and frozen probes tie (60 paired tests, none significant) | The 208-note test split has been read by about 110 runs; a further accuracy gain could not be shown honestly without a fresh test set, and there are no unused JaalTaka notes |
| Raspberry Pi 5 latency | READY_FOR_DEVICE | `savior_glass/scripts/benchmark_pi5.py` must run on the Pi |
| User study | READY_FOR_DEVICE | `USER_STUDY_PROTOCOL.md`; needs participants |
| MVP-N cross-domain check | Not on disk | Would need a download (about 2–3 GB) |

## Component scores now

| Component | Score | Source |
|---|---|---|
| Taka detector, synthetic test | mAP@0.5 0.995 | `results/training_v2/test_eval/test_metrics.json` |
| Taka detector, independent photos | 91.5 % (Bangla Money), 18.5 % (NSTU close-ups) | `CROSS_DATASET_TAKA.md` |
| Authentication, JaalTaka, 1 view (3 seeds) | 98.2 ± 0.7 % (prefix, shared BN) | `SAME_ARCH_RESULTS.md` |
| Authentication, 6 views (3 seeds) | 98.7 ± 0.3 % | same |
| Jaal policy safety, in-domain | 0 / 88 counterfeit passed (CI 0–4.2 %) | `results/jaal_whole/policy.json` |
| Wrong verdicts under severe bad light with the gate | 0 / 208 per condition (CI 0–1.8 %) | `results/safety/quality_gate_seed42.json` |
| Calibration, PRMVT raw | ECE 0.0146 | `results/calibration/suite_seed42.json` |
| Emotion (RAF-DB) | 86.5 % | `paper_evidence/emotion/emotion_rafdb.json` |
| OCR CER | 3.2 % English, 27.8 % Bangla | `paper_evidence/ocr/ocr_cer.json` |
| Claims value-checked | 166 / 195, all equal | `validate_claims.py` |

## Iterations, 2026-09-30

| # | Component | Before | Change | After | Kept |
|---|---|---|---|---|---|
| 11 | Counterfeit detection on unseen prints | 87.8 % (6 views) | Add watermark-window detector (validation-fitted combination) | 95.5 % (p = 0.0002) | yes (research; needs back-lit capture on device) |
| 12 | Serial as a detector | — | Blacklist and duplicate-serial rules | 0 / 19 unseen caught; redundant with watermark | no (known-print list only) |
| 13 | Evaluation size | 208 test notes | 5-fold CV over 1,390 notes | 98.3 / 95.8 % | yes |
| 14 | Honest generalisation estimate | note-disjoint only | serial-disjoint split | 90.1 / 87.8 % | yes (lead number) |
| 15 | Concat fusion under prefix training | −8.6 points at 6 views | 4 repairs | best −4.4 (rescaled slots) | no; use pooling heads |
| 16 | Watermark crop | contained the serial | box moved, visually checked | no digits | yes |

**Stopped because.** Further gains on unseen prints need more counterfeit prints: the test has about 20. Deploying the watermark needs back-lit photos from the glass camera. Both require data collection.

## Completion pass, 2026-09-30

| # | Component | Before | Change | After | Kept |
|---|---|---|---|---|---|
| 17 | Watermark model | ResNet-50 + logistic regression, 88.3 % | Fine-tuned MobileNetV3 / V2; V2 chosen on VAL AUC | 92.9 % (AUC 0.976) | yes |
| 18 | Hybrid on unseen prints | 93.8 / 94.7 % | Stronger watermark score | 94.4 / 95.0 % (3 seeds) | yes |
| 19 | Pi watermark model | INT8 broken (3 tries on V3) | MobileNetV2 + static INT8 | 2.6 MB, 100 % the same decisions as FP32 | yes |
| 20 | Whole-note transfer | policy E | AdaBN, CORAL | both worse | no |

Twenty iterations are recorded across the passes. The remaining gains need data: more counterfeit prints and glass-camera photos.

## Master-prompt pass, 2026-09-30

| # | Component | Before | Change | After | Kept |
|---|---|---|---|---|---|
| 21 | Baseline fairness | frozen ResNet-50 only | Fine-tuned ResNet-50, 3 seeds, 256-px view cache | 92.0 / 89.9 % on unseen prints | yes (reported baseline) |
| 22 | End-to-end watermark model | none | Attention fusion over view and watermark tokens | 90.8 / 90.5 % (watermark token +3 points) | reported, not the best |
| 23 | Uncertainty-weighted ensemble | none | Weights 1 / VAL log-loss | 93.4 / 93.1 % | reported, not the best |
| 24 | User-study power | approximation | Exact noncentral t | n = 105 detects dz ≥ 0.28 at 80 % | yes |
| 25 | Print-level uncertainty | not stated | Theorem 16 (m_eff = 23.6) | ±0.28 print-level | yes |

**Why it stops here.** The best method, the watermark hybrid, beats the quarter-resolution fine-tuned ResNet-50 by +5.1 points at six views but reaches p < 0.05 in only 3 of 6 per-seed comparisons. Making that robust needs more counterfeit prints, not more modelling on the same 24 effective prints.

## Cache fix, 2026-09-30

| # | Component | Before | Change | After | Kept |
|---|---|---|---|---|---|
| 26 | Fine-tuned ResNet-50 image input | Quarter-decoded JPEG cache, 92.0 / 89.9 % | Full JPEG resized to 256, then 224; epoch still chosen on validation | 94.9 ± 1.8 / 94.4 ± 2.6 % | yes, as the image baseline |
| 27 | Hybrid plus that fine-tune | Hybrid 94.4 / 95.0 % | Validation-fitted four-feature combiner | 94.9 / 94.6 %; six-view McNemar not significant | no; hybrid alone stays the six-view result |

The six-view watermark hybrid (95.0 ± 0.0 %) is still ahead of the repaired fine-tune (94.4 ± 2.6 %), and the difference is not significant. Further gains need more unseen prints.

## Pass on 2026-09-30, without new counterfeit photos

| # | Component | Change | Result | Kept |
|---|---|---|---|---|
| 28 | Which models the quarter-decode touched | Read the loaders | Only the ResNet fine-tune. Prefix network and watermark crops use other paths | yes, `DATA_PIPELINE_BUG.md` |
| 29 | Synthetic counterfeits from genuine notes | Not built | Stripping security features is not a dataset, and the external sets have no counterfeit labels | no images written |
| 30 | +5 points over the full-resolution fine-tune | Re-read the hybrid and the four-feature combiner | Not met. Six-view gap about 0.5 points, not significant | hybrid unchanged |

Stopped. Another architecture scored on this same test would be a test-set choice. New prints are not available.

## Watermark full decode, 2026-09-30

| # | Component | Change | Result | Kept |
|---|---|---|---|---|
| 31 | Watermark crops | Full JPEG decode, same 700 px registration width, MobileNetV2 seeds 42–44 | 90.5 ± 1.2 % on a slightly different registered set, versus 92.9 % for the published half-decode model | no; published model stays |

The prefix network already reads a full decode. It was not retrained. Synthetic counterfeit images were not generated.
