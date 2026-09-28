# FINAL_RESULTS

> **Correction 2026-09-28.** Q-DUIG accuracies in this file were computed before three evaluation bugs were fixed: NaN entropy for fully confident notes (scored as p = 0.5), a volume feature that changed when views were masked, and a missing view self-gate in the policy path. Every checkpoint was re-evaluated with the fixed code; see `WEAK_RESULTS_FIX.md`. Where those numbers differ from the ones below, they supersede them, and verdicts based on the old numbers should be re-read. The original text is kept unchanged below.

Only numbers that exist as artifacts. Nothing else.

## Prefix-robust Q-DUIG, seed 42, test (MEASURED)

Same note-disjoint split. Same first-k view protocol as the CNN+ViT baseline. Checkpoint chosen by **mean validation accuracy over 1–6 views**, not by the test set.

Sources:
- `realtime_bangla_taka_detection/results/qduig/prefix_ft/seed42/test_views/views_1_to_6.json`
- Checkpoint: `results/qduig/prefix_ft/seed42/checkpoint.pt`
- n_test = 208 notes

| views | baseline | prefix-robust Q-DUIG | target |
|------:|---------:|---------------------:|-------:|
| 1 | 0.7356 | **0.9712** | 0.75 |
| 2 | 0.8702 | **0.9760** | 0.88 |
| 3 | 0.9135 | **0.9760** | 0.92 |
| 4 | 0.9183 | **0.9760** | 0.94 |
| 5 | 0.8990 | **0.9808** | 0.95 |
| 6 | 0.9183 | **0.9760** | 0.9663 |

Every listed target is met on this seed. This is **not** the adaptive stopping policy. These numbers are forced 1, 2, 3, 4, 5, or 6 views in file order.

Training: 6 epochs of mixed view-dropout (per-view gate, per-count batch norm, single-view auxiliary loss, light contrastive), then 3 epochs fine-tune at lr 3e-4 with half the batches using all 6 views. The first stage alone did **not** keep 6-view accuracy (test 6-view 0.9087). The fine-tune did. Seeds 43 and 44 are not in this table.

## Previous 6-view-only Q-DUIG (still true, different checkpoint)

Scientific object: quality-, diversity-, uncertainty-, and information-gain-aware sequential visual acquisition for edge currency authentication. RoboEye / Savior Glass is the deployment context, not the claim.

## Dataset (MEASURED)

Source: `realtime_bangla_taka_detection/results/camva/splits/split_metadata.json`

- Unique genuine notes: **802**
- Unique counterfeit notes: **588**
- Images: **8340** (6 views × 1390 notes)
- Train / val / test notes: **974 / 208 / 208**
- Split seed: **42**
- Leakage train∩val, train∩test, val∩test: **0 / 0 / 0**
- Independent unit: physical note ID (`genuine:note_*` / `counterfeit:note_*`)

## Authentication 1–6 views, same split, seed 42 (MEASURED)

Baseline source: `results/qduig/eval/seed42/baseline/baseline_{k}view/test_metrics.json`  
Proposed source: `results/qduig/eval/seed42/policies/full_proposed_{k}view/test_metrics.json`  
n_test = 208 notes.

| views | baseline acc | Q-DUIG acc |
|------:|-------------:|-----------:|
| 1 | 0.7356 | 0.5865 |
| 2 | 0.8702 | 0.5481 |
| 3 | 0.9135 | 0.5673 |
| 4 | 0.9183 | 0.6298 |
| 5 | 0.8990 | 0.7115 |
| 6 | 0.9183 | 0.9663 |

**Honest finding:** Q-DUIG is **worse** than the CNN+ViT baseline with 1–5 views. It is **better only at 6 views**. This repeats the earlier CAMVA pattern (`results/camva/`). 6-view fusion is still the reliability operating point.

## 6-view headline, seed 42 (MEASURED)

| metric | baseline | Q-DUIG |
|---|---:|---:|
| accuracy | 0.9183 | 0.9663 |
| macro F1 | 0.9143 | 0.9655 |
| balanced acc | 0.9064 | 0.9648 |
| ROC-AUC | 0.9709 | 0.9891 |
| ECE | 0.0789 | 0.0303 |
| Brier | 0.0673 | 0.0230 |
| NLL | 0.2337 | 0.1092 |

McNemar (predefined: FULL PROPOSED vs CNN+ViT BASELINE): n01=3, n10=13, p=0.0244 (Holm-adjusted 0.0489).  
Source: `results/qduig/eval/seed42/statistics.json`

## Adaptive CRIQP stopping (MEASURED, negative)

Source: `results/qduig/eval/seed42/policies/full_proposed_adaptive/test_metrics.json`

- accuracy **0.5865**, average views **1.00**
- val Pareto over λ ∈ {0.02, 0.05, 0.08, 0.12, 0.20, 0.35}: every setting also stops at 1 view (val acc 0.644). No point meets the pre-set val floor 0.90 (`pareto_val.json`, `chosen: null`).

The information-gain policy **does not** keep 6-view reliability while using fewer views.

## Oracle analysis (MEASURED, analysis only)

Source: `results/qduig/eval/seed42/oracle.json`

- Oracle subset acc **0.9856** at mean **2.10** views
- Learned 6-view acc **0.9663** at 6 views
- Accuracy gap (oracle − learned) **0.0192**
- View gap (learned − oracle) **3.90**
- Notes unsolvable by any subset: **3**

Useful subsets exist; the learned policy does not find them. Never used for training.

## Calibration, 6-view test (MEASURED; fit on val only)

Source: `results/qduig/eval/seed42/calibration.json`

| method | ECE | adaptive ECE | Brier | NLL |
|---|---:|---:|---:|---:|
| predictive confidence | 0.0303 | 0.0159 | 0.0230 | 0.1092 |
| temperature (T=1.107) | 0.0319 | 0.0181 | 0.0232 | 0.1068 |
| entropy mapping | 0.0427 | — | — | — |
| HER (novel) | 0.0240 | — | — | — |
| MC dropout | 0.0296 | — | — | — |

HER lowest ECE among the five methods on this split. Temperature did not beat raw confidence.

## Robustness, 6 views (MEASURED)

Source: `results/qduig/eval/seed42/robustness.json`  
Clean: baseline 0.918 / Q-DUIG 0.966.

Q-DUIG **higher** than baseline on gaussian blur, motion blur, brightness, contrast, JPEG, rotation, perspective, scale, sensor noise.  
Q-DUIG **lower** on low-light (0.587 vs 0.615), glare (0.894 vs 0.904), occlusion (0.668 vs 0.798), heavy occlusion (0.668 vs 0.846).

Occlusion remains a failure mode (same qualitative finding as prior CAMVA).

## Edge (MEASURED on this host; Pi 5 NOT_MEASURED)

Source: `results/qduig/edge/edge.json`

- Device: CUDA (not Raspberry Pi 5)
- End-to-end median **42.6 ms**, P95 **69.5 ms** (batch-1, 6 views, 50 repeats)
- Encode median **12.4 ms**
- Energy: NOT_MEASURED

## Preserved prior measurements (not re-run)

- Emotion RAF-DB: n=3068, acc **0.8654**, macro-F1 **0.795** (`savior_glass/results/emotion_rafdb.json`)
- OCR: n=80, CER **0.1547** (`savior_glass/results/ocr_cer.json`)
- Wild Taka: see `paper_evidence/detection/wild_note_metrics.json`

## NOT_MEASURED

- Training seeds 43 and 44 (proposed)
- Ablation re-trains (11 configs)
- Continual QWER-VPC vs FT/replay/EWC
- Simulated federated QDW-Fed
- Raspberry Pi 5
- Camera/session-disjoint authentication (JaalTaka `camera_id=unknown`)
- Metric pose accuracy (no calibrated intrinsics + real corners protocol in this run)
- VLM quality labels
- Energy / joules
- Human study

## Hypotheses (pre-registered; not revised)

See `FINAL_AUDIT.md`. H1 and H4 fail. H2 passes. H3 fails on occlusion / low-light degradation.

<!-- NOVEL_ALGORITHMS_START -->
## Ten additional algorithms, seed 42

Prefix-robust training (PRMVT) remains the main accuracy result in this file. The ten runs below are separate checkpoints. SFAQ does not detect security features. MTPT does not predict denomination or emotion. SFPL is simulated in one process. Negative and flat results are left as measured.

# Novel algorithms, seed 42

Every number below is copied from `results/novel/<algo>/seed42/test/test_metrics.json`.
Missing files are NOT_MEASURED. Nothing here was typed in by hand.

| algorithm | 1-view | 2-view | 3-view | 4-view | 5-view | 6-view | 6-view F1 | 6-view ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ogpd | 0.9231 | 0.9231 | 0.9279 | 0.9135 | 0.9087 | 0.9183 | 0.9139 | 0.0815 |
| vcie | 0.5769 | 0.5769 | 0.5769 | 0.5769 | 0.5769 | 0.5769 | 0.3659 | 0.0128 |
| apc | 0.9423 | 0.9423 | 0.9375 | 0.9183 | 0.8990 | 0.8317 | 0.8135 | 0.1647 |
| sfaq | 0.9471 | 0.9423 | 0.9663 | 0.9375 | 0.9375 | 0.9423 | 0.9405 | 0.0326 |
| igcr | 0.9471 | 0.9423 | 0.9519 | 0.9375 | 0.9279 | 0.9327 | 0.9298 | 0.0338 |
| ugf | 0.9471 | 0.9519 | 0.9567 | 0.9519 | 0.9375 | 0.9375 | 0.9352 | 0.0350 |
| ndal | 0.9135 | 0.9038 | 0.9231 | 0.9231 | 0.8990 | 0.9087 | 0.9046 | 0.0415 |
| sfpl | 0.5769 | 0.5769 | 0.5769 | 0.5769 | 0.5769 | 0.5769 | 0.3659 | 0.0420 |
| cvs | 0.8942 | 0.9135 | 0.8942 | 0.8846 | 0.8702 | 0.8750 | 0.8720 | 0.0938 |
| mtpt | 0.5769 | 0.5769 | 0.5769 | 0.5769 | 0.5769 | 0.5769 | 0.3659 | 0.0112 |

Algorithms with a test file: 10 of 10.

<!-- NOVEL_ALGORITHMS_END -->

## v2 retrains, seed 42

v1 files were kept. v2 numbers are in `results/novel_v2/<algo>/seed42/test/test_metrics.json`. PRMVT is unchanged (1-view 0.9712, 6-view 0.9760). UGF, SFAQ, and IGCR were not retrained.

The three majority-class runs recovered. VCIE is 0.8798–0.9135. MTPT is 0.9135–0.9519. SFPL, after the FedAvg integer-buffer crash was fixed, is 0.7404 at 1 view and 0.8317–0.8606 at 2–6 views. It is no longer a constant predictor. It is still below the CNN+ViT baseline from 2 views up. FedAvg+EWC was not trained.

APC v2 is 0.9615–0.9856 and no longer drops at 6 views (0.9663). NDAL v2 is 0.9567–0.9808. CVS v2 is 0.9327–0.9712. OGPD v2 fell to 0.7596–0.8413, so the OGPD number that stands is the v1 row above (0.9087–0.9279).

New runs: PRAVT 0.9519–0.9760, VAT 0.9663–0.9760, MAVT 0.9183–0.9663, SAVS 0.9279–0.9375, CRIS 0.9038–0.9519. SFAQ still has no security-feature labels. MTPT still has no denomination or emotion labels. CVS is still a leave-one-view logit change, not a fitted causal graph.


## A* assembly

The draft, multi-seed table, and ablation table are regenerated by `scripts/eval/write_astar_paper.py` from the JSON artifacts. Pi 5 and Android remain NOT_MEASURED.

## Prefix-protocol ablation, seed 42 (MEASURED)

Source: `paper_evidence/ABLATION_RESULTS.md`. Full is the saved 6+3 epoch checkpoint, not a new 4-epoch run. Removing redundancy or information gain drops 1-view accuracy from 0.9712 to 0.9327. A prefix-trained encoder with the auxiliary losses off (`her_base`) reaches 0.9904 at 1 view. Diversity alone is 0.9808 at every view count. Those are not improvements to claim as the method. They say the auxiliary stack is not what holds 1-view accuracy on this split.

The sequential policy on that same checkpoint is a negative result. With the validation cost weight 0.02 it stops at 1.00 view and scores 0.4135. With the cost weight at 0 it uses 2.22 views and scores 0.4231. Forced first-k evaluation of the same weights remains 0.9712 at 1 view. The policy is not selecting the training prefix.

## Calibration, seed 42, 6 views (MEASURED)

Source: `paper_evidence/CALIBRATION_RESULTS.md`. Fit on validation, scored on test, 10 bins, 10 dropout passes.

PRMVT HER ECE is 0.0328. Uncalibrated PRMVT ECE is 0.0608. NDAL entropy mapping ECE is 0.1679. That mapping was not refit on the test set. UGF temperature ECE is 0.0267, slightly lower than UGF HER at 0.0325.

## Remaining measurements, seed 42

HER, fit on validation and applied to the 6-view test scores of the saved PRMVT checkpoint, changes threshold-0.5 accuracy from 0.9759615384615384 to 0.9711538461538461. Temperature scaling and the entropy map stay at 0.9759615384615384. Source: `results/calibration/accuracy_seed42.json`.

The matched cost pair is in `ABLATION_RESULTS.md`. `cost_entropy` 1-view accuracy is 0.9519230769230769. `no_cost_matched` 1-view accuracy is 0.9615384615384616. Attaching the cost to predictive entropy did not raise 1-view accuracy on this split.

Severity curves for PRMVT, NDAL, and the CNN+ViT baseline are in `SEVERITY_CURVES.md`. Gaussian blur at severity 3 and low light at 0.35 match `results/robustness/top_seed42.json`. Low light at multiplier 0.2 drops PRMVT 6-view accuracy to 0.5769 (drop 0.3990). Heavy occlusion at 0.55 drops it to 0.5625.

A Gaussian PAC-Bayes posterior on the saved prefix-robust weights has smallest McAllester penalty 57.5892990573559 on the fixed prior grid. The bound is vacuous. Source: `results/theory/pacbayes_prmvt_seed42.json`.

Pi 5, mobile, and the user study remain NOT_MEASURED.

## Fix pass, seed 42

Details are in `POLICY_FIX_RESULTS.md`, `COST_FIX_RESULTS.md`, `AUXILIARY_FIX_RESULTS.md`, `LOWLIGHT_FIX_RESULTS.md`, `OCCLUSION_FIX_RESULTS.md`, `PACBAYES_FIX_RESULTS.md`, and `HER_FIX_RESULTS.md`.

Prefix stopping on the saved PRMVT checkpoint, λ = 0.02 chosen on validation, scores 0.9711538461538461 at 1.0096153846153846 views. The old out-of-order policy remains 0.41346153846153844.

Normalized entropy cost, 1 view, scores 0.9615384615384616, equal to the matched run without that term. Reducing auxiliary weights scores 0.9663461538461539 at 1 view, below full PRMVT. `her_base`, with those losses off, remains 0.9903846153846154.

A later auxiliary-stack pass is in `AUXILIARY_FIX_RESULTS.md`. Loss-scale, PCGrad, and every-fourth-batch fine-tunes of full PRMVT stay below 0.9711538461538461 at 1 view. Kendall matches full PRMVT at 0.9711538461538461. A 6-epoch curriculum reaches 0.9807692307692307. Stop-gradient from `her_base` keeps 1-view at 0.9903846153846154 and drops 6-view to 0.9086538461538461. A separate auxiliary tower with the authenticator and its batch-norm frozen matches `her_base` at every view count, with nonzero tower losses.

## Remaining-items pass, seed 42

Mean-pool with the auxiliary losses kept scores 0.9663461538461539 at 1 view. RSQA with those loss weights at 0 scores 0.9375. Neither closes the gap to `her_base`. Source: `RSQA_VS_MEANPOOL_ANALYSIS.md`.

Occlusion 0.55, median fill on the occlusion fine-tune, scores 0.875 at 6 views. That is the highest measured repair and is below 0.90. Source: `OCCLUSION_FIX_RESULTS.md`.

A full-network McAllester bound of 0.6484603925932461 was measured for posterior standard deviation 10. The Gibbs training error is 0.42299794661190965. The tight zero-mean prior remains vacuous at penalty 57.5892990573559. Source: `PACBAYES_FIX_RESULTS.md`.

SFPL v2 stays below the baseline from 2 views. The new full-view and short-step runs do not replace it. Source: `SFPL_FIX_RESULTS.md`.

OGPD v1 remains the OGPD result. v2 is worse at every view. Source: `OGPD_STATUS.md`.

Seeds 45 and 46 were not trained. NSTU-BDTAKA authenticity accuracy is NOT_MEASURED: that dataset has denomination labels, not genuine/counterfeit labels.

VCIE v2 remains the VCIE result. An 8-epoch rerun scored 0.7548076923076923 at 1 view. MTPT v2 remains the MTPT result. An authenticity-only rerun scored 0.9471153846153846 at 1 view.

HER scale 0.75 at threshold 0.5 still scores 0.9759615384615384 with ECE 0.024493631835167225. The frozen separate tower still matches `her_base` at 6 views, so the stop-gradient continuation is not required for that target.

Low light 0.2 after the robust fine-tune scores 0.9375 at 6 views. Clean 6-view accuracy of that checkpoint is 0.9663461538461539. Occlusion 0.55 after a further occlusion fine-tune scores 0.6682692307692307. That is below 0.90, and clean accuracy falls to 0.8990384615384616 at 6 views.

A linear-head PAC-Bayes posterior on half the training notes has McAllester bound 0.444538876551335. The bound on the full weight vector remains vacuous. HER scale 0.75 at threshold 0.5 scores 0.9759615384615384 with ECE 0.024493631835167225. The validation-chosen threshold 0.8867766261100769 scores 0.9711538461538461 on test.


---

## Update 2026-09-28: remaining issues

| Area | Result | Evidence |
|---|---|---|
| Occlusion 55% (6 views, test) | 51.9% → **88.5%** (seed 42); **87.0 ± 2.2%** over 3 seeds; **88.9%** 3-seed ensemble; clean stays 98–99%. Target 90% not met. | `OCCLUSION_FINETUNE.md`, `OCCLUSION_ENSEMBLE.md` |
| Occlusion with rejection | answers 46.6% of occluded test notes at 99.0% accuracy; wrong verdicts 0.5% of notes | `OCCLUSION_REJECTION.md` |
| Occlusion root cause | covered genuine notes were called counterfeit; earlier fine-tunes never trained properly | `DIAGNOSIS_OCCLUSION.md` |
| GPU speed (RTX 3050) | Taka YOLO 19.1 ms (52 FPS, 5.5× CPU); emotion 29.4 ms; PRMVT 16 ms | `GPU_SPEED.md` |
| Live camera | loop runs at 7.5 FPS; models 31 ms + 24 ms per frame; camera capture dominates | `LIVE_CAMERA_TEST.md` |
| Buttons / haptics | all logic checks pass with simulated GPIO; haptic verdict patterns fixed | `GPIO_TEST.md`, `HAPTICS_TEST.md` |
| Speech | stressed-user verdict dropping and Bangla fallback bugs fixed | `SPEECH_TEST.md` |
| Raspberry Pi 5 | NOT_MEASURED | `PI5_RESULTS.md` |
| User study | PROTOCOL_READY, DATA_NOT_COLLECTED | `USER_STUDY_PROTOCOL.md` |

## Update 2026-09-29

VCIE selected by validation 1-view accuracy: test 0.9278846153846154 at 1 view and at 6 views (`results/novel_v2/vcie_k1/seed42/test/test_metrics.json`). MTPT trained 6 epochs then 3, selected the same way: test 0.9807692307692307 at 1 view and 0.9663461538461539 at 6 views (`results/novel_v2/mtpt_prefix_ft/seed42/test/test_metrics.json`).

A network trained on 487 training notes has McAllester bound 0.11271182900151713 on the other 487 (`results/theory/pacbayes_d1_bound_seed42.json`). The test split was not used. This does not replace the vacuous 57.59 penalty on the published weights under a zero-mean prior.

Detector composites with 2, 3, and 5 pasted notes, 12 images each, detected every note at IoU ≥ 0.5. Mean IoU 0.9457147895426855, 0.9355042416533456, and 0.9365677215828507 (`results/bbox/bbox_multinote.json`). Live camera boxes were not measured.

Pi 5, a phone, and a user study remain unmeasured. Protocols: `PI5_RESULTS.md`, `MOBILE_RESULTS.md`, `USER_STUDY_PROTOCOL.md`.
