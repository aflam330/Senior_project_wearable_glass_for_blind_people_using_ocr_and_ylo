# Original checks, 2026-09-29

These statements were checked against saved JSON. They were not used to train, to pick a checkpoint, or to pick a threshold.

## Count identity for the subset oracle

Source: `results/qduig/eval/seed42/oracle.json`, n = 208.

Let A be the number of test notes a label-using subset selector gets right, and L the number the learned policy in that file gets right. The file stores accuracies. Multiplying by 208 and rounding recovers integers, and dividing those integers by 208 returns the stored accuracies within 1e-9.

| quantity | notes |
| --- | ---: |
| oracle correct | 205 |
| learned correct | 201 |
| gap | 4 |
| unsolvable by any subset | 3 |

3 = 208 − 205. The script `scripts/eval/final_scientific_pass.py` exits if either identity fails. This run exited 0. The gap 4/208 equals the stored `accuracy_gap_oracle_minus_learned` 0.019230769230769273.

Consequence that follows from this file alone: three test notes are wrong under every view subset the oracle searched. No policy that only chooses views from that same search can score above 205/208 on this test file.

The prefix-stop policy in `results/qduig/prefix_ft/seed42/prefix_stop_policy.json` is a different operating point. Its test accuracy is 0.9711538461538461. It is not subtracted from the oracle gap above.

## Wilson intervals

For k correct out of n = 208, with z = 1.959963984540054,

centre = (k/n + z²/(2n)) / (1 + z²/n),

half-width = z · sqrt( (k/n)(1 − k/n)/n + z²/(4n²) ) / (1 + z²/n).

The intervals are stored in `paper_evidence/final_pass_20260929.json` under `wilson`. They describe sampling uncertainty of a fixed classifier on 208 notes. They are not a PAC-Bayes bound.

## What this adds

Prior write-ups reported the oracle gap as a float. This pass shows it is exactly four notes, and that the unsolvable count is exactly the notes outside the oracle's correct set. The Wilson intervals are new derived numbers from those same counts.

---

## Correction 2026-09-29: which model the oracle file describes

`results/qduig/eval/seed42/oracle.json` was produced by `scripts/evaluate_all.py` for the early
`results/qduig/proposed` model (its 6-view accuracy, 0.966, is that model's), not for the deployed
PRMVT. It was also computed before three evaluation bugs were fixed on 2026-09-28 (missing view
self-gate in `run_oracle`, NaN entropy for confident notes, masked-view volume; see
`WEAK_RESULTS_FIX.md`). The oracle was re-run on the deployed checkpoints with the fixed code,
test split, n = 208 (`results/qduig/oracle_20260929/`):

| checkpoint | oracle accuracy | oracle mean views | 6-view accuracy | gap (notes) | unsolvable by any subset |
|---|---:|---:|---:|---:|---:|
| PRMVT prefix_ft seed 42 | 0.9952 (207) | 1.00 | 0.9808 (204) | 3 | 1 |
| occlusion-robust seed 42 | 1.0000 (208) | 1.00 | 0.9808 (204) | 4 | 0 |
| occlusion-robust seed 43 | 1.0000 (208) | 1.00 | 0.9904 (206) | 2 | 0 |
| occlusion-robust seed 44 | 0.9952 (207) | 1.00 | 0.9904 (206) | 1 | 1 |

The oracle reads the label, so these are analysis ceilings, not policies. For the deployed model
almost every note has at least one single view it classifies correctly; the "three notes no subset
can solve" in the section above belongs to the early model and to the pre-fix code.

---

## New contributions, 2026-09-29 (evening)

Each item was measured in this pass. Sources are named.

1. **VCDS is a stability failure, not only an accuracy drop.** In the controlled same-network comparison (`SAME_ARCH_RESULTS.md`), fixed 6-view training gives one-view accuracies that swing across seeds: 131–194 correct notes out of 208. Prefix training stays at 197–206. Theorem 3's corollary predicts exactly that: nothing constrains R_k for k < N under fixed-view training.

2. **Per-count batch-norm is unnecessary.** The shared-BN prefix network is at least as good as PRMVT at every view count. It is preferred by the validation rule (0.9869 vs 0.9843 mean validation accuracy over 1–6 views; `BENCHMARK_FINAL.md`).

3. **Registration-based view synthesis.** JaalTaka views 1–4 were registered to a whole-note template on 200 training notes (SIFT + RANSAC). Their positions are stable (interquartile range within ±0.03 of note width; `results/jaal_whole/view_geometry.json`). Cutting those windows from a detected whole note turns the deployment input into the training input without generating images. On whole-note photographs this raises the counterfeit-ranking AUC from 0.640 to 0.829 for PRMVT and to 0.966 for a ResNet-50 probe (`results/jaal_whole/summary.json`).

4. **A shortcut audit for small counterfeit sets.** A classifier that reads only file metadata catches 76 % of the counterfeit set's counterfeit images while flagging 1.5 % of genuine ones (leave one counterfeit group out). This test is cheap and should be run on any small counterfeit dataset before its numbers are trusted. The same set's counterfeit series shares a serial number with JaalTaka counterfeits.

5. **A one-sided safe policy with a pre-registered threshold.** "Likely genuine" above the highest validation-counterfeit score; otherwise "check by hand"; never "counterfeit". Measured: 0 / 88 JaalTaka test counterfeits and 0 / 25 whole-note counterfeit photographs passed, and "counterfeit" spoken 0 times in 1,889 app runs (`JAAL_VERDICT_FIXED.md`). A threshold calibrated on close-ups at the 99th percentile passed 16 / 25 whole-note counterfeits, so calibration does not transfer across capture conditions.

6. **Confidence rejection fails under corrupted input.** At one view and threshold 0.99, PRMVT still gives 59 / 208 wrong verdicts under low light 0.35 and 71 / 208 under low light 0.2 (`results/safety/rejection_seed42.json`). An image-quality gate fixed on clean validation is evaluated in `results/safety/quality_gate_seed42.json`; see `SAFETY_FRAMEWORK.md`.

7. **Frozen backbones are a necessary baseline.** A per-view linear probe on ImageNet ResNet-50 features scores 98.1 % at one view and 99.0 % at six on the same 208 notes (`SOTA_COMPARISON.md`). Per-view scoring cannot suffer VCDS, so VCDS claims apply to joint fusion networks.

8. **The claim validator now checks values.** 107 of 136 existing claims are compared with the number in their artifact, and all agree after one stale pre-fix value was corrected (`CLAIM_VALUE_AUDIT.md`).
