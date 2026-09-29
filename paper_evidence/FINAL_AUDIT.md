# FINAL_AUDIT

```
DATA: PASS
BASELINE: PASS
PROPOSED METHOD: PASS
MULTI-SEED: NOT_MEASURED
ABLATIONS: NOT_MEASURED
CALIBRATION: PASS
ROBUSTNESS: PASS
GENERALIZATION: NOT_MEASURED
EDGE: PASS (current CUDA host) / Pi 5 NOT_MEASURED
EMOTION: PASS (preserved 86.5%)
OCR: PASS (preserved 15.5% CER)
USER STUDY: NOT_MEASURED
CLAIM VALIDATION: PASS
FINAL PAPER EVIDENCE: READY (with listed gaps)
```

## Hypotheses (pre-registered before this test eval)

| ID | Statement | Result | Artifact |
|---|---|---|---|
| H1 | Proposed equals or beats CNN+ViT while using fewer views at a predefined operating point | **FAIL** | adaptive acc 0.5865 @ 1.0 views; val Pareto `chosen: null` |
| H2 | Proposed is better calibrated than baseline | **PASS** | ECE 0.0303 vs 0.0789 (6-view) |
| H3 | Proposed degrades less under selected corruptions | **FAIL** | larger drop and lower acc on occlusion / heavy occlusion / low-light |
| H4 | Proposed policy reduces acquisition cost at comparable reliability | **FAIL** | CRIQP stops at 1 view; 6-view fusion remains the only reliable point |

## What the method actually showed

- 6-view HGEF+RSQA fusion beats the retrained CNN+ViT baseline on this note-disjoint test set (McNemar p=0.024).
- Sequential acquisition (CRIQP) did **not** recover that reliability with fewer views. Oracle analysis says useful 2-view subsets exist (oracle mean 2.10 views, acc 0.986); the learned surrogate/policy does not select them.
- This is the same qualitative limitation as the previous CAMVA adaptive run (`results/camva/`), now measured with an explicit information-gain policy rather than a confidence threshold.

## Honesty notes

- Existing CAMVA artifacts under `results/camva/` were not overwritten.
- Seeds 43/44 and trained ablations are absent; they must not be imputed.
- No “first”, SOTA, or venue-acceptance claim.
- HER improved ECE vs raw confidence on this split; that is one seed.

## Claim validation

`python realtime_bangla_taka_detection/scripts/validate_claims.py` → PASS (11 claims; 6 NOT_MEASURED entries allowed).

<!-- NOVEL_ALGORITHMS_START -->
## Additional algorithms

Ten extra algorithms were trained at seed 42. See `NOVEL_ALGORITHMS_COMPARISON.md`. v2 retrains are in `FAILED_FIXES_RESULTS.md`, `WEAK_IMPROVEMENTS_RESULTS.md`, and `NEW_ALGORITHMS_RESULTS.md`. PRMVT stays the primary accuracy result. VCIE, SFPL, and MTPT no longer predict only the majority class. OGPD v2 regressed, so v1 is the OGPD result. SFAQ security-feature labels, MTPT denomination and emotion, FedAvg+EWC, and CVS do-calculus identification are NOT_MEASURED.
<!-- NOVEL_ALGORITHMS_END -->


## A* addendum 2026-09-26 03:28 UTC

- Seed-42 numbers in `PAPER_DRAFT.md` are read from `test_metrics.json` and `views_1_to_6.json`.
- Seeds 43 and 44 are included only when their test files exist. See `MULTI_SEED_RESULTS.md`.
- Pi 5 and Android remain NOT_MEASURED.
- OGPD v2 is a negative result. SFPL v2 is a weak result.
- Literature labels are EXTENDS or ADAPTED. The search did not cover Scopus, IEEE Xplore, ACM DL, or Web of Science as separate databases.
- PAC-Bayes was not evaluated numerically.

## Publication pass 2026-09-26

- Ablation: PASS for the trained prefix variants. no_cost and no_calibration share the full checkpoint's forced-view accuracy because those switches do not change that training. The policy eval is FAIL as an acquisition method (0.4135).
- Figures: PASS for 12 files except that figure 7 is the collapsed policy, not a successful selector. Pi 5 latency figure was not produced.
- Calibration: PASS. NDAL entropy ECE 0.1679 is reported as measured.
- Theory: PASS for the definitions. Numerical PAC-Bayes is NOT_MEASURED.
- Pi 5, mobile, user study: NOT_MEASURED.

## Fix pass 2026-09-27

- Prefix policy: PASS. Test accuracy 0.9711538461538461 at 1.0096153846153846 views. The saved out-of-order policy remains a failure at 0.41346153846153844.
- Normalized cost: PASS at 1-view accuracy 0.9615384615384616.
- Smaller auxiliary weights: FAIL at 1-view accuracy 0.9663461538461539. Removing the auxiliary losses (`her_base`) remains 0.9903846153846154.
- Auxiliary stack, seed 42: loss scale, PCGrad, and every-fourth-batch FAIL below 0.9711538461538461. Kendall is PARTIAL at 0.9711538461538461. Curriculum is PARTIAL at 0.9807692307692307. Stop-gradient is 0.9903846153846154 at 1 view and 0.9086538461538461 at 6 views. A separate tower with frozen batch-norm matches `her_base` at 1–6 views. See `AUXILIARY_FIX_RESULTS.md`.
- RSQA versus mean-pool: mean-pool with auxiliary losses on is 0.9663461538461539 at 1 view. RSQA with auxiliary weights at 0 is 0.9375. Neither reaches 0.9903846153846154.
- Occlusion 0.55 median fill: 0.875 at 6 views. Below 0.90.
- Full-network PAC-Bayes: McAllester 0.6484603925932461 at posterior std 10, Gibbs error 0.42299794661190965. Not a certificate for the deterministic checkpoint.
- SFPL, VCIE, and MTPT new runs do not replace v2. Seeds 45 and 46 were not trained. NSTU-BDTAKA authenticity is NOT_MEASURED.
- Low light 0.2: PASS at 0.9375 after fine-tuning. Inference gamma repair scored 0.8317307692307693.
- Occlusion 0.55: FAIL at 0.6682692307692307.
- PAC-Bayes on the full weight vector: still vacuous. Linear-head McAllester bound on a training holdout: 0.444538876551335.
- HER scale 0.75 at threshold 0.5: accuracy 0.9759615384615384, ECE 0.024493631835167225. The validation-maximizing threshold does not keep that accuracy.


---

## Audit update 2026-09-28 (remaining-issues pass)

```
OCCLUSION 55%:        PARTIAL  51.9% -> 88.5% (seed 42), 87.0 +/- 2.2% (3 seeds), 88.9% ensemble; target 90% NOT_MET
OCCLUSION REJECTION:  PASS     wrong verdicts on occluded test notes 11.1% -> 0.5% (answers 46.6%)
MULTI-SEED:           PASS     PRMVT seeds 42-44; occlusion-robust seeds 42-44; CAMVA/baseline seeds 42-44
GPU SPEED:            PASS     RTX 3050: Taka YOLO 19.1 ms (5.5x CPU)
LIVE CAMERA LOOP:     PASS     7.5 FPS on laptop webcam; live-note accuracy NOT_MEASURED
GPIO BUTTONS:         PASS (simulated GPIO); hardware NOT_MEASURED
HAPTICS:              PASS (simulated GPIO, bug fixed); hardware NOT_MEASURED
SPEECH:               PASS (2 bugs fixed); subjective quality NOT_MEASURED
RASPBERRY PI 5:       NOT_MEASURED (no Pi; benchmark ready)
USER STUDY:           PROTOCOL_READY, DATA_NOT_COLLECTED
```

The test split was not used for any training, epoch choice, threshold or model choice. Every such choice was made on validation, with rules written before test was scored. Each claim is in `CLAIM_REGISTRY.json` with its source file.

## Audit update 2026-09-29

VCIE 1-view selection: PASS against the 0.9183 baseline at both 1 and 6 views (0.9278846153846154).
MTPT 6+3 schedule: PASS at 1 view (0.9807692307692307), which is above PRMVT's 0.9711538461538461.
PAC-Bayes half-data McAllester: PASS as a non-vacuous bound (0.11271182900151713) on that network only.
Occlusion ensemble and the 0.99 rejection rule: unchanged from `occlusion_decision.json`.
Multi-note composites: PASS at IoU ≥ 0.5 on 24, 36, and 60 pasted notes. Live photographs of several notes: NOT_MEASURED.
Pi 5, mobile latency, user study, and foreign-currency authenticity: NOT_MEASURED. Protocols are in the files named in `PAPER_FINAL.md`.

## Audit update 2026-09-29, second pass

Project Python compile: 168 files, 0 errors (`final_pass_20260929.json`).
Claim validation: exit 0, 120 claims.
Oracle identity: 205 correct, 201 learned, gap 4, unsolvable 3.
External image folders: 0 images (`external_bdt_manifest.json`).
Test split was not read to choose a model in this pass. No JaalTaka checkpoint was replaced.

## Audit update 2026-09-29, comparison sets

Four Bangladeshi image collections and ModelNet40 point clouds are in `data set for comparison`. Manifest: `comparison_dataset_manifest.json`. Authenticity was not scored on the denomination and coin folders.

ModelNet standing render test (`subset10_seed42_e40.json`): fixed and prefix both 0.64 at 1 view and 0.77 at 6 views, n = 100. Prefix validation peaked before the last epoch. MVP-N is absent. Foreign-currency folders were not added.

---

## Final audit, 2026-09-29 (evening)

```
SYNTAX:                 PASS  243 Python files, 0 errors
CHECKPOINTS:            PASS  158 model files load
METRICS REPRODUCE:      PASS  970 / 970 stored accuracies recomputed from saved predictions
CLAIMS:                 PASS  195 claims, 0 missing artifacts, 166 value-checked (all equal)
TEST LEAKAGE:           PASS  note-disjoint split, 0 overlap; every threshold and model choice on validation;
                              jaal operating points written before whole-note scores were read
CONTAMINATION:          PASS  "wild 96.9 %" labelled a source-domain check everywhere it appears as a result;
                              Diverse dataset excluded (MD5 = training source)
SAME-ARCHITECTURE:      PASS  3 arms x 3 seeds complete; prefix wins at 1 view on every seed with shared BN
SOTA (same split):      PASS as reported; NO state-of-the-art claim (frozen probes tie; 60 paired tests, none significant)
JAAL VERDICT:           ON as safe policy E for 500 / 1,000 Taka. Never says "counterfeit".
                              JaalTaka test counterfeit passed 0 / 88 (CI 0-4.2 %); whole-note 0 / 25 (4 groups; margin 2.4 log-odds)
SAFETY UNDER BAD LIGHT: PASS with quality gate (0 / 208 wrong per severe condition); confidence rejection alone: FAIL (59-86 wrong at 1 view)
THEOREMS:               PASS after 2 corrections (Theorem 6 wording; VCDS_THEOREM noise claim withdrawn)
PI 5:                   READY_FOR_DEVICE
USER STUDY:             READY_FOR_DEVICE (protocol ready, no participant data)
```

Details: `FINAL_SCAN.md`, `CORRECTIONS.md` (rows 10–19), `QUALITY_LOOP_FINAL.md`, `PAPER_FINAL.md`.

## Audit addendum, 2026-09-29 (night)

```
EXTRA SEEDS:        DONE  MTPT, VCIE, APC, CRIS, MAVT, SAVS seeds 43-44; 18 three-seed methods (BENCHMARK_FINAL.md)
FIGURES 1-12:       REDRAWN from post-fix files; superseded Q-DUIG-era set moved to Unused/superseded_figures
APPROACH A:         DONE, NEGATIVE  synthetic whole notes: real AUC 0.689, 14/20 real counterfeits passed -> not deployed
MVP-N:              DONE  attention head: prefix fixes VCDS (40.7 -> 48.0 % at 1 view); concat head: counterexample
SERIAL AUDIT:       NEW FINDING  JaalTaka counterfeits share serials; serial lookup = 90.2 % on test (JAALTAKA_SERIAL_AUDIT.md)
THEOREMS 8-11:      ADDED with proofs and measured checks
POLICY FIGURE 7:    RE-RUN with fixed code; 0.4135 -> 0.8606 (CORRECTIONS.md row 20)
CLAIMS:             PASS  211 claims, 182 value-checked, all equal
METRICS REPRODUCE:  PASS  970 / 970
```

## Audit addendum, 2026-09-30

```
SERIAL-DISJOINT SPLIT:  DONE   prefix network 90.1 / 87.8 % (1 / 6 views) on unseen counterfeit prints (was 98 %)
WATERMARK DETECTOR:     DONE   back-lit window, 1,261 / 1,390 registered; alone 88.3 % on unseen prints
HYBRID (net + wm):      DONE   95.5 % on unseen prints at 6 views, FCR 3/121, McNemar p = 0.0002
SERIAL BLACKLIST:       DONE   0 / 19 unseen-serial counterfeits (JaalTaka); real photos 12 / 25 flagged (1 of 4 prints), 0 / 450 genuine
CV OVER ALL NOTES:      DONE   98.3 / 95.8 % (note / serial folds, 1 view)
FUSION-GENERAL:         DONE   prefix helps pooling heads; concat trade not removable (4 repairs)
THEORY:                 ADDED  Propositions 12-14 (serial recall, print-sharing optimism, detector combination)
PI 5 / USER STUDY:      READY_FOR_DEVICE (PI5_READY.md, USER_STUDY_READY.md, SUS/NASA-TLX scorer self-checks pass)
TEST LEAKAGE:           none: every threshold, C and combiner fitted on TRAIN / VAL; serial-disjoint rule written before training
CLAIMS:                 PASS 234, value-checked 205
```
