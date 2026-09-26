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
