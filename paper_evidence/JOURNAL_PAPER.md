# View-Count Distribution Shift and Watermark-Based Counterfeit Detection for Assistive Currency Recognition

*Journal draft (target: IEEE Access, Pattern Recognition Letters), 2026-09-30. Numbers from saved result files; each section names them. Authors to be added. The shorter conference version is `PAPER_FINAL.md`; the focused workshop version is `WORKSHOP_PAPER.md`.*

## Abstract

We study counterfeit recognition for an offline assistive glass that reads Bangladeshi Taka to blind users.
- **View-count shift.** A multi-view network trained on a fixed number of photographs collapses when given fewer, and does so unstably across seeds (71.2 ± 11.4 % at one view). Training on random view prefixes removes the collapse (98.2 ± 0.7 %). We explain the instability with a non-identifiability result and show the fix depends on the fusion design.
- **The benchmark's weakness.** The public JaalTaka benchmark is not print-disjoint: its counterfeits share a few serials, and accuracy falls to 89.9 % on unseen prints.
- **The watermark.** A watermark-window detector on a back-lit photo raises this to 95.0 %, with 3.3 % of genuine notes flagged. A serial-number blacklist catches no new print.
- **Deployment.** The glass announces denominations (91.5 % on independent photos) and uses a counterfeit policy that never says "counterfeit". It carries a finite-sample safety guarantee, and passed no counterfeit in-domain (0 / 88) or on real whole-note photos (0 / 25).

## 1. Introduction

Blind users in Bangladesh handle cash daily. An assistive device must name the note reliably and must not falsely accuse a genuine note of being counterfeit. We address three questions:
1. Does a counterfeit model stay accurate when the user takes fewer photos than it was trained on?
2. Does it recognise counterfeit prints it has never seen?
3. How can it be deployed safely when its whole-note accuracy is not good enough to accuse?

## 2. Related work

- Bangladeshi counterfeit and currency recognition;
- transmitted-light and security-feature imaging;
- serial-number recognition and duplicate-serial detection;
- leakage and group-disjoint evaluation;
- view dropout and incomplete multi-view learning;
- assistive currency readers.

Details and sources are in `LITERATURE_SEARCH.md`. View dropout is known; our contribution is the analysis of when and why it works, plus its evaluation under print shift.

## 3. View-count distribution shift: theory

Summary; proofs in `THEOREMS.md`, index in `THEORY_FINAL.md`.
- **Theorem 1.** Bayes risk cannot fall when views are removed.
- **Theorem 3.** Prefix-mixture training gives R_k(ĥ) ≤ R*(k) + (ε_H + 2U_n)/π_k. The stronger R_k ≤ R_N + O(1/√n) is false (counterexample).
- **Proposition 11.** Fixed-view training does not identify the k < N predictor, so its fewer-view behaviour is set by optimisation noise. This matches the measured seed spread (131–194 of 208 correct).
- **Theorem 15.** Prefix-mixture SGD converges at O(1/√T) (standard, applied).

## 4. Prefix-robust training (experiments)

**Same network, JaalTaka** (`SAME_ARCH_RESULTS.md`, 3 seeds):
- fixed, shared BN: 71.2 / 98.1 % at 1 / 6 views;
- prefix, shared BN: 98.2 / 98.7 %.

**Fusion designs** (MVP-N, 44 classes, `FUSION_GENERAL.md`):
- prefix helps attention pooling (40.7 → 48.0 % at 1 view) and mean pooling (47.1 → 49.5 %);
- for slot concatenation it trades six-view accuracy (80.5 → 71.9 %); four repairs did not remove that trade.

**Baselines** (`SOTA_COMPARISON.md`). Frozen ImageNet probes tie the network on the standard split (60 paired tests, none significant).

## 5. Watermark detector

Method, data and all results: `SERIAL_WATERMARK_DETECTOR.md`.
- **Window.** Back-lit view 6 is registered to a template (1,261 / 1,390 notes); the window is cropped below the serial.
- **Model.** A MobileNetV2 fine-tuned on window crops; 92.9 % alone on unseen prints (AUC 0.976). INT8, 2.6 MB, same decisions as FP32.
- **Hybrid.** A validation-fitted combination with the network gives 94.4 ± 0.5 % / 95.0 ± 0.0 % on unseen prints.
- **Against a fine-tuned ResNet-50** (92.0 / 89.9 %), the gain is +2.4 / +5.1 points, significant in 3 of 6 per-seed comparisons. The prefix network alone does not beat the fine-tuned ResNet-50 on unseen prints (`BEAT_RESNET50_FINAL.md`).

## 6. Serial anomaly

- **Findings** (`JAALTAKA_SERIAL_AUDIT.md`, `SERIAL_FIX.md`). OCR reveals shared counterfeit serials. A blacklist scores 90.2 % on the standard split but catches 0 counterfeits from unseen prints (Proposition 12).
- **On real whole-note photos:**
  - flags 1 of 4 counterfeit prints;
  - flags 0 / 450 genuine photos.
- **Role.** Useful only for known bundles.
- **Masking the serial** lowers PRMVT from 97.1 % to 92.3 %, so the network does not mainly read it.

## 7. Safety framework

- **Policy E.** Answer "likely genuine" only above the highest validation-counterfeit score, else "check by hand"; never "counterfeit".
- **Theorems 8–9.** Pass probability 1/(n+1), P(pass rate > ε) = (1 − ε)^n, plus a total-variation shift term.
- **Theorem 7.** Optimality of confidence rejection.
- **Image-quality gate.** Wrong verdicts under severe dark, over-exposure and black occlusion go from 59–86 to 0 of 208 at one view (`SAFETY_FRAMEWORK.md`).
- **Rejection on unseen prints** gives about 3 % wrong among answered notes against 1 % on validation. Calibration does not transfer across prints.

## 8. Experiments summary

- **All results:** `FINAL_RESULTS.md`.
- **Seeds and intervals:** `3SEEDS_ALL.md`.
- **Leaderboard:** `BENCHMARK_FINAL.md`.
- **Unseen-print table:** `tables/table12_unseen_prints.md`.
- **Baseline comparison:** `BEAT_RESNET50_FINAL.md`.
- **Figures:** 13–27 (`paper_evidence/figures/`).

## 9. Real-world deployment

The glass (Raspberry Pi 5 target, five buttons, bone-conduction audio) runs offline.

| Component | Result |
|---|---|
| Denomination, real photos | 91.5 % on 1,536 Bangla Money photos; 18.5 % on NSTU hand-held close-ups |
| OCR character error | 3.2 % English, 27.8 % Bangla |
| Emotion recognition | 86.5 % |
| Object naming | mAP50 0.604 |
| Counterfeit, whole-note photos | Nine approaches tried (`SYNTHETIC_FIX.md`); none safe enough to accuse, so policy E ships. It confirms few genuine whole notes (0.4–2 %) |
| Watermark | "Hold to the light" module built, off until tested on the glass camera |
| Pi 5 latency | READY_FOR_DEVICE (`PI5_READY.md`) |

## 10. User study (protocol ready; results pending)

Design for 105 participants, a 30-day plan, an IRB draft and SUS / NASA-TLX scoring are in `USER_STUDY_READY.md` and `USER_STUDY_READY_FINAL.md`. No participant data exists yet; this section will report it.

## 11. Discussion and limitations

- **Evidence base.**
  - One counterfeit dataset.
  - About 24 effective unseen prints, so absolute unseen-print numbers carry ±0.28 print-level uncertainty (Theorem 16); paired comparisons are firmer.
- **Deployment gaps.**
  - The watermark needs back-lit capture.
  - Whole-note counterfeit data from the glass camera is the missing ingredient.
- **The prefix fix** is view dropout, known elsewhere; our findings are when it helps (pooling fusion), why fixed-view fails (non-identifiability), and how the result changes under print shift.

## 12. Conclusion

- Counterfeit models for assistive devices should be trained on variable view counts, with pooling fusion.
- They should be evaluated on unseen counterfeit prints.
- They should lean on physical security features such as the watermark.
- They should never assert "counterfeit" without whole-note evidence.
