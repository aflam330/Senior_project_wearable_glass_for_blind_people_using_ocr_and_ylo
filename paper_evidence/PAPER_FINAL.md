# View-Count Distribution Shift in Multi-View Authentication: Measurement, Fix, and Safe Deployment with Watermark-Aware Detection

*Draft, 2026-09-30. Every number is read from a saved result file named in the text or in `CLAIM_REGISTRY.json`. Authors, affiliation and supervisor: to be filled in by the authors.*

---

## Abstract

Multi-view counterfeit detectors are trained on a fixed number of photographs per note but are used with however many a person manages to take. We call this view-count distribution shift (VCDS) and study it on JaalTaka (1,390 Bangladeshi 500 and 1,000 Taka notes, six views each) and on the MVP-N multi-view benchmark.

- **VCDS and its fix.** A joint-fusion network trained on six views collapses at one view and becomes seed-unstable (71.2 ± 11.4 %). Training on random view prefixes removes the collapse (98.2 ± 0.7 %; three seeds, exact McNemar p ≤ 9.3 × 10⁻¹⁰). A non-identifiability result explains the instability. On MVP-N the fix carries over to pooling fusion but not to slot concatenation.
- **The benchmark's weakness.** Most JaalTaka counterfeits share a few printed serial numbers: 279 of 322 readable counterfeit 500s carry one serial, and a serial lookup alone scores 90.2 %. On a serial-disjoint split whose test counterfeits come from unseen prints, accuracy falls from about 98 % to 88–90 %.
- **Watermark-aware detection.** A detector reading the back-lit watermark window restores much of that loss. It raises six-view accuracy on unseen prints from 87.8 % to 95.5 % (p = 0.0002), with 2.5 % of genuine notes flagged, while a serial blacklist catches none of them.
- **Safe deployment.** On an offline assistive glass we deploy a policy that never says "counterfeit". It passed 0 of 88 JaalTaka test counterfeits and 0 of 25 real whole-note counterfeit photos, and comes with a finite-sample guarantee.

---

*Earlier abstract (2026-09-29), kept for reference:*

Multi-view counterfeit detectors are trained on a fixed number of photographs per note but are used with however many a person manages to take. We call the gap view-count distribution shift (VCDS) and study it on JaalTaka, 1,390 Bangladeshi 500 and 1,000 Taka notes with six views each and a note-disjoint split. In a controlled comparison, the same network trained only on six views loses accuracy when given fewer views and becomes unstable across seeds: 71.2 ± 11.4 % at one view against 98.1 ± 0.5 % at six (three seeds, 208 test notes). Training on random view prefixes removes the collapse: 98.2 ± 0.7 % at one view and 98.7 ± 0.3 % at six. The prefix network wins at one view on every seed (exact McNemar p ≤ 9.3 × 10⁻¹⁰), and no arm differs at six views. We give a bound for prefix-mixture training, correct an unprovable stronger form, and show that frozen ImageNet backbones with per-view scoring, which cannot suffer VCDS, are a strong baseline (98.1 % at one view for ResNet-50). We then test the detector where it would be used, on whole-note photographs taken by a wearable camera. It does not transfer: the deployed configuration called 27–61 % of genuine notes counterfeit. We cut JaalTaka-shaped views out of each detected note, using view windows measured on training notes, and adopt a policy that never says "counterfeit": it says "likely genuine" only above a threshold fixed on validation and otherwise asks the user to check by hand. At that threshold no JaalTaka test counterfeit (0 / 88, 95 % CI 0–4.2 %) and no whole-note counterfeit photograph (0 / 25, four physical-note groups) was passed. The same check shows that the only whole-note counterfeit data available is too small and too biased to certify a counterfeit detector: a classifier reading only file metadata separates it. Running the glass's own code on 1,889 photographs spoke "counterfeit" zero times. The detector, the policy, Bangla/English OCR, object naming and emotion-adaptive speech run in an offline smart-glass prototype.

## 1. Introduction

Blind users in Bangladesh handle cash every day and have few offline, Bangla-speaking aids. A wearable can name a banknote reliably. Saying whether it is counterfeit is harder, and a wrong answer in either direction costs the user.

Authentication models built on several photographs of a note face a practical problem: the number of photographs at test time is not the number used in training. A user may capture one view, or three. A joint fusion network trained on exactly N views has never seen fewer, and nothing in its training constrains what it does then.

**Contributions.**
1. **A controlled measurement of VCDS.** One network, three seeds, four training arms that differ only in view-count exposure and batch-norm sharing (Section 5.1).
2. **Theory.**
   - A per-view-count bound for prefix-mixture training, and a counterexample to the stronger "err_k ≤ err_N + O(1/√n)" form.
   - A non-identifiability result explaining why fixed-view training is seed-unstable.
   - A finite-sample safety guarantee for the deployed threshold, with its behaviour under shift (Section 4).
3. **Honest baselines on the same split.** Twelve methods with three seeds and five frozen backbones, with paired tests (Sections 5.2–5.3).
4. **A deployment study and a safe policy.** Evidence that the close-up detector fails on whole notes, a geometry-based input adaptation, and a policy that never asserts "counterfeit", with its safety measured in-domain and on whole-note photographs (Section 6).
5. **An offline assistive glass** that uses these parts (Section 7).

## 2. Related work

**Bangladeshi currency.** Published work reports denomination recognition near 98–100 % on single datasets (MobileNet and NASNetMobile, arXiv:2101.05081; EfficientNet-B0, ICCIT 2025) and authentication near 99 % on NoteShieldBD (ResNet-101). None reports a score on the JaalTaka note-disjoint split, and we do not compare numbers across datasets (`SOTA_BEAT.md`). The JaalTaka data article (Data in Brief, DOI 10.17632/2m7wk5cy4c.2) evaluates ImageNet CNNs; its split was not available, so we re-run those backbones on ours (Section 5.3).

**Multi-view learning with missing views.** Missing-modality and view-dropout training is known in multi-view and multimodal learning. Our setting is narrower: views arrive in a fixed order and stop early, so the relevant shift is over prefix length.

**Selective prediction.** Chow's rule and its constrained form underlie our rejection analysis (Theorem 7). The deployment policy is a one-sided selective classifier: it may only abstain or say "likely genuine".

The full bibliography is in `docs/RELATED_WORK_BIBLIOGRAPHY.md`.

## 3. Method

**Network.** Each view is encoded by a MobileNetV3-Small with a TinyViT branch (128 × 128 input); a fusion head produces p(genuine) from a mask over the available views (`roboeye/qduig/`). PRMVT adds per-view-count batch-norm; the shared-BN variant uses one batch-norm for all counts.

**Prefix-mixture training.** Each training batch draws a prefix length k from a distribution π with π_k > 0 for k = 1..6, masks views k+1..6, and adds a single-view auxiliary loss (weight 0.5). Six epochs are followed by three fine-tuning epochs. The checkpoint is chosen by mean validation accuracy over one to six views.

**Fixed-view training.** The same network and schedule with π_6 = 1 and no single-view loss.

## 4. Theory

Full statements and proofs are in `THEOREMS.md`; two are summarised here.

**Theorem 1.** Under a label that does not change when views are dropped, the Bayes risk cannot fall when views are removed: R*(N) ≤ R*(k).

**Theorem 3 (prefix mixture).** Let ĥ minimise the empirical π-mixture risk over a class H, U_n(δ) a uniform-convergence bound, and ε_H the joint approximation gap inf_h max_k [R_k(h) − R*(k)]. With probability at least 1 − δ, for every k,

  R_k(ĥ) ≤ R*(k) + (ε_H + 2U_n(δ)) / π_k.

With fixed-view training (π_k = 0 for k < N) the argument bounds only R_N and leaves R_k unconstrained. The frequently requested stronger form, R_k ≤ R_N + O(1/√n), is false. A counterexample: if only view N carries the label, R*(k) = 1/2 for k < N.

**Proposition 11 (non-identifiability).** Fixed-view training never evaluates the network on fewer-view inputs. On a class rich enough to set those outputs freely, every minimiser of the fixed-view loss has the same six-view behaviour but arbitrary k-view behaviour. What training returns at k < N is then decided by the seed, not the objective.

**Theorem 8 (safety of the deployed threshold).** If τ is the highest score among n validation counterfeits and a new counterfeit is exchangeable with them, it passes with probability 1/(n+1). With confidence 1 − (1 − ε)^n the true pass rate is at most ε. For n = 88: expected 1.1 %, and at most 5 % with 98.9 % confidence. **Theorem 9** adds the total-variation distance between the source and deployment score distributions to that bound. That is why the whole-note result is evidence, not a guarantee.

**What the theory predicts and what we see.** The corollary predicts no guarantee, and hence possibly large seed-to-seed variation, at k < N for fixed-view training. Section 5.1 shows exactly that variation.

Two corrections were made during this work. Theorem 6's (1 − 1/e) inapproximability holds for the advantage over chance, not for accuracy. An earlier claim that the baseline's one-to-six-view drop exceeded Hoeffding noise was withdrawn; separating two estimated risks needs twice the band (`CORRECTIONS.md`, rows 15–16).

## 5. Experiments on JaalTaka

**Data.** 1,390 notes (802 genuine, 588 counterfeit), six views each: views 1–4 are overlapping front close-ups, 5 is the back, 6 is back-lit. The note-disjoint split is 974 / 208 / 208 (seed 42) with zero overlap (`results/camva/splits/split_metadata.json`). Test: 120 genuine, 88 counterfeit. Every checkpoint, threshold and model choice uses validation only.

### 5.1 Same network, fixed-view versus prefix training (Table 1, Figure 13)

**Table 1.** Test accuracy (%), mean ± sample std over seeds 42, 43, 44 (`SAME_ARCH_RESULTS.md`).

| Training | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 6-view, shared BN | 71.2 ± 11.4 | 73.1 ± 17.1 | 85.7 ± 6.6 | 90.5 ± 2.2 | 90.5 ± 3.1 | 98.1 ± 0.5 |
| **Prefix, shared BN** | **98.2 ± 0.7** | 98.4 ± 0.3 | 98.6 ± 0.5 | 98.6 ± 0.5 | 98.7 ± 0.3 | 98.7 ± 0.3 |
| Fixed 6-view, per-count BN | 87.2 ± 9.7 | 92.8 ± 6.3 | 95.5 ± 4.9 | 95.7 ± 4.6 | 95.8 ± 4.3 | 99.0 ± 0.0 |
| Prefix, per-count BN (PRMVT) | 96.5 ± 1.5 | 98.4 ± 0.6 | 98.1 ± 0.5 | 97.6 ± 0.8 | 98.6 ± 0.5 | 98.1 ± 1.0 |

- With shared BN, prefix training is better at one view on every seed: 203 vs 138, 204 vs 131, 206 vs 175 correct notes (exact McNemar p = 5.4 × 10⁻²⁰, 2.1 × 10⁻²², 9.3 × 10⁻¹⁰).
- With per-count BN, PRMVT is better at one view on seeds 42 and 44 (p = 8.4 × 10⁻¹², 0.013) but not on seed 43 (197 vs 194, p = 0.63).
- At six views no pair differs significantly (p ≥ 0.22).
- The fixed-view arm's one-view accuracy ranges over 131–194 correct notes across seeds and BN choices; the prefix arms stay within 197–206.

**Model choice.** By the project's validation rule (mean validation accuracy over one to six views), the shared-BN prefix network is preferred: 0.9869 against 0.9843 for PRMVT (`BENCHMARK_FINAL.md`). Per-count BN adds nothing measurable.

### 5.2 Leaderboard (Table 2, Figure 14)

**Table 2.** Methods with all three seeds, test accuracy (%) at one and six views (`BENCHMARK_FINAL.md`).

| Method | k=1 | k=6 |
|---|---:|---:|
| Prefix, shared BN | 98.2 ± 0.7 | 98.7 ± 0.3 |
| PRMVT | 96.5 ± 1.5 | 98.1 ± 1.0 |
| VAT | 97.3 ± 0.7 | 96.3 ± 0.3 |
| NDAL | 97.0 ± 1.0 | 95.8 ± 1.4 |
| APC | 97.0 ± 1.0 | 96.2 ± 0.5 |
| MTPT (6 + 3 epochs) | 96.8 ± 1.2 | 96.8 ± 0.3 |
| PRAVT | 96.2 ± 1.3 | 96.5 ± 1.5 |
| CVS | 95.4 ± 1.9 | 96.8 ± 0.6 |
| CRIS | 94.9 ± 0.6 | 93.6 ± 1.5 |
| MAVT | 93.6 ± 1.7 | 94.9 ± 1.0 |
| SAVS | 93.6 ± 0.7 | 92.9 ± 1.5 |
| VCIE (k1) | 93.4 ± 1.5 | 93.3 ± 0.5 |
| PRMVT checkpoints, score-level late fusion | 96.5 ± 1.5 | 94.7 ± 2.5 |
| PRMVT stage 1 only | 93.3 ± 1.9 | 93.8 ± 3.8 |
| CNN+ViT baseline | 76.1 ± 3.6 | 92.3 ± 0.8 |
| CAMVA quality attention | 54.0 ± 3.4 | 97.1 ± 0.8 |

About 110 runs have scored the same 208 test notes (`NEW_TEST_SET.md`). Single-seed variants are listed in `BENCHMARK_FINAL.md` but not ranked, since the best of many single runs is optimistic by up to about three points.

### 5.3 Frozen backbones on the same split (Table 3)

Frozen ImageNet features with one logistic regression on single training views (C chosen on validation), averaged over the first k views (`SOTA_COMPARISON.md`):

| Backbone | k=1 | k=6 |
|---|---:|---:|
| ResNet-50 | 98.1 | 99.0 |
| DenseNet-121 | 97.6 | 99.0 |
| VGG-16 | 96.6 | 99.0 |
| MobileNet-V2 | 96.6 | 98.6 |
| Inception-V3 | 96.6 | 98.6 |
| *Prefix, shared BN (3 seeds)* | *98.2 ± 0.7* | *98.7 ± 0.3* |
| *PRMVT (3 seeds)* | *96.5 ± 1.5* | *98.1 ± 1.0* |

We ran 60 paired exact McNemar tests: each probe against PRMVT and against the shared-BN prefix network, for each seed, at one and at six views. None is significant at p < 0.05.

These probes score every view alone, so they cannot suffer VCDS. On this test set they are statistically indistinguishable from the best joint network. We therefore make no state-of-the-art claim on JaalTaka. VCDS and its fix matter for joint fusion networks, which are what one builds when views should inform each other (quality weighting, view selection, attention). The CNN+ViT, CAMVA and fixed-view rows show that such networks do collapse.

### 5.4 Robustness, calibration and rejection

- **Occlusion** of 55 % of a view: a three-seed ensemble keeps 88.9 % (185 / 208) at six views. With a validation-chosen confidence threshold of 0.99, 0.48 % of occluded test notes receive a wrong verdict and 46.6 % receive one (`results/robustness/occlusion_decision.json`).
- **Calibration** after the NaN fix: PRMVT's raw confidence has ECE 0.0146 (10 bins). Temperature scaling (0.0168), HER (0.0178) and MC dropout (0.0136) do not clearly improve it (`results/calibration/suite_seed42.json`). Earlier HER figures of 0.0328 and 0.0245 were computed on pre-fix scores (`CORRECTIONS.md`, rows 12–13).
- **Cross-domain VCDS, rendered objects.** On ten rendered ModelNet classes: 64 % at one view, 77 % at six (`VCDS_UNIVERSAL.md`).
- **Cross-domain VCDS, real photographs (MVP-N, 44 classes, 4,400 test sets of 2–6 views, frozen ResNet-50 features, 3 seeds; `VCDS_MVPN.md`).** An attention-fusion head trained on six views scores 40.7 ± 1.0 % at one view, 14.7 points below a per-view reference (55.3 %). Prefix training raises it to 48.0 ± 0.7 % and also improves six views (82.2 vs 75.8 %). A zero-padded concatenation head is a counterexample: prefix training leaves one view unchanged (49.7 vs 48.7 %) and costs 8.6 points at six views. So the fix transfers to attention fusion, not to every fusion head.
- **Bad light.** Confidence rejection alone still answers wrongly under bad light. At one view with a clean-validation threshold, 59 of 208 test notes get a wrong verdict at brightness × 0.35, 71 at × 0.2 and 38 at × 2.2 (`results/safety/rejection_seed42.json`).
- **Image-quality gate.** A gate on mean luma, saturated share and near-black share, each set at the 1st or 99th percentile of clean validation views, removes every wrong verdict in those conditions (0 / 208 each, upper bound 1.8 %). It keeps 193 of 208 clean notes answered at one view (`results/safety/quality_gate_seed42.json`). These corruptions are severe and the occluder is black, so this shows the gate catches gross failures, not subtle ones.

### 5.5 Unseen counterfeit prints and the watermark window (Table 6)

**Print sharing.** Serial OCR on reconstructed whole notes shows that JaalTaka counterfeits share a few printed serials. 279 of 322 readable counterfeit 500s carry 3274658, while genuine notes have unique serials. A serial lookup therefore scores 90.2 % on the standard test split (`JAALTAKA_SERIAL_AUDIT.md`). Masking the serial at test time lowers PRMVT only from 97.1 % to 92.3 %, with genuine and counterfeit notes affected alike, so the network is mostly not reading it.

**Serial-disjoint split.** We re-split so that no test counterfeit shares a print with a training counterfeit: 222 test notes, 101 counterfeit (`SERIAL_FIX.md`).

**Watermark window.** JaalTaka view 6 is back-lit, so the watermark (portrait and denomination electrotype) is visible. Counterfeits carry an imitation: blank, faint, or printed. We register view 6 to the note template, crop the window below the serial (so no digits enter), and fit a logistic regression on frozen ResNet-50 features, with C chosen on validation. A second logistic regression, fitted on validation only, combines it with the network.

**Table 6.** Serial-disjoint test, 222 notes (`SERIAL_WATERMARK_DETECTOR.md`).

| Detector | Accuracy | Genuine called counterfeit | Counterfeits missed |
|---|---:|---:|---:|
| ResNet-50 probe, 1 / 6 views | 85.1 / 83.8 % | 1 / 121 | 32 / 35 of 101 |
| Prefix network, 1 / 6 views | 90.1 / 87.8 % | 1 / 121 | 21 / 26 of 101 |
| Watermark window alone (197 registered) | 88.3 % | 1 / 98 | 22 / 99 |
| **Prefix network, 6 views + watermark** | **95.5 %** | **3 / 121** | **7 / 101** |
| Serial blacklist | — | — | catches 0 (unseen prints) |

Adding the watermark fixes 19 notes and breaks 2 (exact McNemar p = 0.0002). On the standard split, the same combination reaches 99.0 %.

**Over three seeds** of the serial-disjoint network, the watermark raises unseen-print accuracy from 89.9 ± 0.7 % to 93.8 ± 1.0 % at one view, and from 89.3 ± 1.3 % to 94.7 ± 0.9 % at six views. It helps in all six seed-and-view comparisons, significantly in four. Genuine false alarms stay at or below 5 / 121 at six views on every seed. A device-sized MobileNetV3-Small watermark classifier (6.1 MB ONNX) reaches 91.9 % (AUC 0.962) on its own (Figure 16, `SERIAL_WATERMARK_DETECTOR.md`).

**Evaluation over every note.** Five-fold cross-validation over all 1,390 notes gives 98.3 % at one view with note folds and 95.8 % with serial-grouped folds (`EXPANDED_TEST.md`).

**Theory.** Proposition 12: a serial blacklist has zero recall on unseen prints. Proposition 13: the note-disjoint optimism equals (1 − q)(c_seen − c_unseen). Proposition 14: error bounds for combining detectors (`THEOREMS.md`).

## 6. Deployment: from close-ups to whole notes

### 6.1 The failure

The glass sees a whole note, not a close-up. Passing the detected note crop to PRMVT as one view (configuration S0) produced p(genuine) whose median depended on the photo collection more than on the note. S0 said "counterfeit" for 27.1 % of genuine notes in the counterfeit-set photos, 45.1 % on Bangla Money and 61.5 % on BanglaTaka (`results/jaal_whole/summary.json`). The glass's verdict was switched off on 2026-09-29.

### 6.2 What whole-note counterfeit data exists

The only whole-note counterfeit photographs on disk (a public 500 / 1,000 BDT set) contain 87 counterfeit images. Of these, 60 are augmented copies. The rest come from about four physical notes: two phone bursts, one bank-stamped note and one 500 BDT close-up series whose serial number also appears among JaalTaka counterfeits. A logistic regression on file metadata alone (size, aspect, format, bytes per pixel), trained leaving one counterfeit group out, catches 76 % of counterfeit images while flagging 1.5 % of genuine ones. The set therefore cannot certify a counterfeit detector by itself. A classifier trained on it (Approach B) fits it (0.1 % of its genuine images flagged) and then flags 86.7 % of independent BanglaTaka genuine notes.

### 6.3 View synthesis from the note crop

JaalTaka views 1–4 are front close-ups at stable positions. We registered them to a clean whole-note template (SIFT + RANSAC) for 200 training notes. The median windows, as fractions of note width, were view 1 [0, 0.48], view 2 [0.28, 0.83], view 3 [0.57, 1.0] and view 4 [0.51, 1.0], all at full height, with interquartile ranges within ±0.03 (`results/jaal_whole/view_geometry.json`). At test time the glass cuts these windows from the detected note and gives them to PRMVT as views 1..k.

**Table 4.** Whole-note photographs, 500 / 1,000 BDT. Share of genuine photos called counterfeit at p = 0.5, and ROC-AUC on the counterfeit set's original photographs (1,194 genuine, 25 counterfeit).

| Checker | cf-set genuine | Bangla Money genuine | BanglaTaka genuine | AUC |
|---|---:|---:|---:|---:|
| S0: whole crop, 1 view (deployed) | 27.1 % | 45.1 % | 61.5 % | 0.640 |
| S3: cut views 1–3 | 1.1 % | 20.5 % | 30.8 % | 0.819 |
| S4: cut views 1–4 | 5.6 % | 43.4 % | 60.2 % | 0.829 |
| R2: ResNet-50 probe, cut views 1–2 | 1.6 % | 12.5 % | 16.5 % | 0.966 |
| B: whole-crop probe trained on cf-set | 0.1 % | 26.4 % | 86.7 % | 0.886 |

Cutting views raises AUC from 0.64 to 0.83 for PRMVT and to 0.97 for the ResNet-50 probe. No checker reaches a false-counterfeit rate below 5 % on every genuine collection at the 0.5 threshold, so the glass must not say "counterfeit".

### 6.4 A policy that never says "counterfeit"

The glass says "সম্ভবত আসল" (likely genuine) if p(genuine) > τ, and otherwise "আসল কিনা হাতে যাচাই করুন" (check by hand). τ is the highest score of any JaalTaka validation counterfeit note. The checker is the one that confirms the most validation genuine notes at that τ. Both rules were written down before any whole-note score was read (`results/jaal_whole/PREREGISTERED_OPERATING_POINTS.md`). They selected S4 with τ = 0.99959.

**Table 5.** Policy E, S4, τ = 0.99959 (`results/jaal_whole/policy.json`).

| Data | Counterfeit said "likely genuine" | Genuine said "likely genuine" |
|---|---:|---:|
| JaalTaka test, real views 1–4 | **0 / 88** (95 % CI 0–4.2 %) | 69 / 120 (57.5 %) |
| Whole-note counterfeit photos (originals) | **0 / 25** (0 / 4 groups; image CI 0–13.3 %) | — |
| cf-set genuine | — | 194 / 1,194 (16.2 %) |
| Bangla Money genuine | — | 1 / 288 (0.3 %) |
| BanglaTaka genuine | — | 18 / 1,657 (1.1 %) |

**The policy is safe on all measured data but rarely confirms a genuine whole note.** It never produces a false "counterfeit". It passed no counterfeit in-domain or on whole notes. On independent genuine collections it confirms about 1 %, so on the glass it will nearly always say "check by hand" for 500 and 1,000 Taka. The looser threshold (99th percentile of validation counterfeits) confirms most genuine notes, but it also passed 13–16 of 25 whole-note counterfeit photographs. A threshold calibrated on close-ups does not transfer to whole notes. With four counterfeit groups, the whole-note safety evidence is thin (group-level 95 % upper bound about 49 %).

The app-level check ran the glass's own code on 1,889 photographs (`results/jaal_whole/app_check.json`):
- "Counterfeit" was spoken 0 times.
- 0 of 20 checked counterfeit originals and 0 of the augmented copies were said to be likely genuine.
- Genuine confirmation rates were 16.9 % (cf-set), 0.4 % (Bangla Money) and 2.0 % (BanglaTaka).

## 7. The assistive glass

The Savior Glass prototype (Raspberry Pi 5 target, five buttons, camera, bone-conduction audio, haptic motor) runs offline:

- **Currency.** YOLOv8s trained on 22,333 composites: test mAP@0.5 0.995, mAP@0.5:0.95 0.849 on 2,282 composites. 0.992 on 566 fresh composites. **91.5 %** correct denomination on 1,536 independent Bangla Money photos. 18.5 % on NSTU hand-held close-ups. AP50 0.796 on NSTU detection (`CROSS_DATASET_TAKA.md`, `NEW_TEST_SET.md`).
- **OCR.** EasyOCR (bn + en): CER 3.2 % English, 27.8 % Bangla on 80 rendered lines.
- **Objects.** YOLOv8s COCO: mAP@0.5 0.604 on 250 val2017 images.
- **Emotion.** EfficientNet-V2-S: 86.5 % on the RAF-DB test set (3,068), used to adapt speech rate and wording.
- **Speed.** Taka YOLO 19.1 ms on an RTX 3050 laptop GPU. Live loop 7.5 FPS on a laptop webcam.
- **Raspberry Pi 5.** Latency not yet measured; `savior_glass/scripts/benchmark_pi5.py` is ready.

## 8. Discussion

**When VCDS matters.** A per-view scorer with averaging is immune by construction and, on JaalTaka, as accurate as any joint network. A joint fusion network trained on a fixed view count is not immune. It can lose 10–40 points at one view, and how much it loses depends on the seed. Prefix training is a one-line change that removes that risk at no cost at six views.

**Why the deployed verdict was the harder problem.** Every in-domain number in Section 5 is at least 96 %. On whole notes the same model's scores track the photo collection. The binding constraint was data, not architecture: grouped whole-note photographs of known counterfeits. The policy in Section 6.4 is what can be shipped safely without them.

## 9. Limitations

- **One authentication dataset.** One split, 208 test notes, reused by about 110 runs. Only the three-seed rows are used for conclusions.
- **The serial-disjoint results are one seed of one split,** with 101 test counterfeits from about 20 small prints.
- **The watermark needs back-lighting.** The glass's front-lit whole-note photos cannot show it. A guided "hold the note up to the light" capture is proposed but not measured.
- **Pen-mark confound.** Circulated genuine notes often carry pen marks in the watermark window, and the watermark model was not tested against this.
- **JaalTaka counterfeits share serial numbers.** 279 of 322 readable counterfeit 500 BDT notes carry serial 3274658, and a serial lookup alone scores 90.2 % on the test split. PRMVT catches 98.5 % of test counterfeits whose serial it saw in training, and 89.5 % of the 19 whose serial it did not. Masking the serial at test time lowers one-view accuracy only from 97.1 % to 92.3 %, with genuine and counterfeit notes affected alike, so PRMVT is mostly not reading the serial (`JAALTAKA_SERIAL_AUDIT.md`). The split should still be made serial-disjoint.
- **Synthetic whole notes do not fix deployment.** PRMVT fine-tuned on reconstructed whole notes pasted on COCO backgrounds (Approach A) improves on synthetic test notes (AUC 0.875 → 0.933). But on real photographs its AUC falls to 0.689, and it would pass 14 / 20 real counterfeits. It was not deployed.
- **The view-count fix is not architecture-free.** On MVP-N it helps an attention-fusion head and does not help a concatenation head. A per-view scorer beats both there.
- **Thin whole-note counterfeit evidence.** About four physical notes, one of which may share a print batch with JaalTaka. The policy's whole-note safety is supported only weakly. The highest whole-note counterfeit score (0.99567) sits 2.4 log-odds below τ (0.99959).
- **Composite-heavy detector evaluation.** Detector test metrics are on composites. Independent real-photo results are 91.5 % on full notes and 18.5 % on close-ups.
- **Not yet measured.** No Raspberry Pi 5 measurements, no user study, no live-camera note accuracy.
- **No state-of-the-art claim on JaalTaka.** Frozen probes match the best network.

## 10. Conclusion

Training a joint multi-view detector on random view prefixes removes a real, seed-dependent failure at small view counts, with a guarantee whose form we state correctly. Moving the detector onto a wearable exposed a larger problem than view count: close-up training does not transfer to whole notes. Cutting training-shaped views from the note, a validation-fixed threshold and a policy that never says "counterfeit" make the feature safe on everything we could measure, at the price of rarely confirming a note. The next step is not a new model: it is grouped whole-note photographs of known counterfeit notes.

---

## Figures and tables

| Item | File | Source |
|---|---|---|
| Figure 13 | `figures/fig13_same_arch.png` | Table 1 |
| Figure 14 | `figures/fig14_leaderboard.png` | Tables 2–3 |
| Figure 15 | `figures/fig15_jaal_whole.png` | Table 4 |
| Figures 1–12 | `figures/` (`FIGURE_NOTES.md`) | earlier passes; figures dated before 2026-09-28 may show pre-fix PRMVT values |
| LaTeX tables | `tables/*.tex` | regenerated 2026-09-29 by `write_paper_tables.py` with post-fix values |

## Reproduction

| Result | Script |
|---|---|
| Table 1 | `scripts/train/run_same_arch.py`, `scripts/eval/write_same_arch_results.py` |
| Table 2 | `scripts/eval/write_benchmark_final.py` |
| Table 3 | `scripts/eval/eval_backbone_probes.py`, `scripts/eval/write_sota_comparison.py` |
| View windows | `scripts/eval/jaal_view_geometry.py` |
| Table 4 | `scripts/eval/jaal_whole_note.py` |
| Table 5 | `scripts/eval/jaal_policy.py` |
| App-level check | `scripts/eval/eval_jaal_safe_app.py` |
| Figures 13–15 | `scripts/eval/make_final_figures.py` |

All paths are under `realtime_bangla_taka_detection/`.

---

## Appendix: earlier evidence record (kept verbatim)

# Paper record, 2026-09-29

This file cites artifacts. It does not replace the paper source. Rounded headlines in older notes are the exact fractions below.

## Authenticity

PRMVT prefix fine-tune, test n=208: 1-view 0.9711538461538461, 6-view 0.9759615384615384.

VCIE selected on validation 1-view: 1-view and 6-view 0.9278846153846154. Above the CNN+ViT 6-view baseline 0.9182692307692307.

MTPT with the 6-then-3 epoch schedule, selected on validation 1-view: 1-view 0.9807692307692307, 6-view 0.9663461538461539.

The frozen side tower matches `her_base`: 1-view 0.9903846153846154, 6-view 0.9182692307692307.

## Occlusion

The three-seed ensemble, test occlusion 0.55: 0.8894230769230769 (`occlusion_decision.json`). With confidence at least 0.99, chosen on validation, wrong verdicts are 0.004807692307692308 of occluded test notes, and 0.46634615384615385 of those notes receive a verdict.

Median fill on the earlier occlusion fine-tune remains 0.875 on all 208 notes. Learned inpainting scored 0.7259615384615384. Those are different checkpoints.

## Boxes

Synthetic test, 2052 notes: mean IoU 0.9083, IoU ≥ 0.5 on all of them, IoU ≥ 0.75 on 0.9839, zero false alarms on 230 empty images (`bbox_eval.json`). Dark synthetic images, 183: app path mean IoU 0.909 and IoU ≥ 0.5 on all of them (`bbox_app_eval.json`). Multi-note composites: `BBOX_EXTENDED.md`. Live camera box accuracy: NOT_MEASURED.

## Theory

Half-data network, McAllester 0.11271182900151713 on 487 held-out training notes. Not a bound on published PRMVT. Full-network zero-mean penalty 57.5892990573559.

## Not measured

Pi 5 latency, phone TFLite latency, user-study outcomes, NSTU/USD/EUR/INR authenticity, ModelNet/MOSI/ADNI/COCO view-count drops, live multi-note photographs.

## Addendum 2026-09-29

`write_astar_paper.py` was not re-run. It rewrites `CLAIM_REGISTRY.json` and `ABLATION_RESULTS.md` from an older claim list. The standing registry has 122 claims and `validate_claims.py` exits 0. New count checks, Wilson intervals, and the download log are in `SOTA_BEAT.md`, `ORIGINAL_RESEARCH.md`, `CROSS_DATASET_DOWNLOAD_LOG.md`, and `QUALITY_LOOP_REPORT.md`.

## Correction 2026-09-29

PRMVT 6-view 0.9759615384615384 is the pre-fix value. After the NaN-entropy fix it is 0.9807692307692307 (`results/qduig/prefix_ft/seed42/test_views_20260928/`). The deployed glass no longer speaks a jaal verdict (`JAAL_VERDICT_FIXED.md`).
See `paper_evidence/CORRECTIONS.md`.
