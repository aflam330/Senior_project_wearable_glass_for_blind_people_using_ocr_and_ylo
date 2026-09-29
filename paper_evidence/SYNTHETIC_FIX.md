# Whole-note transfer: every approach tried (2026-09-30)

**Problem.** A checker trained on JaalTaka close-ups fails on whole-note camera photos (`JAAL_VERDICT_FIXED.md`).

| # | Approach | Real whole-note AUC (counterfeit-set originals) | Real counterfeits passed or missed | Kept? |
|---|---|---:|---|---|
| 1 | Whole crop as one view (old deployed) | 0.640 | 27–61 % of genuine called counterfeit | no |
| 2 | Cut JaalTaka-shaped views, 1–4 (S1–S4) | up to 0.829 | used inside policy E | **yes (S4 + policy E)** |
| 3 | Frozen ResNet-50 probe on cut views | 0.966 | passes 59 % of counterfeit images at 0.5 | no (miscalibrated) |
| 4 | Classifier trained on the counterfeit set (B) | 0.886 | 86.7 % of BanglaTaka genuine called counterfeit | no (shortcut) |
| 5 | Synthetic whole notes from JaalTaka on COCO backgrounds, fine-tune PRMVT (A) | 0.689 | 14 / 20 passed at its validation threshold | no |
| 6 | Watermark-aware checking | not applicable to front-lit photos: the watermark is visible only against light | — | READY_FOR_DEVICE as a "hold to the light" capture step |
| 7 | Serial-aware checking (blacklist on whole-note OCR) | see `SERIAL_WATERMARK_DETECTOR.md` | — | — |

**Why synthetic whole notes failed.** They are made from close-up photos taken under JaalTaka's lighting and camera. Pasting them on COCO backgrounds changes the background, not the note's appearance. The model learns the synthetic notes better (AUC 0.875 → 0.933), but real whole-note photos of other notes worse (0.810 → 0.689).

**What would fix it.** Whole-note photographs from the glass camera, grouped by physical note, including known counterfeits. That is data collection, which is left to the authors. Until then the glass runs policy E: "likely genuine" only above a validation-fixed threshold, otherwise "check by hand", never "counterfeit". It passed 0 of 25 real counterfeit photos.

## Unsupervised domain adaptation (approaches 8 and 9, 2026-09-30)

**Setup** (`scripts/eval/domain_adapt_whole.py` → `results/jaal_whole/domain_adapt.json`).
- **Adaptation data:** 88 unlabelled 500 / 1,000 BDT photos from Bangla Money "Testing", never scored.
- **Success rule (fixed before scoring):** AUC ≥ baseline and < 5 % of genuine called counterfeit on Bangla Money and BanglaTaka.

| # | Method | AUC, counterfeit-set originals | Genuine called counterfeit: cf-set / Bangla Money / BanglaTaka | Rule met |
|---|---|---:|---|---|
| — | PRMVT, 4 cut views (baseline) | 0.810 | 4.6 / 43.4 / 60.1 % | no |
| 8 | + AdaBN (batch-norm statistics re-estimated on the adaptation views) | 0.641 | 29.8 / 33.8 / 72.2 % | no |
| — | ResNet-50 probe, 2 cut views (baseline) | 0.961 | 1.8 / 12.8 / 16.5 % | no |
| 9 | + CORAL (feature covariance aligned to JaalTaka train) | 0.916 | 22.9 / 45.6 / 74.2 % | no |

**Both adaptations make the check worse.** The difference between close-ups and whole notes is not a shift of feature statistics that unlabelled photos can correct: the model has simply never seen what a counterfeit looks like in a whole-note photo. **Nine approaches have now been measured.** The glass keeps policy E. The fix remains labelled whole-note counterfeit photos from the glass camera.
