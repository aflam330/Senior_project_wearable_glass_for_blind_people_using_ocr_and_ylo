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
