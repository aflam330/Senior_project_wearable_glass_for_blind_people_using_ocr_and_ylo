# JaalTaka serial-number audit (2026-09-29)

**Finding: most JaalTaka counterfeit notes carry one of a few printed serial numbers. The note-disjoint split is therefore not print-disjoint, and a serial lookup alone scores 90.2 % on the test split.**

Scripts: `realtime_bangla_taka_detection/scripts/eval/jaaltaka_serial_audit.py` (OCR) and the analysis recorded in `results/jaal_whole/serial_split.json`. Raw OCR: `results/jaal_whole/serial_audit.json`.

## Method

For each of the 1,383 reconstructed whole notes (`synth_whole_notes.py`), the lower-left serial was read with EasyOCR (Bangla + English). Bangla digits were mapped to 0–9, and the longest 6–8 digit run was kept. OCR can misread one digit, so the comparison below uses the first six digits.

## Serial counts

| Class | Notes | Serial read | Distinct serials | Most common serial |
|---|---:|---:|---:|---|
| 500 BDT counterfeit | 324 | 322 | 27 | 3274658 on **279** notes (86.6 % of those read) |
| 1000 BDT counterfeit | 261 | 201 | 66 | 1843834 on 43; the near-identical 1843838 and 1843833 on 16 and 13 more (likely OCR variants of one serial) |
| 500 BDT genuine | 397 | 331 | 233 | 1.2 % |
| 1000 BDT genuine | 401 | 270 | 227 | 1.5 % |

Genuine notes have almost one serial per note, as real banknotes do. Counterfeit notes are dominated by a few printed serials. The counterfeit-set 500 BDT photos (`JAAL_VERDICT_FIXED.md`) show the same serial, 3274658.

## Effect on the test split

| Measure | Result |
|---|---:|
| Serial-lookup rule (counterfeit iff the serial's first six digits occur among TRAIN counterfeits), test accuracy, 205 notes | **0.9024** |
| Test counterfeits whose serial was seen on a TRAIN counterfeit | 68 / 87 |
| Test counterfeits with an unseen or unreadable serial | 19 / 87 |
| Genuine test notes wrongly matched to a TRAIN counterfeit serial | 1 |
| PRMVT, 1 view, counterfeits caught: serial seen / unseen | 67 / 68 (98.5 %) vs 17 / 19 (89.5 %) |
| PRMVT, 6 views, counterfeits caught: serial seen / unseen | 68 / 68 vs 17 / 19 |

## What it means

- **Serial lookup beats the CNN+ViT baseline at one view.** A reading of the serial number, which needs no knowledge of counterfeiting, reaches 90.2 %. The baseline is 73.6 % at one view and 91.8 % at six.
- **Some of the JaalTaka accuracy may be memorisation.** PRMVT catches fewer counterfeits whose serial it never saw in training (89.5 % vs 98.5 %). With 19 such notes this is suggestive, not proof, and the unseen group includes notes whose serial OCR could not read. View 1, the view used at one view, contains the lower-left serial.
- **Consequence for every JaalTaka number in this project, and for any published JaalTaka result:** the reported accuracies are on a split where most counterfeit test notes share a print serial with training notes. A **serial-disjoint split**, where no test counterfeit's serial appears in training, is the right protocol. It would leave only a few test counterfeit prints, because there are so few prints.
- **The safety policy is less affected.** Its whole-note check used a separate dataset. But that set's 500 BDT counterfeit also carries 3274658, so catching it may also be serial recognition.

## Test-time serial masking (done 2026-09-29)

Script: `scripts/eval/serial_mask_test.py`; raw: `results/jaal_whole/serial_mask_test.json`. Views 1–4 of every test note were registered to the template. The two serial boxes were mapped back into each view and filled with the view's median colour, and PRMVT (seed 42) was re-scored. There was no retraining.

| Views | Accuracy, unmasked → masked | Genuine correct | Counterfeit caught, serial seen (69) | Counterfeit caught, serial unseen (19) |
|---|---|---|---|---|
| 1 | 0.971 → 0.923 | 117 → 112 | 68 → 65 | 17 → 15 |
| 2 | 0.981 → 0.933 | 118 → 113 | 69 → 67 | 17 → 14 |
| 3 | 0.981 → 0.947 | 118 → 118 | 69 → 66 | 17 → 13 |
| 4 | 0.981 → 0.942 | 117 → 112 | 69 → 68 | 18 → 16 |

**Reading.** Masking removes 3–5 accuracy points, and genuine notes lose about as much as counterfeits. Counterfeits whose serial was seen in training stay at 65–68 of 69. If the model were recognising memorised serials, that group would collapse. It does not, so **PRMVT's accuracy is mostly not serial memorisation.** The masked patch is itself an unfamiliar input, which likely explains most of the small drop. The serial shortcut remains a flaw of the split (a lookup reaches 90.2 %), but it does not explain PRMVT's result.

## Recommended follow-up

1. Report every JaalTaka accuracy both on the standard split and on the unseen-serial counterfeits.
2. Test-time masking is done (above). Training with the serial masked would remove the remaining doubt.
3. Tell the JaalTaka authors. The data article does not mention serial duplication.
