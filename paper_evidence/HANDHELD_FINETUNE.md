# Hand-held notes: a detector fine-tuned with synthetic close-ups (15 October 2026)

Scripts (in `realtime_bangla_taka_detection/`): `scripts/dataset/make_closeups.py`, `scripts/train_closeups.py`, `scripts/eval/eval_closeup_ft.py`, `scripts/eval/eval_handheld_int8.py`, `scripts/quantize_int8.py`.
Results: `results/closeup_ft/closeup_ft.json`, `int8_check.json`, `handheld_int8.json`; `savior_glass/results/multi_note_total.json` (earlier detector: `multi_note_total_detector_v1.json`).
Models: `models/best_handheld.pt` (laptop), `models/best_handheld_int8.onnx` (Pi). The earlier ones stay (`CURRENCY_DETECTOR=v1`).

## Problem

On NSTU-BDTAKA hand-held close-ups the detector found a note in fewer than half of the photos and named the right value in about one in eight. Its training pictures show whole notes inside scenes; a blind user holds the note close, folded, under fingers, moving.

## Method

**6,000 synthetic hand-held close-ups**, built only from note photos of the detector's training split: the note warped, turned by any angle, folded in 40 % (one part darker and foreshortened), scaled to fill or overflow the frame (longer side 1.2-2.6 times the frame in 70 %), one to three finger shapes in skin tones across it, motion blur, low resolution (down to 224-320 px and back), colour and brightness changes, noise, JPEG. Label: the box of the visible part. No NSTU photo was used to make them.

**Fine-tune** of the deployed YOLOv8s for 12 epochs on the original training pictures plus the close-ups (SGD, learning rate 0.002, mosaic 0.5, mixup 0.1, no flips), one checkpoint per epoch.

**Rules, fixed before the test sets were read:**
1. Epoch: the most correct-minus-wrong answers on NSTU *validation* hand-held photos (one per original photo, 1,611) at the glass's announce rule, among epochs whose synthetic-validation mAP@0.5 stays within 0.005 of the deployed detector's. **Epoch 8 was chosen.**
2. Announce threshold (added after the epoch scores showed many more wrong values, before any test read): the open-set rule that set 0.60 for the earlier detector, on its validation pool (all NSTU validation photos as known notes, 600 demonetized notes and 300 coins as unknown; maximise correct − wrong − unknown announced). **0.75** for the new detector.
3. Pi version: 8-bit ONNX, two calibrations (validation scenes as before; training close-ups + scenes), each with its own threshold by rule 2; the one with the higher validation utility is the Pi model. **Scene calibration, threshold 0.80.**

## Validation (NSTU hand-held, 1,611 photos, at 0.60)

| Detector | Right value | Wrong value | No answer | Note found |
|---|---:|---:|---:|---:|
| Deployed | 13.6 % | 84 | 1,308 | 45.8 % |
| Fine-tuned, epoch 1 / 4 / 8 / 12 | 40.8 / 49.8 / 57.1 / 61.1 % | 399 / 323 / 321 / 389 | | ~100 % |

Synthetic-validation mAP@0.5 stayed at 0.992-0.993 in every epoch (deployed 0.993).

## Test, read once

**Laptop (full precision)**

| Test set | Earlier detector, 0.60 | Hand-held detector, 0.75 |
|---|---:|---:|
| NSTU hand-held close-ups (1,127): right value | 11.8 % | **47.5 %** |
| — wrong value | 68 (34 % of answers) | 141 (21 % of answers) |
| — a note found at all | 44.5 % | 99.9 % |
| Bangla Money, other people's photos (1,536): right | 89.6 % | **94.1 %** |
| — wrong | 50 | **17** |
| One-taka notes (not a class, 101) announced as a value | 22.8 % | 36.6 % (worse) |
| Synthetic test mAP@0.5 / 0.5:0.95 | 0.995 / 0.849 | 0.995 / 0.848 |
| Several notes in one picture (900): total right | 99.2 % | 97.0 % (never too many; misses one note more often with three) |

**Raspberry Pi (8-bit)**

| Test set | Earlier Pi detector, 0.60 | Hand-held Pi detector, 0.80 |
|---|---:|---:|
| NSTU hand-held: right value | 8.7 % | **33.5 %** |
| — wrong value | 23 | 59 |
| Bangla Money: right / wrong | 88.7 % / 8 | 87.6 % / 10 |
| One-taka notes announced | 5.0 % | 8.9 % |

The 8-bit copy loses part of the gain: on hand-held validation photos at 0.75 it gave 38.4 % right against 49.7 % in full precision, because quantisation lowers the confidence on close-ups. Calibrating on close-ups restored the right answers (51.0 %) but shifted the scores so that it named more wrong values; by the validation rule the scene calibration with a threshold of 0.80 was kept. Its speed on the Pi is not measured (four more layers kept in full precision than in the earlier 8-bit file, 25.7 MB against 18 MB).

## Reading

- Hand-held notes, the way a blind user holds them, go from about one in eight to about one in two right on the laptop and one in three on the Pi. A note is now almost always found.
- Whole notes in other people's photos also improve on the laptop (94.1 %, fewer wrong answers). On the Pi they are level.
- The cost is unknown notes: one-taka notes are announced as a value more often (36.6 % on the laptop). The validation pool's unknown items (demonetized notes, coins) were easier than one-taka notes, as noted when 0.60 was chosen.
- Still not good enough for hand-held notes on the Pi: one answer in seven is a wrong value. The glass already asks the user to hold the note flat and closer when unsure; the counterfeit and watermark checks are unchanged.
- One crash and one session stop on the way; see `CRASH_FIXED.md`.
