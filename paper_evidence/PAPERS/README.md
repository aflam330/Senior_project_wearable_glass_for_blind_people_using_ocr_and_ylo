# Paper drafts by venue (index, 2026-10-13)

The four requested papers map onto drafts that already exist in `paper_evidence/`. They are not copied here, so there is one version of each. This index says what each draft has, what the results of 11–13 October add, and what is still missing before it could be sent.

No acceptance probability is given. `VENUE_READINESS.md` explains why any such number is a judgement, not a measurement.

| Requested paper | Existing draft | State |
|---|---|---|
| 1. A* workshop: watermark + print-disjoint | `WORKSHOP_PAPER.md` (109 lines), `ASTAR_WORKSHOP_PAPER.md`, `WATERMARK_PAPER.md` | Complete in its measurements. Closest to ready |
| 2. ASSETS: study with blind users | none with results | **Cannot be written.** No participant has completed the study. Protocol: `USER_STUDY_PROTOCOL.md`, `STUDY_PREREGISTRATION.md` |
| 3. Q1 journal, extended | `JOURNAL_PAPER.md` (122 lines), `PAPER_FINAL.md` (332 lines) | Complete in its measurements; needs the additions below |
| 4. IEEE Access, application | `REGIONAL_PAPER.md` (13 lines, outline only) | Outline. The thesis is the full text to cut it from |

## Results to add (all in result files)

| Result | Number | File | Fits paper |
|---|---|---|---|
| DMWR watermark classifier | 91.5 ± 0.3 % on unseen prints with 47,818 parameters; McNemar p ≥ 0.375 against MobileNetV2 (92.9 %) | `WATERMARK_DMWR.md` | 1, 3 |
| Quotient-image and matched-filter statements | Propositions 5 and 6 | `THEORY_STRONG.md` | 1, 3 |
| UV fibre check | 103 of 111 genuine notes accepted (92.8 %), blocked 5-fold; no counterfeit tested | `UV_CHECK_IMPROVED.md` | 4 (as a prototype); not 1 or 3 until counterfeits are tested |
| Whole-page reading | Greedy decoder 6.5 s against 8.9 s per 12-line page at the same validation CER; line-by-line speech | `savior_glass/results/ocr_bench/page_speed_compare.json`, `page_streaming.json` | 4 |
| Bangla speech fix | Sentences no longer cut at punctuation; long texts kept whole | `savior_glass/scripts/test_tts_queue.py` | 4 |
| Object detection, full COCO val2017 | YOLOv8s 0.551 (not 0.604); 0.493 → 0.521 after removing a needless shrink | `OBJECT_IMPROVED.md` | 4. **Corrects a number in the thesis** |
| Text reading on real photographs | see `OCR_REAL.md` | `OCR_REAL.md` | 3, 4 |

## What blocks each paper

1. **Workshop paper.** About 24 unseen counterfeit prints; the hybrid ties a fully fine-tuned ResNet-50. Honest as a "dataset flaw + safe deployment" paper; not as a "better detector" paper.
2. **ASSETS.** A completed user study. Nothing else substitutes for it at that venue.
3. **Journal.** Raspberry Pi 5 timings of the current code, and results through the glass camera.
4. **IEEE Access.** The same Pi and glass-camera results, and at least the pilot of the user study.

---

## Added 15 October 2026: results of 14-15 October and where they fit

| Result | Number | File | Fits paper |
|---|---|---|---|
| Hand-held Taka recognition with synthetic close-ups | NSTU hand-held test 11.8 → 47.5 % (Pi 8.7 → 33.5 %); Bangla Money 89.6 → 94.1 %; costs on one-taka notes reported | `HANDHELD_FINETUNE.md` | 3, 4 (a clear domain-gap result built without real training photos) |
| Recogniser trained on real words, blended with released weights | real-word CER 39.2 → 28.7 % Bangla, 62.5 → 50.6 % Latin; rendered text not worse | `OCR_REAL_IMPROVED.md` | 3, 4 |
| Voice commands for a blind user in noise | 12 unseen voices: whisper 98.9 %, noise as loud as the voice 82.5 %, other people talking 54 % (3 % without a wake word); false commands 212 → 13 | `VOICE_COMMANDS_FAST.md` | 4; 2 only with people |
| Offline Bangla command words | 45-62 % on unseen voices; not usable yet | `BANGLA_VOICE_COMMANDS.md` | 4 (as a negative result) |
| Heat on the Pi | idle 71 °C; load model T∞ = 74.8 + 0.105 × busy %; governor | `THERMAL.md` | 4 |

**Still blocking.** Unchanged: the Pi timings of the current code, the glass camera, real counterfeit prints, and the user study. The new voice and hand-held results are all on synthetic voices or other people's photographs.

**Venue probabilities.** Still not given, for the reason in `VENUE_READINESS.md`: an acceptance chance cannot be measured, and a number would be invented.
