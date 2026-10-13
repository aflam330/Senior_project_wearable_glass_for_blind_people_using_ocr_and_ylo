# UV fibre check against fibre-free printed sheets: a simulation, a failure, and a repair (2026-10-13)

Sources: `savior_glass/scripts/uv_stress_sim.py` → `results/uv_check/uv_stress_sim.json`; `scripts/uv_robust_study.py` → `results/uv_check/uv_robust_study.json` and `uv_robust_brightness_failed.json`.

**No counterfeit note and no real printed sheet was photographed under the UV lamp.** The fibre-free side of everything below is simulated. The genuine side is the 111 real UV photographs.

## 1. How a fibre-free sheet is simulated

- **Print:** normal-light photographs of genuine 200, 500 and 1,000 Taka notes (40 per value). Security fibres are invisible in normal light, so this is real print without fibres.
- **Under the lamp:** the sheet reflects the lamp's blue light with the green/blue and red/blue ratios measured on the real UV photographs (0.204 and 0.045), at their brightness (median blue level 111).
- **Paper glow:** on unprinted paper, extra green of strength d × blue. The plain surface in the real photographs is about 0.05 above the notes.
- **Camera:** noise at the level measured on the real photographs (σ = 0.40 at the scale the check works at), then JPEG compression.

These are assumptions. A real counterfeit's ink and paper may behave differently under UV.

## 2. The check as it was fails this test

| Simulated sheet (120 sheets) | Wrongly accepted as "UV marks found" | Fibres counted, median |
|---|---:|---:|
| Dull paper, no noise, no JPEG | 0 | 0 |
| Dull paper, measured noise | 0 | 0 |
| Dull paper, measured noise, JPEG 90 | 112 (93 %) | 28 |
| Dull paper, measured noise, JPEG 50 | 120 (100 %) | 148 |
| Dull paper, 8 × noise, JPEG 90 | 115 (96 %) | 37 |
| Glowing paper d = 0.10, JPEG 90 | 120 (100 %) | 121 |

Noise alone does nothing. **JPEG compression alone makes the check accept nearly every fibre-free printed sheet.** Compression stores colour at a lower resolution than brightness, which smears colour across the edges of the print; the check counted those smears as fibres.

This also changes the reading of the earlier result. The 92.8 % of genuine photographs "accepted" (`UV_CHECK_IMPROVED.md`) measured that thin coloured structures are found on a note. A printed copy has them too.

## 3. Repairs tried

**Brightness condition (failed).** A fibre should also be brighter than its surroundings. No threshold separated the two sides: balanced accuracy 0.50 in every fold.

**Colour condition (kept).** A fibre pixel must be less blue than the paper by more than a threshold, and lie on lit paper (not in dark ink, where a small colour error is a large share of a small blue value). Explored on the development sheets first:

| Colour threshold | Genuine photographs: fibres at 5th / 50th / 95th percentile | Development sheets: 50th / 95th / max |
|---:|---:|---:|
| none | 6.5 / 41 / 254 | 162 / 426 / 611 |
| 0.10 | 2.5 / 31 / 240 | 47 / 204 / 417 |
| 0.20 | 0 / 5 / 50 | 0 / 4 / 14 |
| 0.30 | 0 / 1 / 12 | 0 / 0 / 1 |

With a colour threshold of 0.2, a genuine note has a median of 5 strongly coloured fibres, not 41: most of the earlier count was not fluorescence.

## 4. Fair measurement of the colour condition

Protocol, written in the script before the run: the 111 genuine photographs in the same five contiguous blocks as before; 60 development and 60 held-out sheets, each under four conditions (240 held-out pictures); for each fold the colour threshold and the fibre count T are chosen together on the four training blocks and the development sheets, by balanced accuracy, and applied to the held-out block and the held-out sheets.

| | Check as it was (T = 6) | With the colour condition (nested) |
|---|---:|---:|
| Genuine photographs accepted | 103 / 111 (92.8 %) | 95 / 111 (85.6 %, Wilson 77.9–90.9) |
| &nbsp;&nbsp;200 / 500 / 1,000 Taka | 45/45, 36/40, 22/26 | 44/45, 31/40, 20/26 |
| Held-out simulated sheets rejected | 4 / 240 (1.7 %) | 88.6 % (mean over the five folds) |

Chosen per fold: colour threshold 0.20 or 0.22, T = 1.

Setting now in the glass (`UV_COLOUR_THRESHOLD = 0.22`, `UV_MIN_FIBRES = 1`, chosen on all genuine photographs and the development sheets): 223 of 240 held-out sheets rejected.

| Held-out sheets, by condition | Rejected |
|---|---:|
| Measured noise, JPEG 90 | 60 / 60 |
| Measured noise, JPEG 50 | 48 / 60 |
| Glow 0.10, JPEG 90 | 60 / 60 |
| Glow 0.20, JPEG 90 | 55 / 60 |

## 5. What this is and is not

- It is a demonstration that the first design could not tell a note from a printed copy, and a repaired design that can in a simulation, at a cost of 7 points of genuine acceptance.
- The genuine side of the nested figures is not fully blind: the colour condition was designed after looking at fibre counts on all genuine photographs. The held-out sheets were not looked at before the run.
- T = 1 means a single strongly coloured fibre decides. That is fragile, and it is what these photographs support: they are phone screenshots, already compressed once, taken by hand.
- AUC is not reported: with T = 1 chosen in every fold the decision is one point on a curve, and a curve over simulated negatives would look more precise than it is.
- **It is not evidence about counterfeit notes.** The check stays switched off by default and still says "check by hand", never "counterfeit".

## 6. What would settle it

Photographs under the UV lamp, through the glass camera and saved without lossy compression, of: genuine notes (with a note ID), plain and colour-printed paper, and counterfeit notes at a bank or with the police.
