# UV fibre check: second evaluation (2026-10-13)

Source: `savior_glass/scripts/uv_blocked_cv.py` → `savior_glass/results/uv_check/uv_blocked_cv.json`.
First evaluation: `savior_glass/scripts/uv_dataset.py` → `savior_glass/results/uv_check/uv_fibre_check.json`.
Photographs: `data set/uv_notes/` (118 phone photographs of genuine 200, 500 and 1,000 Taka notes under a UV lamp; not in the repository).

## What was wrong with the first number

The first evaluation set the threshold on the first 60 % of each value's photographs and tested on the last 40 %: 29 of 43 accepted (67 %). The last 40 % of the 500 and 1,000 Taka photographs show the back of the note, taken from further away. The calibration part had none of those, so the threshold (14 fibres) was set for a different kind of photograph.

That test split has been read once. It was not used to tune anything here.

## What changed

Only the evaluation protocol and the threshold. The fibre finder, the note detector and `UV_REL_THRESHOLD` are the same code.

Protocol, written in the script before it was run:

- Each value's photographs, in capture order, are cut into 5 contiguous blocks, so neighbouring photographs of the same note stay together.
- For each block the threshold comes from the other four blocks, by the same rule as before: the largest threshold that accepts at least 95 % of the training notes, never below (most fibres on a plain-surface patch) + 2.
- Two variants, both reported: one threshold for all values, and one threshold per value.

## Results

111 of the 118 photographs could be judged (the detector found no note box on 7).

| Variant | Genuine notes accepted | 95 % Wilson interval | 200 Taka | 500 Taka | 1,000 Taka | Plain-surface patches rejected |
|---|---:|---:|---:|---:|---:|---:|
| First evaluation, one split | 29 / 43 (67.4 %) | 52.5–79.5 % | 19 / 19 | 7 / 16 | 3 / 8 | 21 / 21 |
| A. Blocked 5-fold, one threshold | 103 / 111 (92.8 %) | 86.4–96.3 % | 45 / 45 | 36 / 40 | 22 / 26 | 71 / 71 |
| B. Blocked 5-fold, threshold per value | 102 / 111 (91.9 %) | 85.3–95.7 % | 41 / 45 | 36 / 40 | 25 / 26 | 71 / 71 |

Thresholds per fold: variant A 4, 6, 4, 10, 6. Variant B 20–26 for 200 Taka, 4–10 for 500 Taka, 4 for 1,000 Taka.

The glass uses variant A: one number to set again on the glass camera. `UV_MIN_FIBRES = 6` is the same rule applied to all 111 photographs. It is a setting, not a test result.

Notes still rejected in variant A: four 500 Taka back-side photographs (2 to 8 fibres found) and four 1,000 Taka photographs (1 to 7 fibres).

## What these numbers do not show

- **No counterfeit note was tested.** None was available. "Accepted" means a genuine note was recognised as having fibres; it says nothing about how often a counterfeit would be caught.
- **The only "no fibres" examples are smooth.** They are patches of the plain surface beside the note. Printed paper without security fibres was not photographed, and the thresholds are low (4 to 10 fibres). In the first evaluation, added sensor noise raised the fibre count, so a noisy picture of a printed sheet might reach a low threshold. This is the main open risk.
- **Phone camera, not the glass camera.** Blur hides fibres: slight defocus cut acceptance from 67 % to 30 % in the first evaluation.
- **Few physical notes.** The photographs are bursts of a small number of notes; the number of distinct notes was not recorded.

## Not done from the list of approaches

CLAHE, bilateral filtering, wavelet and Gabor detectors, a U-Net and an ensemble vote were not tried. With no negative examples that carry print, a more sensitive detector cannot be told apart from one that counts noise, and choosing among them on these photographs would be choosing on the data already read.

## What would move this forward

1. Photographs through the glass camera with the lamp in a hood, 10–15 cm from the note, with a note ID for every physical note.
2. Printed sheets without fibres (plain and colour-printed paper) under the same lamp, as negatives with print.
3. Counterfeit notes photographed at a bank or with the police.
