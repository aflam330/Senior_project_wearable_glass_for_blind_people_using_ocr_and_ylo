# Learned watermark localizer (2026-10-01)

**What changed.** The watermark window used to be found by registering the back-lit photo to a 500 or 1,000 Taka template (SIFT + RANSAC) and cutting a fixed box, (0.70, 0.33, 0.93, 0.85). A MobileNetV3-Small now regresses the window's four corners directly from the 320 × 320 photo. There is no template, no SIFT and no per-denomination box at run time. The crop goes to the published MobileNetV2 watermark classifier, unchanged.

**Honest scope.**
- **Labels.** The corners are the registration box mapped back into each photo. The localizer learned where that box is. It did not discover the window on its own.
- **Data.** It is one currency and the same ~24 unseen counterfeit prints as before.
- **What it removes.** The template from the device path, and SIFT failures.
- **What it does not show.** That the method transfers to another currency. That needs the new dataset (`PRINT_DISJOINT_DATASET_PLAN.md`), with windows clicked by hand (`scripts/dataset/annotate_watermark.py`). The training script already accepts that dataset.

## Setup

**Code.**
- Labels: `scripts/eval/watermark_localizer_labels.py`. Training and test: `scripts/train/train_watermark_localizer.py <seed>`.
- Output: `results/watermark_localizer/seed{42,43,44}.json`.

**Split and labels.**
- Serial-disjoint split (unseen prints).
- Registration gives a label on train 867/946, val 197/222 and test 197/222 photos.

**Training and selection.**
- 40 epochs with affine and photometric augmentation.
- The epoch is chosen on mean validation corner error. The test is read once per seed.

## Results (test, unseen prints, three seeds)

| | Seed 42 | Seed 43 | Seed 44 | Mean |
|---|---:|---:|---:|---:|
| Mean corner error (fraction of photo) | 0.025 | 0.025 | 0.025 | 0.025 |
| Median quad IoU with the registration window | 0.925 | 0.927 | 0.929 | 0.927 |
| Accuracy, localizer crop, 197 registered notes | 92.4 % | 92.4 % | 92.9 % | 92.6 % |
| Accuracy, registration crop, same 197 notes | 92.9 % | 92.9 % | 92.9 % | 92.9 % |
| McNemar p, localizer vs registration | 1.0 | 1.0 | 1.0 | — |
| Accuracy, 25 notes registration could not read | 92.0 % | 92.0 % | 96.0 % | 93.3 % |
| **Accuracy, all 222 test notes** | 92.3 % | 92.3 % | 93.2 % | **92.6 ± 0.5 %** |
| Genuine called counterfeit (of 121) | 6 | 5 | 4 | 5.0 |
| Counterfeit passed (of 101) | 11 | 12 | 11 | 11.3 |

**What the table shows.**
- **Accuracy.** The localizer ties registration on the notes both can read (no significant difference on any seed).
- **Coverage.** It adds the 25 test notes (11 %) where SIFT found no window. Coverage goes from 197 to 222 of 222 at the same accuracy.
- **Watermark alone versus the hybrid.** Alone the watermark is below the six-view hybrid (95.0 %) and the full-resolution ResNet-50 (94.4 %). It is one input of the hybrid, not a replacement.

## Device path

**Configuration.** `savior_glass/config.py`:
- `WATERMARK_LOCALIZER_PATH` points to the seed chosen on validation corner error: seed 42, 4.3 MB FP32 ONNX.
- SIFT stays as the fallback.

**Plausibility check.** `WatermarkChecker` rejects an implausible quad (not convex, or area outside 0.5–50 % of the photo) and asks the user to hold the note straight.

**App path on the 222 test photos.** This is the localizer ONNX plus the MobileNetV2 INT8, the Pi configuration.
- The window was found on all 222 photos.
- Accuracy was 93.2 %.
- It made the same decision as the research FP32 path on 99.1 % of photos.
- Source: `results/watermark_localizer/app_path_check.json`.

**Laptop CPU latency.** 60 test photos, AMD laptop, not the Pi (`laptop_latency.json`):

| Path | Median | 95th percentile |
|---|---:|---:|
| Localizer path | 15.9 ms | 19.2 ms |
| SIFT path | 97.9 ms | 121.5 ms |

**Pi benchmark.** `benchmark_pi5.py` times both paths separately (`watermark_int8_localizer`, `watermark_int8_sift`).

**Speech.** The glass wording is unchanged. It never says "counterfeit".
