# Data pipeline bug (2026-09-30)

## What the bug was

`scripts/train/cache_views_256.py` originally decoded each JaalTaka view with `cv2.IMREAD_REDUCED_COLOR_4`, then resized the short side to 256. On `genuine:note_001` view 1 the file is 1672×1929. The quarter decode is 418×483. The fine-tune then resized that cache to 224×224.

The published fine-tune in `results/serial_split/ft_resnet50/` used that cache. The current script writes `cache/views256_full` with `cv2.IMREAD_COLOR` and leaves the old cache in place.

## One-image check, same output size

On that same view, Laplacian variance after a short-side resize to 256:

| Decode | Size before the 256 resize | Laplacian variance at 256 |
|---|---|---:|
| Full JPEG | 1672×1929 | 1814 |
| Quarter JPEG (`IMREAD_REDUCED_COLOR_4`) | 418×483 | 1047 |

This is one genuine view, not a test-set metric. It shows the training images were softer. Laplacian variance is not comparable across different pixel grids, so the full-frame and quarter-frame variances are not reported as a loss ratio.

## Which models used it

| Model | Loader | Affected by this bug? |
|---|---|---|
| Fine-tuned ResNet-50, serial-disjoint | `cache/views256` | **Yes.** Retrained; see below |
| Frozen ResNet-50 probe | `results/sota/feats_resnet50.npz`, full `cv2.imread` path in the feature scripts | No. Not retrained |
| Prefix network | `roboeye/camva/data.py` `cached_resized_rgb`: full `cv2.imread`, then resize to 128 | No |
| Watermark MobileNetV2 | Crops from `watermark_features.py`: `IMREAD_REDUCED_COLOR_2`, then width 700, then a 224 crop | **Not the quarter-decode bug.** A full-decode rerun was measured; it did not help. See below |
| Whole-note composites | `synth_whole_notes.py`: `IMREAD_REDUCED_COLOR_2`, then width 700 | Same half-size path. Not this bug |

The prefix network and the watermark detector were not trained on the quarter cache. Their published numbers are not revised.

## Measured effect on the fine-tune

Serial-disjoint test, 222 notes, seeds 42–44, epoch chosen on validation. Sources: `beat_resnet.json`, `hybrid_ft_fullres.json`.

| | 1 view | 6 views |
|---|---:|---:|
| Quarter-decoded cache, 5 epochs | 92.0 ± 2.3 % | 89.9 ± 4.4 % |
| Full JPEG cache, 8 epochs | 94.9 ± 1.8 % | 94.4 ± 2.6 % |
| Frozen probe (unchanged) | 85.1 % | 83.8 % |
| Watermark hybrid (unchanged) | 94.4 ± 0.5 % | 95.0 ± 0.0 % |

Exact McNemar, full-resolution fine-tune versus the frozen probe: p ≤ 1.2×10⁻⁴ on every seed and both view counts. Mean gain +9.8 points at one view and +10.7 points at six views.

Versus the quarter-resolution fine-tune, six views, seeds 42 and 43: p = 4.2×10⁻⁷ and 0.002. Seed 44 at six views is lower (91.4 % versus 93.7 %, p = 0.13). Its best validation epoch was epoch 1; later epochs fell, so epoch 1 was kept.

## What this does to the paper claim

Earlier text that compared the watermark hybrid with a fine-tuned ResNet-50 was a comparison with an under-resolved fine-tune. Against the full-resolution fine-tune the hybrid is +0.5 points at six views and is not significant (per-seed McNemar p = 0.38, 1.0, 0.057, the last in the hybrid's favour on seed 44). The hybrid remains the most stable six-view system (95.0 ± 0.0 %). It is not a 5-point win over a properly trained ResNet-50.

Figures: `figures/fig28_pipeline_bug.png`, `figures/fig29_fullres_vs_hybrid.png`.

## Watermark crops, full JPEG decode (measured, not kept)

`python scripts/eval/watermark_features.py full` decodes view 6 at full size and still scales the width to 700, which is the registration canvas. Output: `results/watermark_fullres/`. The published crops were not overwritten.

| Decode | Notes that registered |
|---|---:|
| Half JPEG (`IMREAD_REDUCED_COLOR_2`) | 1,261 of 1,390 |
| Full JPEG | 1,255 of 1,390 |

MobileNetV2, 10 epochs, epoch chosen by validation AUC, seeds 42–44. Files: `watermark_fullres/mobilenetv2_seed{42,43,44}.json`. ONNX export failed because `onnx` is not installed in this interpreter; the accuracy below is from PyTorch.

| | Accuracy | AUC | Genuine flagged | Counterfeits missed |
|---|---:|---:|---|---|
| Published half-decode model (one run) | 92.9 % | 0.976 | 2 / 98 | 12 / 99 |
| Full decode, three seeds | 90.5 ± 1.2 % | 0.979, 0.978, 0.979 | 1 / 100 each seed | 19, 19, 15 of 97 |

The registered test notes are not the same set (98 genuine and 99 counterfeit versus 100 and 97), so this is not a paired test. The full decode does not beat the published watermark model. The published 2.6 MB INT8 model stays the one to cite. The prefix network was not retrained: `cached_resized_rgb` already uses a full `cv2.imread` and then resizes to 128. Figure 30.
