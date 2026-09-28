# Occlusion fine-tune (attempt C, 2026-09-28)

**Verdict: PARTIAL.** 55% occlusion, 6 views, test: 51.9% (PRMVT) → **88.5%** (seed 42),
**87.0 ± 2.2%** over three seeds, **88.9%** for the validation-chosen three-seed ensemble. Clean
accuracy does not drop. The target of 90% was not reached on test.

## Recipe

Script: `realtime_bangla_taka_detection/scripts/train/train_occlusion_robust.py --epochs 15 --cosine`.
The recipe was chosen before training, from the diagnosis in `DIAGNOSIS_OCCLUSION.md`:
- Start from the deployed PRMVT (`results/qduig/prefix_ft/seed<N>`).
- Each training view gets an independent black box with probability 0.7, area drawn from U(0.2, 0.65).
- Views are a random prefix half the time, to keep 1-view accuracy.
- The last 4 MobileNet blocks are unfrozen (lr 1×10⁻⁵); the rest trains at lr 3×10⁻⁵ (AdamW), with cosine decay over 15 epochs.
- BatchNorm and Dropout stay in eval mode.
- Plain cross-entropy loss.

Training reads only the train split. Each epoch is scored on validation (clean 1 view, clean 6 views, and 55% occlusion at 6 views with the test protocol's seeding). An epoch is **eligible** only if clean 6-view validation stays ≥ 0.9375 and clean 1-view validation drops no more than 2 points. The selected epoch is the eligible one with the best occluded validation accuracy. Zero non-finite gradient batches occurred in any run.

## Validation (selection) and test (evaluated once per seed)

| Seed | Epoch selected | Val clean 6v | Val occ 55% | Test clean 1v | Test clean 6v | Test occ 55% |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 13 | 99.0 | 91.8 | 98.1 | 98.1 | **88.5** |
| 43 | 12 | 99.0 | 90.9 | — | 99.0 | **88.9** |
| 44 | 12 | 99.0 | 88.9 | — | 99.0 | **84.6** |
| mean ± SD | | | | | 98.7 | **87.0 ± 2.2** |

Before the fine-tune, PRMVT scored 51.9% at 55% occlusion and 98.1% clean on test (6 views); the baseline scored 81.3% occluded.

Seed 42, test, from `results/robustness/occlusion_robust_e15_seed42.json` (same `eval_robust_checkpoint.py` as every earlier occlusion number):
- Clean: 98.1% at 1 view and 98.1% at 6 views.
- Low light 0.2 at 6 views: 62.0%. PRMVT scored 58.7% in the severity sweep.
- 55% occlusion at 6 views: 88.5%.

Seeds 43 and 44, and the ensemble, come from `results/robustness/occlusion_decision.json`.

Earlier attempts, for comparison (test, 55% occlusion, 6 views):

| Attempt | File | Test |
|---|---|---:|
| Occlusion fine-tune, lr 3×10⁻⁴, 2 epochs | `OCCLUSION_FIX_RESULTS.md` | 68.3 |
| Median fill on that fine-tune | `OCCLUSION_FIX_RESULTS.md` | 87.5 |
| Learned inpainting (A) | `OCCLUSION_INPAINT.md` | 72.6 |
| Matched fine-tune, collapsed on NaN gradients | `OCCLUSION_FIX_RESULTS.md` | n/a |
| **This fine-tune, seed 42** | this file | **88.5** |

Caveat: the benchmark's occluder is a solid black box. A finger or shadow is not black and not rectangular, so this number does not transfer to real occlusion. Real-world occlusion needs photos of hand-held notes.
