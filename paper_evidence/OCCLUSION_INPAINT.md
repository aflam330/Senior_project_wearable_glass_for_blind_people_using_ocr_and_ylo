# Occlusion learned inpainting

A small U-Net reconstructor was trained on the 974 training notes, all 6 views, for 2 epochs. The hole is a random box with area fraction 0.35 to 0.55. The checkpoint is the lower validation hole-L1. The test split was not used for training or selection.

| epoch | train hole-L1 | validation hole-L1 |
| ---: | ---: | ---: |
| 1 | 0.39891456784572116 | 0.3870161101211627 |
| 2 | 0.37116601814316363 | 0.38206834396631684 |

The kept checkpoint is epoch 2. Hole L1 near 0.38 on pixels in [0, 1] is a coarse fill, not a restoration of the print.

Test, n=208, occlusion 0.55, 6 views, same box seed as the severity sweep. The filled image is classified by `results/qduig/occlusion_ft/seed42/checkpoint.pt`. Source: `results/robustness/occlusion_inpaint_seed42.json`.

6-view accuracy: 0.7259615384615384.

That is below median fill at 0.875. Target 0.90 was not reached. Inference was on the laptop GPU. Pi 5 timing was not measured.
