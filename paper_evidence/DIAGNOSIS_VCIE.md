# Diagnosis: VCIE

Target is test accuracy at least 0.9182692307692307, the CNN+ViT baseline at 6 views. Standing v2 is below that at 6 views and the gap is one note.

## Checkpoint

`results/novel_v2/vcie/seed42/test/test_metrics.json`, seed 42, n=208.

| views | v2 accuracy |
| ---: | ---: |
| 1 | 0.8798076923076923 |
| 2 | 0.875 |
| 3 | 0.9086538461538461 |
| 4 | 0.8990384615384616 |
| 5 | 0.9038461538461539 |
| 6 | 0.9134615384615384 |

6-view is 0.00480769230769229 below the baseline. 1-view is above the baseline 1-view score of 0.7355769230769231.

## Training issue

An 8-epoch continuation selected the checkpoint by the mean of validation views 1 through 6. The best mean was epoch 2. That epoch's validation 1-view accuracy was 0.7548076923076923. Test 1-view of that run is 0.7548076923076923 (`results/novel_v2/vcie_long/seed42/test/test_metrics.json`). Test 6-view is 0.9086538461538461, farther from the baseline than v2.

The failure of that continuation is checkpoint selection, not a missing loss curve. Gradient norms and feature drift for VCIE were not logged. The architecture in v2 uses two set-encoder blocks. A 4-block or 6-block model was not the checkpoint that was selected.

## Root cause

Mean-over-views selection keeps a high-mean, low-1-view epoch. That does not by itself explain the 6-view gap of one note. v2, which is the standing model, is already 0.9134615384615384 at 6 views. Closing 0.9182692307692307 requires one more correct test note. Whether 1-view selection raises 6-view accuracy is not implied by the 8-epoch run; that run lowered both.
