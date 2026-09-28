# Occlusion early stop

Resume `results/qduig/occlusion_ft/seed42/checkpoint.pt`. Each training view gets its own black box, then median fill. Batch-norm and dropout stay in eval mode. An update is kept only when clean 6-view validation stays at or above the starting value 0.9375 and occluded validation improves. Test labels were not used.

Starting validation: clean 0.9375, median-fill occlusion 0.8413461538461539.

| run | clean val | occluded val | kept |
| --- | ---: | ---: | --- |
| learning rate 0.0003, full pass | 0.5769230769230769 | 0.5769230769230769 | no |
| learning rate 0.00001, full pass | 0.5769230769230769 | 0.5769230769230769 | no |
| learning rate 0.00001, frozen batch-norm, full pass | 0.5769230769230769 | 0.5769230769230769 | no |
| 40 notes, non-finite batch skipped | 0.9423076923076923 | 0.8413461538461539 | no |

The three collapsed runs predict genuine for all 208 validation notes. The 40-note run raises clean validation by one note and does not raise occluded validation, so the weights are not kept. The saved file matches the original checkpoint. Test accuracy therefore stays 0.875 with median fill. Target 0.90 was not reached.
