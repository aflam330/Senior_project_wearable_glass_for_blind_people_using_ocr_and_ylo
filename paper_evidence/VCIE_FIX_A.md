# VCIE fix A: checkpoint by 1-view validation

v2 was left in place. This run resumes it and keeps an epoch only when validation 1-view accuracy rises. Seed 42. Four epochs, learning rate 0.0003. Test labels were not used for selection.

Validation 1-view of the loaded v2 weights: 0.875. Epoch 4 reached validation 1-view 0.9182692307692307 and is the saved checkpoint. Source: `results/novel_v2/vcie_k1/seed42/val_metrics.json`.

Test n=208. Source: `results/novel_v2/vcie_k1/seed42/test/test_metrics.json`.

| views | accuracy |
| ---: | ---: |
| 1 | 0.9278846153846154 |
| 2 | 0.9134615384615384 |
| 3 | 0.9278846153846154 |
| 4 | 0.9375 |
| 5 | 0.9375 |
| 6 | 0.9278846153846154 |

1-view is above v2 (0.8798076923076923). 6-view is above v2 (0.9134615384615384) and above the CNN+ViT baseline 0.9182692307692307. Target reached. Separate heads, a deeper set transformer, and extra residual blocks were not trained.
