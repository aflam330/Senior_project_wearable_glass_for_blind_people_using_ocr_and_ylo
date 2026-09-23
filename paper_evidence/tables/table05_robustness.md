# Table 5. Robustness

Source: `results/qduig/eval/seed42/robustness.json`

| corruption | baseline_acc | proposed_acc | Δ baseline | Δ proposed |
|---|---|---|---|---|
| clean | 0.9183 | 0.9663 | NOT_MEASURED | NOT_MEASURED |
| gaussian_blur | 0.9183 | 0.9615 | 0.0000 | -0.0048 |
| motion_blur | 0.9135 | 0.9663 | -0.0048 | 0.0000 |
| low_light | 0.6154 | 0.5865 | -0.3029 | -0.3798 |
| brightness | 0.6587 | 0.7981 | -0.2596 | -0.1683 |
| contrast | 0.9231 | 0.9423 | 0.0048 | -0.0240 |
| glare | 0.9038 | 0.8942 | -0.0144 | -0.0721 |
| occlusion | 0.7981 | 0.6683 | -0.1202 | -0.2981 |
| heavy_occlusion | 0.8462 | 0.6683 | -0.0721 | -0.2981 |
| jpeg | 0.9135 | 0.9760 | -0.0048 | 0.0096 |
| rotation | 0.8750 | 0.8798 | -0.0433 | -0.0865 |
| perspective | 0.8846 | 0.9231 | -0.0337 | -0.0433 |
| scale | 0.9135 | 0.9663 | -0.0048 | 0.0000 |
| sensor_noise | 0.9135 | 0.9567 | -0.0048 | -0.0096 |

