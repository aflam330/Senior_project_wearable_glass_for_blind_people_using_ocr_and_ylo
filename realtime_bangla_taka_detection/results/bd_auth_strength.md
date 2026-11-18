# BD Auth Strength Report

Weights: `E:\Final SP\realtime_bangla_taka_detection\models\bd_counterfeit_auth.pt`
Device: `cuda`
Test images: 380

## Clean vs corruptions (image accuracy)

| Corruption | Image acc | Note acc | Fake recall | Real recall | ECE | Median ms |
|---|---:|---:|---:|---:|---:|---:|
| clean | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 61.3 |
| blur | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 65.4 |
| jpeg | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | 66.0 |
| dark | 0.987 | 0.987 | 1.000 | 0.964 | 0.006 | 61.5 |
| bright | 0.966 | 0.974 | 0.992 | 0.921 | 0.015 | 57.8 |
| noise | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | 62.2 |
| crop | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | 58.0 |
| rotate | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | 57.9 |

## Confidence gate (clean test)

| tau | coverage | acc when decided | false REAL on fakes | genuine confirm rate |
|---:|---:|---:|---:|---:|
| 0.50 | 1.000 | 1.000 | 0 (0.000) | 1.000 |
| 0.70 | 1.000 | 1.000 | 0 (0.000) | 1.000 |
| 0.80 | 1.000 | 1.000 | 0 (0.000) | 1.000 |
| 0.90 | 1.000 | 1.000 | 0 (0.000) | 1.000 |
| 0.95 | 1.000 | 1.000 | 0 (0.000) | 1.000 |
| 0.99 | 0.987 | 1.000 | 0 (0.000) | 1.000 |

## Quick baselines (3 epochs, clean test)

- **mobilenet_v3_small**: image_acc=0.368 fake_rec=0.000 real_rec=1.000
- **resnet18**: image_acc=0.632 fake_rec=1.000 real_rec=0.000

## Paper wording

- Do **not** claim field 95% from clean lab alone.
- Report clean + worst corruption + gated false-REAL rate.
- Glass speaks REAL only above tau; else check-by-hand (never 'counterfeit').
