# CAMVA experiment report

Generated: 2026-09-21T20:52:48.453597+00:00

Every number below is copied from a file under `results/camva/`. If a section says **NOT MEASURED**, the experiment was not run.

## Dataset and split

- Dataset: JaalTaka (`real_notes` / `fake_notes`)
- Independent unit: physical note ID (`genuine:note_XXX` / `counterfeit:note_XXX`)
- All six views of one note stay in exactly one of train/val/test
- Seed and counts: see `splits/split_metadata.json`

## Training

See `configs/train_config.json`. Checkpoints in `checkpoints/` — **does not overwrite** `models/authenticity_cnn_vit.pt`.

## Baseline vs CAMVA (1–6 views)

| views | baseline acc | CAMVA acc | abs Δ | paired p |
|---:|---:|---:|---:|---:|
| 1 | 0.7356 | 0.5337 | -0.2019 | 0.0000 |
| 2 | 0.8702 | 0.5721 | -0.2981 | 0.0000 |
| 3 | 0.9135 | 0.6923 | -0.2212 | 0.0000 |
| 4 | 0.9183 | 0.7067 | -0.2115 | 0.0000 |
| 5 | 0.8990 | 0.7404 | -0.1587 | 0.0000 |
| 6 | 0.9183 | 0.9663 | 0.0481 | 0.0260 |


Predictions: `predictions/baseline_{k}view.json` and `predictions/camva_{k}view.json`.

## Ablations / adaptive / calibration / robustness

- Ablations: `metrics/ablations.json` or NOT MEASURED
- Adaptive thresholds: `metrics/adaptive.json` or NOT MEASURED
- Calibration: `calibration/` (T fitted on **val** only)
- Robustness: `metrics/robustness.json` or NOT MEASURED

## Latency

- Device used in training config: cuda
- RTX 4060 Laptop: NOT MEASURED (this machine may differ)
- Raspberry Pi 5: NOT MEASURED

## Not claimed

- SOTA
- Automatic improvement (read the table)
- User-study outcomes
- Cross-camera generalization (no camera IDs in JaalTaka)

## Limitations

- Offline confidence ordering peeks at stored views to rank them; sequential **stopping** still uses only the prefix already fused.
- Single-seed training unless `train_summary.json` lists three seeds.
