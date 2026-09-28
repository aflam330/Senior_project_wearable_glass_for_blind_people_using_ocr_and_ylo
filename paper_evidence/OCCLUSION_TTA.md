# Occlusion test-time crops

No training. Checkpoint `results/qduig/occlusion_ft/seed42/checkpoint.pt`. Occlusion 0.55, median fill, then the full frame and four 70% corner crops. The crop is chosen by confidence, without labels. Test n=208, 6 views. Source: `results/robustness/occlusion_tta_seed42.json`.

| rule | accuracy |
| --- | ---: |
| full frame | 0.875 |
| mean of the five crops | 0.8653846153846154 |
| most confident crop | 0.8557692307692307 |

Target 0.90 was not reached. Crops score below the full frame.

---

## Update 2026-09-28: horizontal-flip averaging on the occlusion-robust model

Averaging each note's prediction with its mirror image was used only where it raised occluded validation accuracy. That was seed 42 only (91.8% with flip on validation). On test it gave 87.5%, against 88.5% without flip. See `OCCLUSION_ENSEMBLE.md`. Test-time augmentation does not close the gap to 90%.
