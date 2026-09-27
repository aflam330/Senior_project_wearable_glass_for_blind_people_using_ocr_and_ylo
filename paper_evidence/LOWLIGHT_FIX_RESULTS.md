# Low light at multiplier 0.2

Saved PRMVT, 6 views, test accuracy 0.5769230769230769. Source: `results/robustness/severity_seed42.json`.

## Inference repair

Gamma, gain, and CLAHE were scored on the validation split after the same low-light corruption. The selected repair is gamma 0.4, validation accuracy 0.8365384615384616. Test accuracy is 0.8317307692307693. Source: `results/robustness/lowlight_enhance_seed42.json`.

## Fine-tune with darkening and patch occlusion

Three epochs from the saved PRMVT checkpoint, new directory `results/qduig/robust_ft/seed42`. The epoch was chosen by clean validation accuracy, not by the test corruption. Test results in `results/robustness/robust_ft_seed42.json`:

| condition | accuracy |
|---|---:|
| clean, 1 view | 0.9567307692307693 |
| clean, 6 views | 0.9663461538461539 |
| low light 0.2, 6 views | 0.9375 |

Low light 0.2 is above 0.90. Clean 6-view accuracy of this checkpoint is 0.9663461538461539, below the untouched PRMVT checkpoint at 0.9759615384615384.
