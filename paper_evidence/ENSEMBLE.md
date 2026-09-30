# Uncertainty-weighted ensemble (2026-09-30)

Three probabilities are averaged with weights 1 / validation log-loss: the prefix network, the attention fusion, and the MobileNetV2 watermark classifier. Weights are fit on VAL only. Serial is not in the average: it caught 0 of 19 unseen prints.

Source: `results/serial_split/beat_resnet.json`, key `ENSEMBLE`. Seeds 42, 43, 44. 222 serial-disjoint test notes.

| | 1 view | 6 views |
|---|---:|---:|
| Ensemble | 93.4 ± 0.3 % | 93.1 ± 0.3 % |
| Frozen ResNet-50 probe | 85.1 % | 83.8 % |
| Fine-tuned ResNet-50 | 92.0 ± 2.3 % | 89.9 ± 4.4 % |
| Validation-fitted hybrid (network + watermark) | 94.4 ± 0.5 % | 95.0 ± 0.0 % |

Pooled exact McNemar, ensemble versus the frozen probe: p = 5.6 × 10⁻¹⁷ at 1 view and 4.3 × 10⁻¹⁹ at 6 views. Versus the fine-tuned ResNet-50 the pooled p is 0.16 at 1 view and 0.0015 at 6 views. Pooling counts each note three times, so the per-seed tests in `BEAT_RESNET50_FINAL.md` are the ones to cite.

The ensemble beats the frozen probe by about 8 points. It does not beat the validation-fitted hybrid.
