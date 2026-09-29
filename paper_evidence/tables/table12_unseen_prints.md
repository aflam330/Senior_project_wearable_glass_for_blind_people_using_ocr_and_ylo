| Detector (serial-disjoint test, 222 notes, 101 counterfeit from unseen prints) | 1 view | 6 views | Genuine called counterfeit (6 views) | Counterfeits missed (6 views) | Source |
|---|---:|---:|---|---|---|
| ResNet-50 probe (deterministic) | 85.1 % | 83.8 % | 1 / 121 | 35 / 101 | `results/watermark/serial_split_eval.json` |
| Prefix network, shared BN (3 seeds) | 89.9 ± 0.7 % | 89.3 ± 1.3 % | 1, 3, 2 / 121 | 26, 19, 20 / 101 | `results/watermark/hybrid_final_seeds.json` |
| Watermark window alone, MobileNetV2 (197 registered) | 92.9 % | — | 2 / 98 | 12 / 99 | `results/watermark/mobilenetv2.json` |
| **Prefix network + watermark (final, chosen on VAL)** | **94.4 ± 0.5 %** | **95.0 ± 0.0 %** | **4, 4, 4 / 121** | **7, 7, 7 / 101** | `results/watermark/hybrid_final_seeds.json` |
| Prefix network + watermark, V3 variant (not selected) | 95.5 ± 0.5 % | 96.1 ± 0.3 % | 2, 2, 2 / 121 | 7, 6, 7 / 101 | `results/watermark/hybrid_v2_seeds.json` |
| Serial blacklist | catches 0 counterfeits (unseen prints) | — | — | — | Proposition 12 |
