# All three-seed results with 95 % confidence intervals (2026-09-30)

Seeds 42, 43, 44. 95 % CI = mean ± t(0.975, 2) · sd / √3, with t = 4.303. **With three seeds this interval is wide by construction**; it describes seed-to-seed variation only, not test-set sampling (for that, see the Wilson intervals and paired tests in the linked files). Generated from the result files listed.

| Result | Per seed (%) | Mean ± sd | 95 % CI | Source |
|---|---|---:|---|---|
| CNN+ViT baseline (CAMVA repo), 1 view (standard split) | 73.6, 80.3, 74.5 | 76.1 ± 3.6 | [67.1, 85.2] | `BENCHMARK_FINAL.md` |
| CNN+ViT baseline (CAMVA repo), 6 views (standard split) | 91.8, 93.3, 91.8 | 92.3 ± 0.8 | [90.2, 94.4] | `BENCHMARK_FINAL.md` |
| CAMVA quality attention, 1 view (standard split) | 53.4, 51.0, 57.7 | 54.0 ± 3.4 | [45.5, 62.5] | `BENCHMARK_FINAL.md` |
| CAMVA quality attention, 6 views (standard split) | 96.6, 98.1, 96.6 | 97.1 ± 0.8 | [95.0, 99.2] | `BENCHMARK_FINAL.md` |
| PRMVT stage 1 (prefix, 6 epochs), 1 view (standard split) | 91.3, 93.3, 95.2 | 93.3 ± 1.9 | [88.5, 98.0] | `BENCHMARK_FINAL.md` |
| PRMVT stage 1 (prefix, 6 epochs), 6 views (standard split) | 90.9, 92.3, 98.1 | 93.8 ± 3.8 | [84.3, 103.2] | `BENCHMARK_FINAL.md` |
| PRMVT (prefix + fine-tune), 1 view (standard split) | 97.1, 94.7, 97.6 | 96.5 ± 1.5 | [92.6, 100.3] | `BENCHMARK_FINAL.md` |
| PRMVT (prefix + fine-tune), 6 views (standard split) | 98.1, 97.1, 99.0 | 98.1 ± 1.0 | [95.7, 100.5] | `BENCHMARK_FINAL.md` |
| Same network, fixed 6-view, per-count BN, 1 view (standard split) | 76.0, 93.3, 92.3 | 87.2 ± 9.7 | [63.0, 111.3] | `BENCHMARK_FINAL.md` |
| Same network, fixed 6-view, per-count BN, 6 views (standard split) | 99.0, 99.0, 99.0 | 99.0 ± 0.0 | [99.0, 99.0] | `BENCHMARK_FINAL.md` |
| Same network, fixed 6-view, shared BN, 1 view (standard split) | 66.3, 63.0, 84.1 | 71.2 ± 11.4 | [42.9, 99.4] | `BENCHMARK_FINAL.md` |
| Same network, fixed 6-view, shared BN, 6 views (standard split) | 97.6, 98.1, 98.6 | 98.1 ± 0.5 | [96.9, 99.3] | `BENCHMARK_FINAL.md` |
| Same network, prefix, shared BN, 1 view (standard split) | 97.6, 98.1, 99.0 | 98.2 ± 0.7 | [96.4, 100.1] | `BENCHMARK_FINAL.md` |
| Same network, prefix, shared BN, 6 views (standard split) | 98.6, 99.0, 98.6 | 98.7 ± 0.3 | [98.0, 99.4] | `BENCHMARK_FINAL.md` |
| PRMVT checkpoints, score-level late fusion, 1 view (standard split) | 97.1, 94.7, 97.6 | 96.5 ± 1.5 | [92.6, 100.3] | `BENCHMARK_FINAL.md` |
| PRMVT checkpoints, score-level late fusion, 6 views (standard split) | 93.8, 92.8, 97.6 | 94.7 ± 2.5 | [88.4, 101.0] | `BENCHMARK_FINAL.md` |
| CVS, 1 view (standard split) | 93.3, 97.1, 95.7 | 95.4 ± 1.9 | [90.5, 100.2] | `BENCHMARK_FINAL.md` |
| CVS, 6 views (standard split) | 97.1, 97.1, 96.2 | 96.8 ± 0.6 | [95.4, 98.2] | `BENCHMARK_FINAL.md` |
| NDAL, 1 view (standard split) | 98.1, 96.6, 96.2 | 97.0 ± 1.0 | [94.5, 99.4] | `BENCHMARK_FINAL.md` |
| NDAL, 6 views (standard split) | 96.6, 94.2, 96.6 | 95.8 ± 1.4 | [92.4, 99.3] | `BENCHMARK_FINAL.md` |
| PRAVT, 1 view (standard split) | 97.6, 95.7, 95.2 | 96.2 ± 1.3 | [93.0, 99.3] | `BENCHMARK_FINAL.md` |
| PRAVT, 6 views (standard split) | 96.2, 98.1, 95.2 | 96.5 ± 1.5 | [92.8, 100.1] | `BENCHMARK_FINAL.md` |
| VAT, 1 view (standard split) | 97.1, 98.1, 96.6 | 97.3 ± 0.7 | [95.5, 99.1] | `BENCHMARK_FINAL.md` |
| VAT, 6 views (standard split) | 96.6, 96.2, 96.2 | 96.3 ± 0.3 | [95.6, 97.0] | `BENCHMARK_FINAL.md` |
| MTPT (6+3, selected on VAL 1-view), 1 view (standard split) | 98.1, 96.6, 95.7 | 96.8 ± 1.2 | [93.8, 99.8] | `BENCHMARK_FINAL.md` |
| MTPT (6+3, selected on VAL 1-view), 6 views (standard split) | 96.6, 97.1, 96.6 | 96.8 ± 0.3 | [96.1, 97.5] | `BENCHMARK_FINAL.md` |
| VCIE (k1 fine-tune), 1 view (standard split) | 92.8, 92.3, 95.2 | 93.4 ± 1.5 | [89.6, 97.3] | `BENCHMARK_FINAL.md` |
| VCIE (k1 fine-tune), 6 views (standard split) | 92.8, 93.8, 93.3 | 93.3 ± 0.5 | [92.1, 94.5] | `BENCHMARK_FINAL.md` |
| APC, 1 view (standard split) | 96.2, 96.6, 98.1 | 97.0 ± 1.0 | [94.5, 99.4] | `BENCHMARK_FINAL.md` |
| APC, 6 views (standard split) | 96.6, 96.2, 95.7 | 96.2 ± 0.5 | [95.0, 97.3] | `BENCHMARK_FINAL.md` |
| CRIS, 1 view (standard split) | 95.2, 94.2, 95.2 | 94.9 ± 0.6 | [93.5, 96.3] | `BENCHMARK_FINAL.md` |
| CRIS, 6 views (standard split) | 91.8, 94.7, 94.2 | 93.6 ± 1.5 | [89.8, 97.4] | `BENCHMARK_FINAL.md` |
| MAVT, 1 view (standard split) | 91.8, 93.8, 95.2 | 93.6 ± 1.7 | [89.4, 97.8] | `BENCHMARK_FINAL.md` |
| MAVT, 6 views (standard split) | 95.7, 95.2, 93.8 | 94.9 ± 1.0 | [92.4, 97.4] | `BENCHMARK_FINAL.md` |
| SAVS, 1 view (standard split) | 93.8, 94.2, 92.8 | 93.6 ± 0.7 | [91.8, 95.4] | `BENCHMARK_FINAL.md` |
| SAVS, 6 views (standard split) | 93.3, 94.2, 91.3 | 92.9 ± 1.5 | [89.3, 96.6] | `BENCHMARK_FINAL.md` |
| Unseen prints, prefix network, 1 view(s) | 90.1, 89.2, 90.5 | 89.9 ± 0.7 | [88.2, 91.6] | `results/watermark/hybrid_final_seeds.json` |
| Unseen prints, network + watermark (final), 1 view(s) | 94.1, 94.1, 95.0 | 94.4 ± 0.5 | [93.2, 95.7] | `results/watermark/hybrid_final_seeds.json` |
| Unseen prints, prefix network, 6 view(s) | 87.8, 90.1, 90.1 | 89.3 ± 1.3 | [86.1, 92.6] | `results/watermark/hybrid_final_seeds.json` |
| Unseen prints, network + watermark (final), 6 view(s) | 95.0, 95.0, 95.0 | 95.0 ± 0.0 | [95.0, 95.0] | `results/watermark/hybrid_final_seeds.json` |
| MVP-N attn fixed, 1 view(s) | 41.7, 39.7, 40.6 | 40.7 ± 1.0 | [38.2, 43.1] | `results/mvpn/vcds.json` |
| MVP-N attn fixed, 6 view(s) | 76.0, 75.9, 75.5 | 75.8 ± 0.3 | [75.0, 76.5] | `results/mvpn/vcds.json` |
| MVP-N attn prefix, 1 view(s) | 48.0, 48.7, 47.3 | 48.0 ± 0.7 | [46.3, 49.7] | `results/mvpn/vcds.json` |
| MVP-N attn prefix, 6 view(s) | 81.5, 82.4, 82.6 | 82.2 ± 0.6 | [80.7, 83.7] | `results/mvpn/vcds.json` |
| MVP-N concat fixed, 1 view(s) | 48.5, 49.4, 48.1 | 48.7 ± 0.7 | [46.9, 50.4] | `results/mvpn/vcds.json` |
| MVP-N concat fixed, 6 view(s) | 82.5, 79.9, 79.1 | 80.5 ± 1.8 | [76.1, 84.9] | `results/mvpn/vcds.json` |
| MVP-N concat prefix, 1 view(s) | 49.3, 50.2, 49.7 | 49.7 ± 0.4 | [48.6, 50.8] | `results/mvpn/vcds.json` |
| MVP-N concat prefix, 6 view(s) | 71.9, 71.2, 72.4 | 71.9 ± 0.6 | [70.4, 73.3] | `results/mvpn/vcds.json` |
