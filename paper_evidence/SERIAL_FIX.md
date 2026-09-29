# Serial shortcut: evaluation on unseen prints (2026-09-30)

**Problem.** JaalTaka counterfeits share a few printed serials, so the note-disjoint split is not print-disjoint. A serial lookup scores 90.2 % on its test set (`JAALTAKA_SERIAL_AUDIT.md`).

**Fix.** A serial-disjoint split: no test counterfeit shares a print with a training counterfeit (`scripts/eval/make_serial_split.py`, `results/serial_split/`).
- The two largest prints (3274658 and 184383x) are in TRAIN. The other counterfeit prints go whole to VAL or TEST.
- TEST: 222 notes, 101 counterfeit.

| Model | Standard split, 1 / 6 views | Serial-disjoint, 1 / 6 views | Source |
|---|---|---|---|
| Prefix network, shared BN (seed 42) | 97.6 / 98.6 % | **90.1 / 87.8 %** | `SAME_ARCH_RESULTS.md`; `results/serial_split/runs/prefix_shbn/s2/test_views/` |
| ResNet-50 probe | 98.1 / 99.0 % | 85.1 / 83.8 % | `SOTA_COMPARISON.md`; `results/watermark/serial_split_eval.json` |
| Prefix network + watermark window | — | 93.2 / **95.5 %** | `results/watermark/hybrid_serial_split.json` |
| Serial blacklist | 89.9 % | catches 0 counterfeits (by construction) | `results/watermark/hybrid.json` |

On the serial-disjoint split, both genuine-rejection rates are 0.8 %. The drop is almost entirely missed counterfeits.

**Cross-validation over every note** (ResNet-50 probe, `results/sota/cv_probe.json`):
- note folds: 98.3 % at 1 view, 98.9 % at 6;
- serial-grouped folds: 95.8 % and 98.9 %.

**Conclusions.**
1. Standard JaalTaka accuracies (about 98–99 %) overstate performance on new counterfeit prints. The honest one-network figure is about 88–90 %.
2. The trained prefix network generalises better than the frozen probe (+5 points at one view).
3. The watermark window restores most of the gap: 95.5 % at six views.
4. The serial split is one split, trained with three seeds (see below). Its test has 101 counterfeits from about 20 small prints, so it gives an estimate, not a precise number.

**Three seeds (2026-09-30).** Prefix network on unseen prints: 89.9 ± 0.7 % (1 view), 89.3 ± 1.3 % (6 views). With the watermark: 93.8 ± 1.0 % and 94.7 ± 0.9 % (`results/watermark/hybrid_serial_split_seeds.json`).
