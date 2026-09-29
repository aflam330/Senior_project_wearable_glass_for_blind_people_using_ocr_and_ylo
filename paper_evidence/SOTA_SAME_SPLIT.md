# Same-split comparison: wins and losses (2026-09-30)

Full tables: `SOTA_COMPARISON.md` (standard split, 60 paired tests), `SERIAL_FIX.md` and `SERIAL_WATERMARK_DETECTOR.md` (serial-disjoint split).

| Comparison | Standard split | Unseen counterfeit prints | Verdict |
|---|---|---|---|
| Prefix network vs frozen ResNet-50 probe (the strongest of the five published backbones) | 98.2 vs 98.1 % (1 view); 98.7 vs 99.0 % (6) | **90.1 vs 85.1 %** (1 view); **87.8 vs 83.8 %** (6) | tie on the standard split; **network ahead on new prints** |
| Prefix network + watermark vs probe | — | **95.5 vs 83.8 %** (6 views) | **win** |
| Prefix network + watermark vs network alone | 99.0 vs 98.1 % (6) | 95.5 vs 87.8 % (6), McNemar p = 0.0002 | **significant win on new prints** |
| Prefix network vs CNN+ViT baseline (3 seeds) | 98.2 vs 76.1 % (1 view) | not run | win |
| Prefix network vs CAMVA (3 seeds) | 98.2 vs 54.0 % (1 view) | not run | win |

**Published numbers on other data** (NoteShieldBD 99 %, denomination 98–100 %) are not scored on JaalTaka and are not compared (`SOTA_BEAT.md`).

**What can be claimed:**
1. **On counterfeit prints not seen in training**, the watermark-aware hybrid is the best measured JaalTaka authenticator, at 95.5 %.
2. **On the standard split**, no method is significantly better than the frozen probe.
