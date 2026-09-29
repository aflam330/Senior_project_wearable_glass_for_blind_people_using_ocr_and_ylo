---
license: mit
tags: [banknote-authentication, multi-view, bangladesh, counterfeit-detection]
datasets: [JaalTaka (local, not distributed)]
metrics: [accuracy, ece]
---

# PRMVT: prefix-robust multi-view banknote authenticator

**Checkpoint:** `realtime_bangla_taka_detection/results/qduig/prefix_ft/seed42/checkpoint.pt`. Seeds 43 and 44 are in sibling folders.

**Architecture:**
- Per view: frozen ImageNet MobileNetV3-Small (576-d) plus a small ViT (128-d).
- Heads: per-view quality, uncertainty and gate; hypernetwork-gated fusion; per-view-count batch norm; binary classifier.
- Input: 1–6 ordered close-up views, 128×128 each.

**Training:** JaalTaka, note-disjoint split 974 / 208 / 208 (seed 42). Six epochs of mixed prefix/full-view training, then three fine-tune epochs at lr 3e-4. Checkpoint chosen by mean validation accuracy over 1–6 views.

## Evaluation (JaalTaka test, n = 208 notes, re-evaluated after the NaN fix of 2026-09-28)

| Views | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Seed 42 | 0.9712 | 0.9808 | 0.9808 | 0.9808 | 0.9856 | 0.9808 |
| Mean of seeds 42–44 (%) | 96.5 | 98.4 | 98.1 | 97.6 | 98.6 | 98.1 |

Sources: `test_views_20260928/views_1_to_6.json`, `WEAK_RESULTS_FIX.md`.

## Out-of-scope use: do not use on whole-note photos

On whole-note crops from ordinary photos, this model called **20–59 % of genuine notes counterfeit** (`JAAL_VERDICT_FIXED.md`). It is valid only for JaalTaka-style close-up views. The Savior Glass app has its verdict switched off.

## Known limitations

- All data comes from one capture collection, with no camera or session IDs.
- It degrades under heavy occlusion (51.9 % at 55 %), strong low light (58.7 % at 0.2) and over-exposure (`WEAK_RESULTS_FIX.md`).
- It is not a substitute for bank-grade verification.

This card is not a Hugging Face upload.
