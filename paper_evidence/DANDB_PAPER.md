# JaalTaka-Seq: A Sequential Multi-View Counterfeit Benchmark with Print-Disjoint Evaluation

*Datasets & Benchmarks track draft, 2026-09-30. Numbers from saved files. Authors to be added.*

## Abstract

JaalTaka-Seq is a benchmark layer on top of the public JaalTaka dataset (1,390 Bangladeshi 500 and 1,000 Taka notes, 802 genuine and 588 counterfeit, six photos per note, CC BY 4.0). It adds no new photographs. It adds:
1. an ordered six-view protocol for evaluating counterfeit detection at 1–6 views;
2. a **print-disjoint split**, because JaalTaka's counterfeits share a few printed serials: 279 of 322 readable counterfeit 500s carry one serial;
3. a whole-note counterfeit test from a second public dataset;
4. a leaderboard of 18 methods over three seeds plus five frozen backbones;
5. evaluation code, split files and model cards.

The print-disjoint split changes conclusions: accuracies near 98 % on the standard split become 84–90 % on unseen prints, and a watermark-aware method reaches 95.0 %.

## 1. Dataset

| Component | Content | License / source |
|---|---|---|
| JaalTaka photos | 1,390 notes × 6 region photos (views 1–4 front close-ups, 5 back, 6 back-lit) | CC BY 4.0, Mendeley DOI 10.17632/2m7wk5cy4c.2 |
| Whole-note counterfeit test | 1,286 photos of 500 / 1,000 BDT (87 counterfeit, about 4 physical notes; bursts and augmented copies grouped) | public dataset, test only |
| Added metadata | Note IDs, view order, OCR serial per note, print groups, denomination (from template registration), watermark-window crops (derived) | this release |

**Datasheet points.**
- Counterfeits were supplied to the dataset authors by the Rapid Action Battalion; genuine notes by three banks.
- No personal data.
- Pen marks and wear are present on genuine notes.
- The unified manifest (`results/unified_bdt/manifest.json`) also indexes denomination, detection and open-set data from four other Bangladeshi datasets (141,372 units).

## 2. Splits

| Split | Rule | Train / val / test | Test counterfeits | Files |
|---|---|---|---:|---|
| Note-disjoint (standard) | Random physical notes, seed 42 | 974 / 208 / 208 | 88 (68 share a serial print with training) | `results/camva/splits/` |
| **Print-disjoint** | Counterfeits grouped by OCR serial (6 digits); two largest prints in train; other prints whole to val / test | 946 / 222 / 222 | 101 (0 shared prints; effective number of prints 23.6) | `results/serial_split/` |
| 5-fold CV | Note folds or serial-grouped folds over all 1,390 notes | — | every note once | `scripts/eval/cv_probe.py` |

## 3. Evaluation protocol

- **Metric.** Accuracy at the first k views (k = 1..6), plus false-counterfeit rate on genuine notes and counterfeit miss rate.
- **Selection.** Checkpoints, thresholds and combiners are chosen on validation only.
- **Seeds.** Report three training seeds (mean ± sd) and paired exact McNemar tests on the same test notes.
- **Reporting rule.** Report the print-disjoint result alongside any standard-split result. Its uncertainty is governed by the number of prints, not notes (Theorem 16: ±0.28 at 95 % with 24 effective prints).

## 4. Baselines and leaderboard

**Standard split** (`BENCHMARK_FINAL.md`, `SOTA_COMPARISON.md`; 3 seeds unless marked):

| Method | 1 view | 6 views |
|---|---:|---:|
| Prefix network, shared BN | 98.2 ± 0.7 % | 98.7 ± 0.3 % |
| ResNet-50 probe (deterministic) | 98.1 % | 99.0 % |
| DenseNet-121 probe (deterministic) | 97.6 % | 99.0 % |
| VAT | 97.3 ± 0.7 % | 96.3 ± 0.3 % |
| NDAL | 97.0 ± 1.0 % | 95.8 ± 1.4 % |
| APC | 97.0 ± 1.0 % | 96.2 ± 0.5 % |
| MTPT | 96.8 ± 1.2 % | 96.8 ± 0.3 % |
| PRMVT | 96.5 ± 1.5 % | 98.1 ± 1.0 % |
| PRAVT | 96.2 ± 1.3 % | 96.5 ± 1.5 % |
| CVS | 95.4 ± 1.9 % | 96.8 ± 0.6 % |
| CRIS | 94.9 ± 0.6 % | 93.6 ± 1.5 % |
| MAVT / SAVS | 93.6 % | 94.9 / 92.9 % |
| VCIE | 93.4 ± 1.5 % | 93.3 ± 0.5 % |
| Fixed 6-view, per-count BN | 87.2 ± 9.7 % | 99.0 ± 0.0 % |
| CNN+ViT | 76.1 ± 3.6 % | 92.3 ± 0.8 % |
| Fixed 6-view, shared BN | 71.2 ± 11.4 % | 98.1 ± 0.5 % |
| CAMVA | 54.0 ± 3.4 % | 97.1 ± 0.8 % |

**Print-disjoint split** (`table12_unseen_prints.md`, `BEAT_RESNET50_FINAL.md`):

| Method | 1 view | 6 views |
|---|---:|---:|
| Serial blacklist | 0 % recall on counterfeits | — |
| ResNet-50 probe | 85.1 % | 83.8 % |
| ResNet-50 fine-tuned | 92.0 ± 2.3 % | 89.9 ± 4.4 % |
| Prefix network | 89.9 ± 0.7 % | 89.3 ± 1.3 % |
| Attention fusion of view and watermark features | 90.8 ± 0.3 % | 90.5 ± 0.0 % |
| Ensemble (network + fusion + watermark) | 93.4 ± 0.3 % | 93.1 ± 0.3 % |
| **Prefix network + watermark window** | **94.4 ± 0.5 %** | **95.0 ± 0.0 %** |

**Whole-note counterfeit test** (`JAAL_VERDICT_FIXED.md`):
- Best counterfeit-ranking AUC: 0.966 (ResNet-50 probe on cut views).
- The safety policy passes 0 / 25 counterfeit photos.

## 5. Open-source release (prepared; publishing is the authors' decision)

| Item | Path |
|---|---|
| Split files | `results/serial_split/*.json`, `results/camva/splits/*.json` |
| Evaluation | `benchmark/jaaltaka_seq/evaluate.py`, `realtime_bangla_taka_detection/scripts/eval/*.py` |
| Models and cards | `realtime_bangla_taka_detection/models/`, `paper_evidence/model_cards/` |
| Leaderboard | `paper_evidence/BENCHMARK_FINAL.md` |
| Citation | `CITATION.cff` (cite JaalTaka's DOI as well) |

## 6. Ethics and limitations

- **Dual use.** A counterfeit detector's failure analysis could guide counterfeiters: which features a model ignores. We release window crops and scores, not a guide to defeating the watermark check.
- **Harm to users.** A false "counterfeit" harms users, so the reference deployment never asserts it.
- **Limits:**
  - About 24 effective counterfeit prints in the print-disjoint test.
  - Serial OCR can mis-group notes.
  - Genuine notes come from three banks only.
  - The watermark requires back-lit capture.
