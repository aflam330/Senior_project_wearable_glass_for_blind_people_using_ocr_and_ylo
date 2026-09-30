# Watermark-Based Counterfeit Detection: Print-Disjoint Evaluation and a Safety Guarantee

*Workshop draft, 2026-09-30. Same measurements as `WATERMARK_PAPER.md`, framed for a short talk.*

## Introduction

Counterfeit notes in JaalTaka share serials. A note-disjoint split therefore tests reprints. We hold out prints.

## Method

A MobileNetV2 reads the back-lit watermark window. A logistic regression, fit on validation, mixes that score with a multi-view network. The glass never says "counterfeit"; below a validation threshold it asks for a human check.

## Experiments

Serial-disjoint test, 222 notes, three seeds.

| | 1 view | 6 views |
|---|---:|---:|
| Frozen ResNet-50 | 85.1 % | 83.8 % |
| Fine-tune, quarter-decoded cache | 92.0 ± 2.3 % | 89.9 ± 4.4 % |
| Fine-tune, full decode | 94.9 ± 1.8 % | 94.4 ± 2.6 % |
| Watermark hybrid | 94.4 ± 0.5 % | 95.0 ± 0.0 % |

The jump from the second row to the third is a data bug, not a new architecture (`DATA_PIPELINE_BUG.md`, Figures 28 and 29).

## Safety

Six views: 4 of 121 genuine notes flagged, 7 of 101 counterfeits missed, on each seed. Serial blacklist: 0 of 19 unseen prints.

## Limits

About 24 effective prints. No device timing. No participant study. The hybrid ties the corrected ResNet-50.
