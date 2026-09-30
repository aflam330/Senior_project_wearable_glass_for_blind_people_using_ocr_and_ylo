# Data Pipeline Bugs and Print-Disjoint Counterfeit Detection

*Draft for a main-track submission. It is not ready to send as an A* main paper. 2026-09-30.*

## Abstract

Two findings, both measured. First, JaalTaka's usual split lets counterfeit prints appear in both train and test; on a print-disjoint split, accuracy is the 95 % of a watermark hybrid and the 94 % of a full-resolution ResNet-50, not the 98 % of the note-disjoint split. Second, a training cache built with `cv2.IMREAD_REDUCED_COLOR_4` decoded 1672×1929 views at 418×483. Retraining the ResNet-50 fine-tune on a full decode moved it from 92.0 / 89.9 % to 94.9 ± 1.8 / 94.4 ± 2.6 %. The watermark hybrid (95.0 ± 0.0 % at six views) does not significantly beat that fine-tune. A main-track claim of a new detector that beats a strong baseline is not supported.

## 1. Introduction

Assistive currency readers have to work on photographs the user actually takes, and on counterfeit prints the training set never contained. Published numbers on JaalTaka answer a weaker question: can the model recognise another photo of a print it has already seen?

## 2. Pipeline bug

Section copied in full in `DATA_PIPELINE_BUG.md`. Only the ResNet-50 fine-tune used the quarter cache. The prefix network reads a full decode and resizes to 128. The watermark crops use a half decode and a width of 700. Those two numbers stand.

## 3. Print-disjoint evaluation

222 test notes, seeds 42–44, validation used for the epoch and for the hybrid's coefficients.

The effective number of counterfeit prints is 23.6. A note-level confidence interval is the wrong width (Theorem 16).

## 4. What is not claimed

- A +5 point win over a fine-tuned ResNet-50.
- A 5,000-note synthetic counterfeit benchmark. Those images were not made. Stripping security features from genuine notes is not a dataset.
- Pi 5 latency, a user study, or glass-camera back-light.

## 5. Conclusion

The paper that the measurements support is a benchmark audit plus a corrected baseline, with a small watermark model that matches a strong ResNet and is stable. That is a workshop paper. It is not, on this evidence, a main-track architecture paper.
