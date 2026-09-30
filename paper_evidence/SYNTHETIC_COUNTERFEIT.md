# Synthetic counterfeits (2026-09-30)

**Not generated.**

Stripping watermarks, copying serial numbers, and erasing threads or microprint from genuine Bangladeshi notes produces counterfeit-note images. That is not a measurement, and it is not something this project will do. The other listed datasets (NSTU-BDTAKA, BanglaTaka, Bangla Money, Large Scale BDT DB 2026) have no genuine-versus-counterfeit labels (`CROSS_DATASET_TAKA.md`). Painting security features off genuine photos and calling the result a 5,000-note counterfeit test would invent both the images and the labels.

Augmenting the existing 87 whole-note counterfeit images up to 500 by rotation, blur, or JPEG does not create new prints. Those 87 images come from about four physical notes (`LARGE_SCALE_DATASET.md`). A print-disjoint split of augmented copies of the same four notes still has about four prints. Reporting 500 samples as 500 counterfeits would overstate the sample size. Theorem 16 in `THEORY_FINAL.md` is the reason: the effective sample size is the number of prints.

## What is already measured, and is the substitute

| Evaluation | What it is | Result |
|---|---|---|
| Serial-disjoint JaalTaka test | 222 notes, 101 counterfeits from unseen prints | Watermark hybrid 95.0 ± 0.0 % at 6 views; full-resolution fine-tune 94.4 ± 2.6 % |
| Serial-grouped 5-fold over all 1,390 notes | Every note tested once, prints not split across train and test | Frozen probe 95.8 % at 1 view, 98.9 % at 6 views (`EXPANDED_TEST.md`) |
| Whole-note counterfeit photos | 87 images, about 4 physical notes | Policy E passes 0 of the held-out whole-note counterfeits it was scored on; this is not a 500-print test |

A cross-domain number for a detector trained on drawn counterfeits and tested on JaalTaka is not in any artifact, because that training set was not built.
