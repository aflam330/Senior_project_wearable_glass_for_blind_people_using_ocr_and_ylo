# An Offline Assistive Glass for Bangladeshi Taka, with a Print-Disjoint Check

*Regional IEEE application draft, 2026-09-30. System paper. Authors to be added.*

## Summary

The glass names the denomination offline and refuses to accuse a note of being counterfeit. On Bangla Money photographs it has never trained on, denomination accuracy is 91.5 %. On NSTU hand-held close-ups it is 18.5 %, because fingers and folds hide the note. The counterfeit policy says "likely genuine" only above the highest counterfeit score in validation, and otherwise "check by hand". That policy passed 0 of 88 in-domain counterfeits.

A separate study on JaalTaka shows that the usual accuracy near 98 % shares counterfeit prints between train and test. On unseen prints a full-resolution ResNet-50 reaches 94.4 ± 2.6 % at six views and a watermark hybrid reaches 95.0 ± 0.0 %. The watermark model is a 2.6 MB INT8 network. Its speed on a Raspberry Pi 5 has not been measured; the benchmark script is ready (`PI5_FINAL.md`).

## What the talk should not say

It should not say the watermark beats a fine-tuned ResNet-50. It should not quote a user study. It should not treat augmented copies of four physical counterfeit notes as hundreds of counterfeits.
