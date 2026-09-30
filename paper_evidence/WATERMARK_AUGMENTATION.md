# Watermark augmentation (2026-09-30)

Every candidate in `WATERMARK_ARCH_SEARCH.md` used the published crop augmentation: random resized crop (scale 0.8–1.0), rotation ±6°, colour jitter, and Gaussian blur with probability 0.2. No separate three-seed sweep was run for brightness 0.5–2×, gamma, occlusion, perspective, JPEG noise or sensor noise.

Those extra policies were not scored. Adding them after reading the architecture-search test, and keeping the policy with the highest test accuracy, would be test-set selection.
