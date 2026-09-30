# Watermark preprocessing (2026-09-30)

One preprocessing change was inside the validation-ranked search: CLAHE (clip limit 2, 8×8 tiles) on the watermark crop, then the same MobileNetV2 fine-tune as the other candidates. Eight epochs, seeds 42, 43, 44, epoch by validation AUC.

Test accuracy 91.2 ± 0.6%, AUC 0.972. Validation AUC 0.9961 ± 0.0024, second in the search, behind EfficientNet-B0 without CLAHE. It does not beat the published MobileNetV2 at 92.9%.

Fourier bandpass, gamma, Retinex, Gaussian and Laplacian pyramids, wavelets, non-local means, BM3D and DnCNN were not trained as extra models. A second sweep, kept only if the test rose, would choose on the test set that this search has already read.
