# Watermark fine-tuning schedules (2026-09-30)

The architecture search used one schedule for every ImageNet model: full fine-tune, AdamW at 3e-4, weight decay 1e-4, eight epochs, checkpoint = best validation AUC. The CNN-transformer, trained from scratch, used AdamW at 1e-3.

Gradual unfreezing, discriminative learning rates, mixout, stochastic weight averaging, exponential moving average and sharpness-aware minimisation were not run. They were not scored, because choosing among them with the test set already read would be test-set selection.
