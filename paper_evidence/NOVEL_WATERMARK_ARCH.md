# Novel watermark network in the search (2026-09-30)

One new network was trained from scratch inside the validation-ranked search: a stride-4 convolution stem to a 14×14 grid, then two transformer encoder layers (width 64, 4 heads) and a linear head. 114,210 parameters. Seeds 42, 43, 44. Epoch by validation AUC.

Mean validation AUC 0.9527 ± 0.0124. Test accuracy 86.3 ± 0.9%, AUC 0.931. It is last of the ten candidates.

FiLM denomination conditioning, a mixture of experts, a differentiable backlight model and Monte Carlo dropout were not trained.
