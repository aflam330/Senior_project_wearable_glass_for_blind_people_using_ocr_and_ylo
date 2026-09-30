# Winner-seed average (2026-09-30)

The validation winner is EfficientNet-B0. Its three seed probabilities were averaged with equal weight. Threshold 0.5, fixed in the script before this average was treated as a result.

On the 197 registered test notes: accuracy 92.4%, AUC 0.962, 1 of 98 genuine notes flagged, 14 of 99 counterfeits missed.

That is below the published MobileNetV2 (92.9%, AUC 0.976) and below the earlier four-checkpoint MobileNet ensemble at its validation threshold (93.4%, McNemar p = 1.0). Learned ensemble weights, bagging, boosting, snapshot ensembles and test-time augmentation were not fit. The three-seed mean is the only new ensemble number.
