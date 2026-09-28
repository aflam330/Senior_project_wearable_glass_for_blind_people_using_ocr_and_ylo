# MTPT fix A: PRMVT 6+3 schedule

v2 stays in its own directory. This run resumes it, trains 6 epochs at learning rate 0.001, then 3 epochs at 0.0003. Mixed prefix dropout is the trainer's existing view mask. The checkpoint is the best validation 1-view accuracy, including the loaded v2 weights. Test labels were not used.

Loaded v2 validation 1-view: 0.9471153846153846. The 6-epoch stage reached validation 1-view 0.9759615384615384 at epoch 4. The 3-epoch fine-tune did not raise that number. Source: `results/novel_v2/mtpt_prefix/seed42/val_metrics.json` and `results/novel_v2/mtpt_prefix_ft/seed42/val_metrics.json`.

Test n=208. Source: `results/novel_v2/mtpt_prefix_ft/seed42/test/test_metrics.json`.

| views | accuracy |
| ---: | ---: |
| 1 | 0.9807692307692307 |
| 2 | 0.9471153846153846 |
| 3 | 0.9711538461538461 |
| 4 | 0.9519230769230769 |
| 5 | 0.9615384615384616 |
| 6 | 0.9663461538461539 |

1-view is above v2 (0.9519230769230769) and above PRMVT (0.9711538461538461). Target reached. Task-weight search, Kendall weighting, and stop-gradient auxiliary heads were not trained.
