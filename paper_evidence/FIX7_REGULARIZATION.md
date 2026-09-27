# Fix 7: auxiliary loss on every fourth batch

Seed 42. Two epochs from the saved full PRMVT checkpoint. Auxiliary weights are applied when the batch index modulo 4 is 0, and set to 0 on the other batches.

Test accuracy from `results/qduig/auxfix_regularize/seed42/test_views/views_1_to_6.json`:

| views | accuracy |
|---:|---:|
| 1 | 0.9471153846153846 |
| 2 | 0.9375 |
| 3 | 0.9423076923076923 |
| 4 | 0.9471153846153846 |
| 5 | 0.9663461538461539 |
| 6 | 0.9711538461538461 |

1-view is below 0.9711538461538461. Verdict: FAIL.

The other two schedules in the hypothesis (probability 0.25, and auxiliary loss only on epochs 4–6) were not trained. The curriculum run covers a delayed schedule and is reported in `FIX5_CURRICULUM.md`.
