# OGPD status

Seed 42. Test split, n=208. v1 is the result that stands. v2 is a negative result and was not deleted.

## v1

`results/novel/ogpd/seed42/test/test_metrics.json`

| views | accuracy |
|---:|---:|
| 1 | 0.9230769230769231 |
| 2 | 0.9230769230769231 |
| 3 | 0.9278846153846154 |
| 4 | 0.9134615384615384 |
| 5 | 0.9086538461538461 |
| 6 | 0.9182692307692307 |

The checkpoint is `results/novel/ogpd/seed42/checkpoint.pt`.

## v2

`results/novel_v2/ogpd/seed42/test/test_metrics.json`

| views | accuracy |
|---:|---:|
| 1 | 0.7596153846153846 |
| 2 | 0.8413461538461539 |
| 3 | 0.8125 |
| 4 | 0.8125 |
| 5 | 0.7980769230769231 |
| 6 | 0.7932692307692307 |

v2 is below v1 at every view count. The 1-view drop is 0.16346153846153855. v2 used a smaller distillation weight than v1. That change made the student worse, so it is not the OGPD number.

Verdict: PASS for keeping v1. v2 remains a negative result.
