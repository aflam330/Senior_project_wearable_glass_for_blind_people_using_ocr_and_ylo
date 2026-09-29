# View-count shift on MVP-N (real multi-view photos, 2026-09-29)

MVP-N (NeurIPS 2022 Datasets and Benchmarks): 44 object classes; official valid and test lists of 4,400 sets each, 2-6 real views per set. Data downloaded to `data set for comparison/MVP-N-main/data`. Scripts: `realtime_bangla_taka_detection/scripts/eval/mvpn_vcds.py`, `mvpn_reference.py`. Raw: `results/mvpn/vcds.json`, `reference.json`.

Frozen ImageNet ResNet-50 features. Two joint-fusion heads, trained with fixed 6-view samples or random 1-6 view prefixes, 3 seeds each; epoch chosen on VALID (mean over k). Test at k uses the first k views of every test set with at least k views.

## Test accuracy, % (mean ± sd over seeds 42, 43, 44)

| model | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---:|---:|---:|---:|---:|---:|
| concat head, fixed 6-view | 48.7 ± 0.7 | 59.2 ± 0.4 | 62.9 ± 0.5 | 66.4 ± 0.5 | 70.9 ± 0.6 | 80.5 ± 1.8 |
| concat head, prefix | 49.7 ± 0.4 | 58.9 ± 0.1 | 61.7 ± 0.6 | 64.2 ± 0.4 | 67.3 ± 0.3 | 71.9 ± 0.6 |
| attention head, fixed 6-view | 40.7 ± 1.0 | 50.2 ± 0.1 | 53.0 ± 0.6 | 56.9 ± 0.4 | 61.9 ± 0.5 | 75.8 ± 0.3 |
| attention head, prefix | 48.0 ± 0.7 | 55.4 ± 0.7 | 58.4 ± 1.0 | 62.2 ± 1.1 | 67.4 ± 0.7 | 82.2 ± 0.6 |
| per-view logistic regression, probabilities averaged (reference) | 55.3 | 65.8 | 69.8 | 74.1 | 78.6 | 84.7 |

## Reading

- **Information in one view.** The per-view reference reaches 55.3 % at one view. MVP-N deliberately includes uninformative views, so most of the drop from six views to one is information loss (Theorem 1), which no training can remove.
- **Attention head: VCDS present, and prefix training removes much of it.** Fixed training gives 40.7 % at one view, 14.7 points below the reference. Prefix training gives 48.0 % (+7.3), and it is also better at six views (82.2 % vs 75.8 %).
- **Concat head: prefix training does not help.** One view goes 48.7 → 49.7 %, and six views drop 80.5 → 71.9 %. With zero-padded slots, the head must learn one mapping per view count, and on 440 training objects the mixture costs the full-view case more than it gains.
- **Per-view scoring is the strongest option here too:** 84.7 % at six views, above every joint head.

**Consequence for the paper.** VCDS and its prefix fix transfer to a second, non-currency, real-photo dataset for the attention-fusion head, but not universally. The claim must name the architecture. JaalTaka (PRMVT's attention fusion) and MVP-N (attention head) agree; the concat head is a counterexample.
