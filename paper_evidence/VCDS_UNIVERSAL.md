# View-count drop

Definition. For a model scored at forced view counts k = 1 and k = 6 on the same notes, the view-count drop is Acc(6) - Acc(1). A positive drop means the 6-view score is higher. This is an empirical gap on JaalTaka, seed 42, n = 208. It is not a theorem that every multi-view dataset has the same gap.

## JaalTaka, measured

| model | Acc(1) | Acc(6) | Acc(6) - Acc(1) | source |
| --- | ---: | ---: | ---: | --- |
| CNN+ViT baseline | 0.7355769230769231 | 0.9182692307692307 | 0.1826923076923076 | prefix view table in `FINAL_RESULTS.md` |
| PRMVT prefix fine-tune | 0.9711538461538461 | 0.9759615384615384 | 0.0048076923076923 | `results/qduig/prefix_ft/seed42/test_views/views_1_to_6.json` |
| VCIE, 1-view checkpoint | 0.9278846153846154 | 0.9278846153846154 | 0.0 | `results/novel_v2/vcie_k1/seed42/test/test_metrics.json` |
| MTPT, 6+3 schedule | 0.9807692307692307 | 0.9663461538461539 | -0.0144230769230768 | `results/novel_v2/mtpt_prefix_ft/seed42/test/test_metrics.json` |

The baseline loses 0.1827 when only the first view is kept. PRMVT's drop is one note. MTPT's 1-view score is higher than its 6-view score, so the drop is negative.

## Other datasets

ModelNet40, MOSI, ADNI, and a COCO multi-view set were not downloaded and were not trained. Their view-count drops are NOT_MEASURED. No universal claim across those datasets is made.
