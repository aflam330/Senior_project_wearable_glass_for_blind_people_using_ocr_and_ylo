# Open-source release: prepared, not published (2026-09-30)

Nothing has been pushed to GitHub or uploaded to Hugging Face. Publishing is the authors' decision. The work is on the local branch `research-final-2026-09-30`.

## Repository layout

```
savior_glass/                      offline assistive glass app (Pi 5); modes/ incl. currency, watermark_check
realtime_bangla_taka_detection/
  roboeye/                         models and data code
  scripts/train, scripts/eval      every experiment in paper_evidence has its script here
  models/                          detector (PT / ONNX / INT8), watermark MobileNetV2 (FP32 / INT8)
  results/                         saved metrics and predictions behind every number
benchmark/jaaltaka_seq/            benchmark package (evaluate.py, baselines)
paper_evidence/                    results pages, claim registry, figures, tables, papers, model cards
docs/                              inventory, overview, literature
```

## Release items

| Item | State | Path |
|---|---|---|
| License (MIT), citation, contributing | present | `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md` |
| Model cards (Hugging Face front matter) | detector, PRMVT, MTPT, VCIE, occlusion, **watermark MobileNetV2 (new)** | `paper_evidence/model_cards/` |
| JaalTaka-Seq splits | note-disjoint and **print-disjoint** | `results/camva/splits/`, `results/serial_split/` |
| Unified Bangladeshi Taka manifest (141,372 units) | present | `results/unified_bdt/manifest.json` |
| Leaderboard | present | `paper_evidence/BENCHMARK_FINAL.md`, `DANDB_PAPER.md` |
| Pi 5 Docker file | present, **not built** | `savior_glass/docker/Dockerfile` |
| Claim registry with value checks | 246+ claims | `paper_evidence/CLAIM_REGISTRY.json`, `scripts/validate_claims.py` |

## Before publishing

1. **Cite JaalTaka** (CC BY 4.0, DOI 10.17632/2m7wk5cy4c.2) and do not re-host its photos; release only IDs, splits and derived metadata. Window crops are derived images, so check the licence terms before sharing them.
2. **Check the other datasets' licences** (NSTU, Bangla Money, Large Scale BDT, RAF-DB, COCO) before sharing any derived weights.
3. **Git LFS for weights:** git-lfs is installed; the current commits store `.pt` files directly, so decide before pushing to GitHub.
4. **Add author names** to `CITATION.cff`.
