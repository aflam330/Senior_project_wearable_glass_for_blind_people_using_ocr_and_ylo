> **Superseded 2026-09-29.** The counts below are from an earlier pass. Current counts are in `FINAL_SCAN.md` (243 Python files, 158 models, 1,025 test metrics, 211 claims).

# Project scan, 2026-09-29

Generated from `realtime_bangla_taka_detection/results/scan/scan.json` by `scripts/eval/scan_project.py`. venv/, .git/, runs/ and cache/ are skipped.

## Summary

- Python files compiled: **225**; syntax errors: **0**
- Model files (.pt/.pts/.onnx/.torchscript/.pth): **144**; failed to load: **0**
- `test_metrics.json` files: **967**
- Claims in `CLAIM_REGISTRY.json`: **136** (VERIFIED 126, NOT_MEASURED 6, NOT_MET 1, VERIFIED_SIMULATED 2, NOT_MEASURED (PROTOCOL_READY) 1); verified claims without an artifact: **0**

## Datasets

| Dataset | Images | Image folders | Content | Role |
|---|---:|---:|---|---|
| data set/Bangladeshi Counterfeit Currency Image Dataset | 1,286 | 4 | 500/1000 BDT whole-note photos: genuine 1,199, counterfeit 87 | Used: deployed jaal-verdict test (JAAL_VERDICT_FIXED.md) |
| data set/Bangladeshi_Paper_Currency_Raw | 5,073 | 9 | BanglaTaka, 9 denominations, cropped notes + label .txt | Used: compositing source for detector training |
| data set/CK+48 | 981 | 7 | 7 facial expressions | Earlier emotion training (savior_glass/scripts/train_emotion_ckplus.py) |
| data set/coco2017 | 163,957 | 3 | COCO 2017 train/val/test | Used: composite backgrounds (val2017); object-mode subset eval |
| data set/currency_yolo_data | 22,333 | 3 | Synthetic composites, YOLO labels; train 17,839 / valid 2,212 / test 2,282 | Used: detector train/val/test |
| data set/currency_yolo_data_figures_backup | 16 | 2 | 16 composites kept for figures | Figures only |
| data set/facial_expressions-master | 13,981 | 2 | Facial-expression collection (GitHub) | Earlier emotion experiments |
| data set/JaalTaka | 8,340 | 1390 | 1,390 notes x 6 views; genuine 802 / counterfeit 588 | Used: authentication, note-disjoint split 974/208/208 |
| data set/OCR bangla data set | 979,943 | 602 | Ekush handwritten letters (122 classes) + MatriVasha | Used: Ekush CNN (writer-disjoint split) |
| data set/RAF-DB | 15,339 | 14 | 7 emotions, official train/test | Used: emotion model |
| data set/yolo_bbox_annotated | 5,082 | 4 | BanglaTaka crops with tight YOLO boxes | Used: bbox experiments (results/bbox) |
| data set for comparison/A Diverse Image Dataset for Bangladeshi Currency Recognition | 5,073 | 9 | Byte-identical to BanglaTaka raw | Excluded as an external test (it is the training source) |
| data set for comparison/Bangla Money dataset Kaggle | 1,971 | 11 | 8 denominations + 1-taka (Training folders); Testing folder unlabeled | Used: independent detector test, open-set test, jaal-verdict genuine test |
| data set for comparison/Large Scale BDT DB 2026 Kaggle | 100,000 | 10 | Coins (9 classes) + demonetized notes, 10,000 images each | Used: open-set validation unknowns (OPEN_SET_REJECTION.md) |
| data set for comparison/ModelNet40 normal_resampled | 0 | 0 | 40-class point clouds (.txt), no images | Used: rendered 10-class multi-view check (VCDS_UNIVERSAL.md) |
| data set for comparison/NSTU-BDTAKA dataset Mendeley | 31,986 | 30 | Detection (boxes, class 'Taka') + Recognition (9 denominations); train/val/test | Used: detector test; Recognition/validation = open-set validation knowns |

Foreign-currency datasets: none on disk.

## Model files by folder

| Folder | Files | Kinds | Total MB |
|---|---:|---|---:|
| Unused/broken_int8/realtime_bangla_taka_detection/models | 1 | onnx 1 | 11.5 |
| Unused/broken_int8/savior_glass/models | 1 | onnx 1 | 11.5 |
| Unused/replaced_originals/realtime_bangla_taka_detection/results | 1 | baseline 1 | 6.6 |
| Unused/smart-glass/models | 1 | onnx 1 | 44.8 |
| realtime_bangla_taka_detection/models | 13 | dict 6, onnx 4, ultralytics 1, torchscript 2 | 236.7 |
| realtime_bangla_taka_detection/results/camva/checkpoints | 6 | baseline 3, camva 3 | 41.8 |
| realtime_bangla_taka_detection/results/novel/apc | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel/cvs | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel/igcr | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel/mtpt | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel/ndal | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel/ogpd | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel/sfaq | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel/sfpl | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel/ugf | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel/vcie | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel_v2/apc | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel_v2/cris | 1 | dict 1 | 47.8 |
| realtime_bangla_taka_detection/results/novel_v2/cvs | 3 | dict 3 | 140.2 |
| realtime_bangla_taka_detection/results/novel_v2/mavt | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel_v2/mtpt | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel_v2/mtpt_auth | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel_v2/mtpt_prefix | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel_v2/mtpt_prefix_ft | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel_v2/ndal | 3 | dict 3 | 140.2 |
| realtime_bangla_taka_detection/results/novel_v2/ogpd | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel_v2/pravt | 3 | dict 3 | 140.2 |
| realtime_bangla_taka_detection/results/novel_v2/savs | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel_v2/sfpl | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel_v2/sfpl_fullviews | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel_v2/sfpl_localsteps | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel_v2/ugf | 2 | dict 2 | 93.5 |
| realtime_bangla_taka_detection/results/novel_v2/vat | 3 | dict 3 | 140.3 |
| realtime_bangla_taka_detection/results/novel_v2/vcie | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel_v2/vcie_k1 | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/novel_v2/vcie_long | 1 | dict 1 | 46.7 |
| realtime_bangla_taka_detection/results/qduig/ablation_prefix | 20 | qduig 20 | 332.4 |
| realtime_bangla_taka_detection/results/qduig/ablations | 10 | qduig 10 | 166.2 |
| realtime_bangla_taka_detection/results/qduig/aux_small_ft | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/auxfix_curriculum | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/auxfix_kendall | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/auxfix_pcgrad | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/auxfix_regularize | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/auxfix_scale_0p0001 | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/auxfix_scale_0p001 | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/auxfix_scale_0p01 | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/auxfix_scale_0p1 | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/auxfix_scale_divcost | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/auxfix_separate | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/auxfix_separate_frozen | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/auxfix_stopgrad | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/cost_norm_ft | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/fusion_meanpool_aux | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/fusion_meanpool_aux_ft | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/fusion_rsqa_noaux | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/fusion_rsqa_noaux_ft | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/occlusion_ft | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/occlusion_matched | 1 | dict 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/occlusion_robust | 1 | dict 1 | 16.6 |
| realtime_bangla_taka_detection/results/qduig/occlusion_robust_e15 | 3 | dict 3 | 49.9 |
| realtime_bangla_taka_detection/results/qduig/prefix | 4 | qduig 4 | 66.5 |
| realtime_bangla_taka_detection/results/qduig/prefix_ft | 3 | qduig 3 | 49.9 |
| realtime_bangla_taka_detection/results/qduig/proposed | 2 | qduig 2 | 29.1 |
| realtime_bangla_taka_detection/results/qduig/robust_ft | 1 | qduig 1 | 16.6 |
| realtime_bangla_taka_detection/results/robustness/occlusion_inpaint | 1 | dict 1 | 0.2 |
| realtime_bangla_taka_detection/results/same_arch/fixed | 2 | qduig 2 | 33.2 |
| realtime_bangla_taka_detection/results/same_arch/fixed_shbn | 2 | qduig 2 | 33.2 |
| realtime_bangla_taka_detection/results/theory/pacbayes_d1_model | 1 | qduig 1 | 16.6 |
| savior_glass/models | 10 | onnx 2, ultralytics 3, dict 5 | 627.5 |
| savior_glass/results/yolo_runs/taka_wild_ft | 2 | ultralytics 2 | 45.0 |

Each file was loaded: PyTorch files with `torch.load(map_location='cpu')`, TorchScript with `torch.jit.load`, ONNX with an ONNX Runtime CPU session.

## Scripts, results and claims

Script purposes, result tables and the claim list are in `docs/reports/Project_Report_2026-09-29.md` (Parts B, D, I) and in `CLAIM_REGISTRY.json`. The `test_metrics.json` files are per-run, per-view-count outputs under `results/`; the headline ones are traced claim by claim in the registry, which validates with `scripts/validate_claims.py`.
