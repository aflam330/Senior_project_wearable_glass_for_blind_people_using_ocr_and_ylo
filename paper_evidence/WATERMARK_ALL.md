# Watermark integration, every form requested (2026-09-30)

| # | Form | What was done | Result | Source |
|---|---|---|---|---|
| 1 | Watermark as part of the network, trained end-to-end | Attention fusion over [view tokens, watermark-window token], trained on TRAIN with random view counts and watermark dropout; frozen ImageNet ResNet-50 features for both | about 90.5–91.0 % on unseen prints; the same model without the watermark token 86.9–88.7 % | `BEAT_RESNET50_FINAL.md`, `results/serial_split/beat_resnet.json` |
| 2 | Separate watermark classifier | MobileNetV3-Small and MobileNetV2 fine-tuned on window crops; V2 chosen on VAL AUC | V2: 92.9 %, AUC 0.976; V3: 91.9 %, AUC 0.962 | `results/watermark/mobilenetv2.json`, `mobilenet.json` |
| 3 | Watermark + network (+ serial) ensemble | (a) validation-fitted combination with the network; (b) uncertainty-weighted average of network, fusion and watermark; serial added and found redundant | (a) **94.4 ± 0.5 / 95.0 ± 0.0 %** (best); (b) 92.8–93.7 % | `hybrid_final_seeds.json`, `beat_resnet.json`, `hybrid.json` |
| 4 | INT8 watermark model | V3: dynamic, static QDQ, Conv-only (all failed: 45–63 % agreement); V2: static QDQ | **V2 INT8: 100 % the same decisions as FP32, 2.6 MB** | `mobilenetv2.json`, `Unused/README.md` |
| 5 | Watermark on the Pi (< 10 MB) | V2 INT8, 2.6 MB | Pi latency READY_FOR_DEVICE | `PI5_READY_FINAL.md` |
| 6 | Glass app "hold up to the light" mode | `savior_glass/modes/watermark_check.py`, behind `WATERMARK_CHECK_ENABLED` (off) | App path on unseen-print test photos: 92.9 %, 3 / 98 genuine "not clear" | `results/watermark/app_module_check_v2int8.json` |
| 7 | Validation on synthetic back-lit notes | **Not possible with the data on disk.** The reconstructed whole notes are built from front-lit views 1–4, so they contain no back-lit watermark. Generating a back-lit watermark would mean drawing one, which would test the drawing, not the detector | Scoped out; the real test is glass-camera photos held against light | — |

**Summary.**
- The watermark is the most useful single addition for counterfeit prints never seen in training.
- The strongest measured form is a separate small classifier, combined with the multi-view network by a validation-fitted logistic regression: 95.0 % at six views on unseen prints, against 89.3 % for the network alone.
- It runs as a 2.6 MB INT8 model.
