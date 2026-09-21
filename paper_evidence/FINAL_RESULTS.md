# FINAL_RESULTS

Only numbers that exist as artifacts. Nothing else.

## JaalTaka note-disjoint split (MEASURED)

Source: `realtime_bangla_taka_detection/results/camva/splits/split_metadata.json`

- Unique genuine notes: **802**
- Unique counterfeit notes: **588**
- Images: **8340** (6 views × 1390 notes)
- Train / val / test notes: **974 / 208 / 208**
- Seed: **42**
- Independent unit: physical note ID (`genuine:note_*` / `counterfeit:note_*`)

## Authentication 1–6 views, same split, seed 42 (MEASURED)

Source: `realtime_bangla_taka_detection/results/camva/metrics/views_1_to_6.csv`  
n_test notes = 208. Paired bootstrap over notes.

| views | baseline acc | CAMVA acc | Δ acc | paired p |
|------:|-------------:|----------:|------:|---------:|
| 1 | 0.7356 | 0.5337 | −0.2019 | 0.000 |
| 2 | 0.8702 | 0.5721 | −0.2981 | 0.000 |
| 3 | 0.9135 | 0.6923 | −0.2212 | 0.000 |
| 4 | 0.9183 | 0.7067 | −0.2115 | 0.000 |
| 5 | 0.8990 | 0.7404 | −0.1587 | 0.000 |
| 6 | 0.9183 | 0.9663 | **+0.0481** | **0.026** |

6-view CAMVA: acc **0.9663**, F1 **0.9707**, sens **0.9667**, spec **0.9659**, ROC-AUC **0.9829**, ECE **0.0409**.  
Baseline 6-view: acc **0.9183**, F1 **0.9328**.

**Honest finding:** CAMVA is **worse** than the retrained CNN+ViT baseline with 1–5 views. It is **better** only at 6 views (+4.8 pp, 95% CI on paired Δ about [0.01, 0.09], p=0.026). Not SOTA. One seed.

## Calibration (MEASURED)

Source: `results/camva/calibration/`  
T fitted on **val** only: **T = 0.865**. Test 6-view acc unchanged **0.9663**. ECE 0.0409 → **0.0301**. Brier 0.0307 → **0.0300**.

## Ablations, 6 views (MEASURED)

Source: `metrics/ablations.json`

- A CNN+ViT averaging: acc **0.9183**
- B CAMVA encoder + mean fusion: acc **0.7885**
- C attention fusion: acc **0.9663**
- D quality-aware, no adaptive: acc **0.9663**
- E adaptive (thr 0.90, fixed order): acc **0.6779**, avg views **1.88**

Adaptive stopping **does not** keep 6-view accuracy while using fewer views.

## Adaptive thresholds (MEASURED)

Source: `metrics/adaptive.json`  
Highest adaptive acc in this run: confidence-order, t=0.98 → acc **0.8413**, avg views **2.47** (still below 6-view baseline 0.918).

## Robustness, 6 views (MEASURED)

Source: `metrics/robustness.csv`  
Clean: baseline 0.918 / CAMVA 0.966. Low-light 0.35: 0.615 / 0.721. Occlusion 20%: CAMVA drops to **0.582** (baseline 0.827).

## Emotion RAF-DB (MEASURED earlier)

`savior_glass/results/emotion_rafdb.json`: n=3068, acc **0.8654**, macro-F1 **0.795**. Not 3-seed.

## Wild Taka (recomputed from saved counts)

`paper_evidence/detection/wild_note_metrics.json`: n=450, detection **0.9844**, top-1 **0.9689**.

## OCR / COCO (prior saved files, OCR live regen crashed)

- OCR n=80 CER **0.1547** (`savior_glass/results/ocr_cer.json`)
- COCO 250-image subset mAP@0.5 **0.604** (`object_coco.json`)

## NOT MEASURED

- Second and third training seeds
- Raspberry Pi 5
- RTX 4060 (this run used **RTX 3050 Laptop**, CUDA)
- User study
- Camera/session-disjoint auth
- Qwen2-VL quality labels
- Offline network-cut full pipeline
