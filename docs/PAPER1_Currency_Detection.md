# Paper 1 Documentation — Real-Time Bangladeshi Taka Banknote Detection

**Primary source:** `e:\Final SP\realtime_bangla_taka_detection`  
**Existing report:** `Bangla_Currency_Detection_Report.pdf` (July 2026, 16 pages)  
**GitHub:** https://github.com/Mukut313/realtime_bangla_taka_detection

This document consolidates everything needed to write a publication-ready currency paper.

---

## 1. Working title options

1. Real-Time Bangladeshi Taka Banknote Detection via YOLOv8 Trained on Synthetically Composited Real-World Backgrounds  
2. Bridging the Synthetic-to-Real Gap for BDT Currency Detection Using Copy-Paste Domain Randomization  
3. Offline Real-Time Detection and Voice Announcement of Bangladeshi Banknotes

**Suggested subtitle:** A YOLOv8 Object-Detection System Trained on Synthetically Composited Data with Real-World Backgrounds

---

## 2. Abstract (from project report — editable)

This project develops a real-time detection system that localises and classifies the nine denominations of Bangladeshi Taka (BDT) paper currency — 2, 5, 10, 20, 50, 100, 200, 500 and 1000 Taka — from a live camera feed. A recurring problem with publicly available BDT datasets is that they consist exclusively of clean, close-up, full-frame note scans, which teach a model that “the entire frame is the note” and fail on a webcam, where a note appears at an arbitrary scale, angle and position against cluttered real backgrounds. To bridge this synthetic-to-real domain gap, we designed a copy-paste compositing pipeline that pastes 5,073 background-free note images onto 1,500 real-world photographs from the COCO dataset, applying perspective warp, rotation, scale variation, colour/brightness jitter, motion blur, sensor noise and JPEG re-compression, and automatically generating exact bounding-box labels. The resulting 22,315-image dataset was used to fine-tune a COCO-pretrained YOLOv8s detector for 80 epochs. On a held-out test set of 2,238 images the model achieves mAP@0.5 of 0.995, mAP@0.5:0.95 of 0.847, precision of 0.998 and recall of 0.997, at 7.3 ms per image (≈135 FPS) on an NVIDIA RTX 4060 Laptop GPU. The system runs live from a standard webcam and includes automatic low-light brightness correction and push-to-speak voice feedback.

**Keywords:** Object Detection, YOLOv8, Bangladeshi Currency, Synthetic Data, Domain Randomization, Copy-Paste Augmentation, Transfer Learning, Real-Time Inference, Accessibility

---

## 3. Problem statement & motivation

| Aspect | Content for paper |
|--------|-------------------|
| Practical need | Assistive tech for visually impaired users; retail/cash automation; mobile/webcam use |
| Technical gap | Prior BDT work is mostly **classification** on clean, centred notes |
| Failure mode | Full-frame models fail when notes are small, rotated, partially lit, in clutter |
| Reformulation | Treat task as **object detection** (localise + classify) |
| Data strategy | Synthetic compositing + domain randomisation to close sim-to-real gap |
| Accessibility | Push-to-speak offline TTS so users hear the denomination |

**Key insight from development history:** An early pipeline that labelled entire frames as the note (`scripts/prepare_dataset.py`, box `0.5 0.5 1.0 1.0`) scored well on synthetic tests but failed live. That failure motivated the compositing redesign.

---

## 4. Contributions (claim carefully)

1. A **reproducible copy-paste compositing pipeline** that converts a classification dataset (BanglaTaka) into a large detection dataset with **pixel-perfect automatic boxes** (zero manual annotation).  
2. **Domain randomisation** targeting webcam artefacts: perspective, blur, noise, JPEG, COCO clutter, 10% negatives.  
3. A **YOLOv8s** fine-tuned detector with strong held-out synthetic test metrics and real-time laptop-GPU speed.  
4. A **deployable webcam application** with low-light correction, push-to-speak TTS, and TorchScript export.  
5. Positioning: gains attributed primarily to **data engineering**, not a custom detection head.

**Honesty caveat for reviewers:** Test metrics are on held-out *synthetic* composites. Real-world claims should be framed as qualitative webcam evidence unless you add a labelled wild set.

---

## 5. Related work

**Full annotated list (your spreadsheet):** see [`RELATED_WORK_BIBLIOGRAPHY.md`](RELATED_WORK_BIBLIOGRAPHY.md)  
**Source:** `Review and paper links.xlsx` → sheet *Review* (52 usable papers) + *Downloaded Paper*.

### 5.1 From your review sheet — currency-focused

| Sheet # | Paper | Why cite |
|--------:|-------|----------|
| 1 / 17 | Currency recognition system using image processing | Classical IP / classification baseline |
| 10 | **NSTU-BDTAKA: An open dataset for Bangladeshi paper currency detection and recognition** | Direct BDT detection dataset peer work |
| 5 | AIris wearable assistive device | VI + currency in wearable context |
| 26 / 31 / 37 | Blind-assist apps with currency features | Accessibility framing |

### 5.2 From the currency technical report (keep these)

| Ref | Work | Relevance |
|-----|------|-----------|
| [1] | YOLOv5 BDT detection (3,111 annotated images); P/R 0.95/0.93 | Closest detection baseline |
| [2] | arXiv:2510.20267 — YOLOv8n+SE, multi-currency; mAP50 0.972, mAP50-95 0.804 | Strong recent comparator |
| [3] | arXiv:2509.15045 — YOLO + domain randomization | Methodological support |
| [4] | Axis-symmetrical masks NN; 98.57% acc, 8 denoms | Classification line |
| [5] | Real-time CNN for VI users (BDT) | Accessibility angle |
| [6] | BanglaTaka Mendeley dataset | Primary note source |
| [7] | arXiv:2602.07015 — Dual-stream MobileNet+EfficientNet | Classification SOTA-style |
| [8] | Lin et al., COCO, ECCV 2014 | Backgrounds |
| [9] | Ultralytics YOLOv8 v8.4.64 | Framework |

**Comparison table (report Table 8):**

| Work | Task | mAP@0.5 | mAP@0.5:0.95 | P / R (or acc) |
|------|------|---------|--------------|----------------|
| YOLOv5 taka [1] | Detect | — | — | 0.95 / 0.93 |
| YOLOv8n+SE [2] | Detect (30 classes) | 0.972 | 0.804 | 0.965 / 0.952 |
| **This work** | Detect (9 BDT) | **0.995** | **0.847** | **0.998 / 0.997** |

---

## 6. Methodology

### 6.1 Datasets

#### Foreground — BanglaTaka [6]
- Name: *A Diverse Image Dataset for Bangladeshi Currency Recognition*
- Link: https://data.mendeley.com/datasets/3cv2sypkkh/1  
- Size: **5,073** background-free note images  
- Capture: mobile phones + CamScanner-style crops  

| Denom | 2 | 5 | 10 | 20 | 50 | 100 | 200 | 500 | 1000 | Total |
|-------|---|---|----|----|----|-----|-----|-----|------|-------|
| Count | 445 | 660 | 419 | 480 | 416 | 565 | 423 | 1243 | 422 | **5073** |

#### Backgrounds — COCO val2017 [8]
- **1,500** images downloaded (`scripts/download_backgrounds.py`)  
- Intended local path: `C:\currency_backgrounds\`  

#### Classes (`data/data.yaml`)
```
0: 2_taka, 1: 5_taka, 2: 10_taka, 3: 20_taka, 4: 50_taka,
5: 100_taka, 6: 200_taka, 7: 500_taka, 8: 1000_taka
nc: 9
```

### 6.2 Synthetic compositing pipeline

**Script:** `scripts/generate_synthetic_dataset.py`  
**Canvas:** 640×640  
**Composites per source note:** 4  
**Split:** 80/10/10 **by source note** (no leakage)  
**Seed:** 42  
**Negatives:** 10% background-only  

| Split | Positives | Negatives | Total | Note instances |
|-------|-----------|-----------|-------|----------------|
| Train | 16,224 | 1,620 | 17,844 | 16,224 |
| Valid | 2,016 | 217 | 2,233 | 2,016 |
| Test | 2,052 | 186 | 2,238 | 2,052 |
| **Total** | **20,292** | **2,023** | **22,315** | **20,292** |

**Augmentations (note stage):**
| Operation | Setting |
|-----------|---------|
| Perspective warp | 70% prob.; corner jitter 0.08 |
| Rotation | ±30°, expand, transparent fill |
| Scale | note occupies 0.25–0.90 of canvas |
| HSV jitter | H ±18°, S ×0.5–1.5, V ×0.4–1.6 |
| Brightness / contrast | 0.4–1.6 / 0.7–1.3 |

**Post-composite:**
| Operation | Setting |
|-----------|---------|
| Motion blur | 20% (kernel 3/5/7, H or V) |
| Gaussian noise | 50% (std=12) |
| JPEG recompression | quality 55–92 |

**Label generation:** analytic YOLO `cx, cy, w, h` from paste geometry — no hand labelling.

Figures: `results/report_assets/compositing.png`, `methodology.png`

### 6.3 Model & training

| Item | Value |
|------|-------|
| Architecture | YOLOv8s (Ultralytics), COCO-pretrained |
| Backbone / neck / head | CSPDarknet (C2f+SPPF) / PAN-FPN / 3-scale decoupled |
| Losses | CIoU + DFL + BCE (gains 7.5 / 0.5 / 1.5) |
| Epochs | 80 |
| Image size | 640 |
| Batch | 16 (auto) |
| Patience | 15 |
| Optimizer | SGD (auto); lr0=0.01, lrf=0.01, momentum=0.937, wd=0.0005 |
| Warmup | 3.0 epochs |
| Mosaic / Mixup | 1.0 / 0.15 |
| Flip LR / UD | **0.0 / 0.0** (orientation matters for notes) |
| Extra aug | hsv_h=0.02, hsv_s=0.7, hsv_v=0.4, degrees=20, perspective=0.0005, scale=0.5, translate=0.1, shear=5 |
| Framework | Ultralytics 8.4.64, PyTorch 2.6.0+cu124 |
| Hardware | NVIDIA RTX 4060 Laptop 8GB, Ryzen 7, Windows 11 |

**Script:** `scripts/train.py`  
**Weights shipped:** `models/best.pt`, `models/best.torchscript`, `models/best.pts`  
**Curves:** `results/training_v2/epoch_results.png` — mAP@0.5 > 0.97 by epoch 8, > 0.99 by epoch 11

---

## 7. Results

### Overall (test set, 2,238 images)

| Metric | Value |
|--------|-------|
| Precision | **0.998** |
| Recall | **0.997** |
| mAP@0.5 | **0.995** |
| mAP@0.5:0.95 | **0.847** |
| Inference | **7.3 ms/image ≈ 135 FPS** (RTX 4060) |
| Preprocess / postprocess | 1.7 ms / 1.7 ms |

### Per-class (test)

| Class | Images | Inst. | P | R | mAP50 | mAP50-95 |
|-------|--------|-------|---|---|-------|----------|
| 2 Taka | 180 | 180 | 0.998 | 1.000 | 0.995 | 0.838 |
| 5 Taka | 264 | 264 | 0.998 | 0.996 | 0.995 | 0.844 |
| 10 Taka | 172 | 172 | 0.995 | 0.988 | 0.995 | 0.848 |
| 20 Taka | 192 | 192 | 0.998 | 1.000 | 0.995 | 0.847 |
| 50 Taka | 172 | 172 | 1.000 | 0.993 | 0.995 | 0.851 |
| 100 Taka | 228 | 228 | 0.999 | 1.000 | 0.995 | 0.845 |
| 200 Taka | 172 | 172 | 0.999 | 1.000 | 0.995 | 0.843 |
| 500 Taka | 500 | 500 | 1.000 | 1.000 | 0.995 | 0.853 |
| 1000 Taka | 172 | 172 | 0.994 | 1.000 | 0.995 | 0.858 |
| **All** | **2238** | **2052** | **0.998** | **0.997** | **0.995** | **0.847** |

Confusion matrices: `results/training_v2/` (nearly diagonal; residual errors mainly background FP/FN, not denomination swaps).

---

## 8. Real-time system

**Script:** `scripts/realtime_detect.py`

| Setting | Value |
|---------|-------|
| Model | `models/best.pt` |
| Camera | index 0 (`CAP_DSHOW` on Windows) |
| Detect conf | 0.35 |
| Speak conf | 0.55 |
| Frame skip | infer every 2 frames |
| Brighten | if `mean < 80`, gain=3.0, gamma=0.5 |
| TTS | Windows SAPI (`win32com`), background thread |
| Controls | SPACE = speak; q/ESC = quit |

---

## 9. Reproducibility

```powershell
cd "e:\Final SP\realtime_bangla_taka_detection"
python -m venv venv
.\venv\Scripts\pip.exe install -r requirements.txt

# Full pipeline (needs BanglaTaka + disk paths updated):
.\venv\Scripts\python.exe scripts\download_backgrounds.py
.\venv\Scripts\python.exe scripts\generate_synthetic_dataset.py
.\venv\Scripts\python.exe scripts\train.py
.\venv\Scripts\python.exe scripts\evaluate.py

# Live demo (weights included):
.\venv\Scripts\python.exe scripts\realtime_detect.py
```

**Path caveat:** Scripts hardcode paths such as `C:\currency_yolo_data`, `C:\currency_backgrounds`, and a OneDrive BanglaTaka path under another username. Update paths before regenerating data.

**Large data not in git:** generated YOLO images and COCO backgrounds live outside the repo by design.

---

## 10. Limitations & future work (for Discussion)

1. Composites lack real cast shadows, folds, heavy occlusion → collect modest **in-the-wild** photos.  
2. Detects **denomination**, not authenticity / counterfeit.  
3. Training notes are relatively good-condition; worn/torn/folded notes remain open.  
4. Edge deployment: INT8, ONNX / TensorRT / TFLite; continuous audio for VI users.  
5. Integration into Raspberry Pi smart-glass (links to Paper 2).

---

## 11. Figures available for the paper

| Asset | Path |
|-------|------|
| Architecture | `results/report_assets/architecture.png` |
| Compositing | `results/report_assets/compositing.png` |
| Methodology | `results/report_assets/methodology.png` |
| Report previews | `results/report_assets/preview_p*.png` |
| Training curves | `results/training_v2/epoch_results.png`, `results.png` |
| Confusion matrices | `results/training_v2/confusion_matrix*.png` |
| PR/F1 curves | `results/training_v2/test_eval/` |
| Webcam diagnostics | `results/webcam_diag/`, `results/webcam_diag2/` |

---

## 12. Suggested paper section map

See `PAPER_OUTLINES.md` § Paper 1.

---

## 13. Link to Smart Glass (Paper 2)

Paper 1 stands alone as a CV contribution. For a joint narrative: the detector supplies the currency capability used (or intended) by the offline assistive glass in Paper 2. Today the glass repo still uses HSV + optional MobileNet — cite that as current prototype vs planned YOLO integration if you have not merged them yet.
