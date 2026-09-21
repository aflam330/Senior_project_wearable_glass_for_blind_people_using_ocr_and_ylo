# Savior Glass / RoboEye — Progress and Publication Status

**Date:** 21 September 2026  
**Project:** Wearable assistive smart glass (OCR, objects, Bangladeshi Taka, emotion-adaptive speech)  
**Repos:** `savior_glass`, `realtime_bangla_taka_detection`

This note is for sharing with a supervisor or collaborator. Every accuracy figure below was measured in this workspace. Do not replace them with draft paper numbers.

---

## Executive summary

The system is a **working senior-project prototype**, not an A* conference paper.

- Emotion recognition on the **official RAF-DB test set** (3,068 images) is **86.5%** accuracy (macro-F1 0.80). It was **81.0%** before this week’s GPU training. The requested **93–96%** was **not** reached.
- Published 7-class RAF-DB SOTA is about **92.2%** (POSTER V2). 93–96% is above typical published numbers on this split.
- Taka detection on 450 wild notes: **98.4%** detection, **96.9%** top-1 denomination.
- **Not eligible for A* venues** (CVPR, ICCV, NeurIPS, ICLR, CHI) as the work stands: existing models were fine-tuned; there is no new algorithm and no controlled user study.

**Eligible today:** thesis, demo, IEEE application / regional conference.  
**Not eligible today:** CVPR / NeurIPS / CHI.

---

## What is implemented

| Area | Implementation | Status |
|------|----------------|--------|
| Emotion | EfficientNet-V2-S on RAF-DB; live path in `roboeye/fer_emotion.py` (Haar face, FER+ ONNX fallback, emotion-adaptive TTS) | Working, **86.5%** RAF-DB |
| Currency | YOLOv8s, 9 Bangladeshi Taka classes; ONNX/INT8 export | Working |
| Authenticity | JaalTaka CNN+ViT, dual-head, FedAvg, EWC scripts | Code works; **do not publish split acc 1.00** |
| OCR | EasyOCR Bangla + English | Working |
| Objects | YOLOv8 COCO | Working |
| Assistive I/O | Offline TTS, ASR intents, haptics, solvePnP straighten, Grad-CAM overlay | PC/Windows working; not bone-conduction or VR headset |
| Glass app | `savior_glass` Pi + `test_windows.py` | Working |

Live weights for emotion: `savior_glass/models/emotion_faces.pt` (arch `efficientnet_v2_s`, 224 px).

---

## Emotion experiments (this week)

GPU: NVIDIA GeForce RTX 3050 Laptop (CUDA PyTorch installed; it had been CPU-only). Dataset already on disk: official RAF-DB train 12,271 / test 3,068.

| Run | Test set | Accuracy | Kept as live model? |
|-----|----------|----------|---------------------|
| FER+ ONNX (older eval) | FER-2013 holdout, n=5,741 | 57.9% | Fallback only |
| HGB + FER+ blend | FER-2013 holdout, n=5,741 | 67.6% | No |
| MobileNetV3-Large (previous) | RAF-DB official test, n=3,068 | 81.0% | Replaced |
| **EfficientNet-V2-S + flip TTA** | **RAF-DB official test, n=3,068** | **86.5%** | **Yes** |
| ViT (`trpakov/vit-face-expression`) fine-tune | RAF-DB official test | 84.3% peak | Stopped |
| AffectNet EfficientNet-B2 fine-tune | RAF-DB official test | 85.2% then overfit | Stopped |

### Best model — per class (RAF-DB official test)

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| Happy | 1185 | 0.946 | 0.925 | 0.936 |
| Sad | 478 | 0.839 | 0.839 | 0.839 |
| Angry | 162 | 0.789 | 0.809 | 0.799 |
| Surprise | 329 | 0.867 | 0.872 | 0.870 |
| Fear | 74 | 0.774 | 0.554 | 0.646 |
| Disgust | 160 | 0.671 | 0.588 | 0.627 |
| Neutral | 680 | 0.815 | 0.890 | 0.851 |
| **Overall** | **3068** | macro-P **0.815** | macro-R **0.782** | **acc 86.5% / macro-F1 0.795** |

Bottleneck: Fear (74 test images) and Disgust (160). Happy is already strong (F1 0.936).

---

## Other measured results (already in the repo)

| Task | Protocol | Result |
|------|----------|--------|
| Taka wild notes | 450 images, existing YOLOv8s `best.pt` | Detection 98.4%, top-1 96.9% |
| Taka YOLO val | Saved val metrics | mAP@0.5 0.994, mAP@0.5:0.95 0.844 (synthetic composites) |
| OCR | EasyOCR, n=80 (40 BN + 40 EN) | CER 15.5% overall (BN 27.8%, EN 3.2%) |
| Objects | YOLOv8s, 250-image COCO val subset | mAP@0.5 0.60 |
| Authenticity | Local JaalTaka split | Acc 1.00 — **same-capture leak; not a paper number** |

---

## A* conference eligibility

A* here means CORE A* venues such as **CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR**, and CHI-style HCI A* tracks.

| Expectation | This project |
|-------------|--------------|
| New method others will cite | Fine-tunes YOLO, EfficientNet, FER+, EasyOCR |
| SOTA or a large clean gain on a standard benchmark | Emotion 86.5% vs ~92% SOTA; Taka numbers are strong but application-specific |
| Ablations, seeds, fair baselines | Mostly single-run figures |
| Or a CHI/ASSETS user study | No controlled study with visually impaired users |

**Verdict**

- **CVPR / ICCV / ECCV — No.**
- **NeurIPS / ICML / ICLR — No.**
- **CHI / ASSETS — Not yet** (need a user study; the prototype is only a prerequisite).

**What you can submit now:** undergraduate/graduate thesis; system demo; IEEE application or regional venues (e.g. TENCON, ICCIT, ICAEE). A later ASSETS-style paper still needs a user study and honest metrics. It would still not be CVPR unless the *method* is new.

---

## Do not claim in any draft or email

- 93–96% emotion accuracy (not measured).
- 96.7% / 98.1% authenticity (never measured by this stack).
- Raspberry Pi 15 FPS (not timed on a Pi).
- Qwen-VL / LLaVA / bone-conduction hardware (not what runs today).
- Full 22,315-image composite set on this machine (pipeline exists; full set not generated here).

---

## Recommended next step

Freeze **86.5%** emotion and present this as a **thesis + live demo**. Record a short video: note detection, speech, emotion-adaptive TTS.

Only restart emotion training if the goal is a higher *honest* RAF-DB number (e.g. POSTER V2), and report that number even if it is 90–92%.

---

## Key files

- `savior_glass/models/emotion_faces.pt` — live EfficientNet-V2-S weights  
- `savior_glass/results/emotion_rafdb.json` — official RAF-DB metrics  
- `savior_glass/scripts/train_emotion_rafdb.py` — training entrypoint  
- `realtime_bangla_taka_detection/roboeye/fer_emotion.py` — runtime detector  
- `realtime_bangla_taka_detection/models/best.pt` — Taka YOLO  

*Prepared from local evaluation JSON in this workspace, 21 September 2026.*
