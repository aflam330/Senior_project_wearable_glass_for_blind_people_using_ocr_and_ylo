# Savior Glass: An Offline, Bangla-First Assistive Smart Glass with Bangladeshi Taka Recognition and Multi-View Counterfeit Authentication

**Technical project report, generated from the repository state of 29 September 2026**

| | |
|---|---|
| Repository | `E:\Final SP` (GitHub: `aflam330/Senior_project_wearable_glass_for_blind_people_using_ocr_and_ylo`) |
| Author(s), supervisor, institution | Not available in the current project repository. Git commits are by `aflam330`; the project repository is `aflam330/Senior_project_wearable_glass_for_blind_people_using_ocr_and_ylo`. The thesis template in `Thesis Report and paper/` has an IUB logo, but the repository does not name the institution. |
| Commit range inspected | `922d8ec` (2026-09-05) to `71ad34c` (2026-09-29), 11 commits |
| Method of preparation | Source code, configuration, saved result files (JSON/CSV/PNG), and the write-ups in `paper_evidence/` and `docs/` were inspected. No code was executed to produce new numbers for this report. Every number is copied from a saved file named next to it. |

---

> **Update, later on 2026-09-29.** Several findings below have since been acted on (`paper_evidence/CORRECTIONS.md`):
>
> - **Jaal verdict switched off.** The glass's genuine/jaal verdict was measured on whole-note photographs. It was wrong for 47.7 % of genuine Bangla Money photos, 59.2 % of BanglaTaka photos and 20.4 % of genuine counterfeit-dataset photos, and it missed 40 % of counterfeits (`paper_evidence/JAAL_VERDICT_FIXED.md`). It is now switched off, and the glass says the check was not done. Sections 8.7 and F.1 item 7 describe the earlier behaviour.
> - **Detector test re-run.** The test split was re-evaluated: 2,282 images, P 0.9949, R 0.9963, mAP50 0.9949, mAP50-95 0.8489 (`results/training_v2/test_eval/test_metrics.json`). This replaces the "partially verified" README numbers. Consistency Audit items 1–4, 7–9 and 11 are corrected in the source documents.

## How to read this report

The report follows the requested structure:

- **Part A**: project understanding
- **Part B**: architecture
- **Part C**: the full academic report (Chapters 1–9)
- **Parts D–J**: inventory, results, limitations, contribution, figures, claim audit and completeness scorecard

It ends with a consistency audit and a list of the information still needed.

Three labels are used throughout:

- **VERIFIED**: the number or behaviour is in a saved artifact or in source code that was read.
- **PARTIALLY VERIFIED**: the evidence exists but is incomplete, for example a plot without a metrics file, or a test on simulated hardware.
- **NOT MEASURED**: the repository says so explicitly, or no evidence was found.

---

# PART A — PROJECT UNDERSTANDING

| Item | Finding |
|---|---|
| **Project title** | *Savior Glass* / "Wearable Smart Glass for Blind People (OCR + YOLO)" (`README.md`, thesis draft `Thesis Report and paper/thesis report/My report/abstract.tex`) |
| **Project type** | Hybrid system: (1) an embedded/edge assistive application for Raspberry Pi 5; (2) a computer-vision research project on banknote detection; (3) a deep-learning research project on multi-view counterfeit authentication, with a large set of controlled experiments |
| **Main objective** | Give visually impaired users in Bangladesh an offline device that reads Bangla/English text aloud, announces nearby objects, and identifies Bangladeshi Taka notes, including a genuine-versus-counterfeit ("jaal") check |
| **Core technologies** | Python; PyTorch/torchvision; Ultralytics YOLOv8; ONNX / ONNX Runtime (FP32 and INT8); OpenCV; EasyOCR; espeak-ng / Piper TTS (Pi) and Windows SAPI (desktop); rpi-lgpio (GPIO); MobileNetV3-Small + a small Vision Transformer; EfficientNet-V2-S (emotion) |
| **Main components** | `savior_glass/`: the Pi application (modes, buttons, TTS, camera). `realtime_bangla_taka_detection/`: Taka detector, the `roboeye` library (authenticity models, Q-DUIG/CAMVA, emotion, pose, haptics, ASR), training/evaluation scripts and saved results. `paper_evidence/`: result write-ups, tables, figures and a claim registry. |
| **Implementation status** | The software is implemented and runs on a Windows laptop (with an RTX 3050 GPU). The Pi application code is complete, but it has **not been run on a Raspberry Pi 5** (`paper_evidence/PI5_RESULTS.md`). GPIO and haptics were tested only with a simulated GPIO module. **No user study** with visually impaired participants has been conducted (`paper_evidence/user_study/statistical_report.json`). |

---

# PART B — PROJECT ARCHITECTURE

## B.1 Architecture explanation

The workspace contains two cooperating code bases:

1. **`savior_glass/`: the deployable device application.** `main.py` builds a `SmartGlass` object that owns:
   - a camera thread (`utils.CameraManager`, OpenCV `VideoCapture`, 640×480, 2-frame buffer)
   - a TTS worker thread (`utils.TTSEngine`: Piper first, espeak-ng fallback, queue of 5)
   - a detection thread (object-mode auto-scan every 1.5 s)
   - GPIO interrupt callbacks (`button_handler.ButtonHandler`, rpi-lgpio)

   Four modes are cycled by the MODE button: OCR → Object → Currency → Claude (online) (`config.py`, `main.py:131-141`).

2. **`realtime_bangla_taka_detection/`: models and research code.** It supplies the trained YOLOv8s Taka detector (`models/best.pt`, ONNX and INT8 ONNX exports) and the authenticity network. The authenticity network is the prefix-robust Q-DUIG checkpoint `results/qduig/prefix_ft/seed42/checkpoint.pt`, called **PRMVT** in the evidence files. The glass imports both at runtime: `savior_glass/modes/currency_mode.py:149-223` searches sibling paths, and `savior_glass/assistive/__init__.py` adds `roboeye` to `sys.path`.

The same repository also holds a desktop demonstration, `scripts/roboeye_live.py` ("RoboEye" HUD). It combines detection, authenticity, facial-expression recognition, Grad-CAM, voice commands and haptics for a laptop webcam.

## B.2 Component relationships

```
                        ┌──────────────────────── savior_glass (Raspberry Pi 5 target) ───────────────────────┐
 5 GPIO buttons ──────► │ ButtonHandler ──► SmartGlass (main.py) ──► current Mode.process_frame(frame)        │
 (BCM 17,27,24,22,23)   │                         ▲                         │                                 │
 Camera (640×480) ────► │ CameraManager ──────────┘                         ▼                                 │
                        │                                          TTSEngine (Piper → espeak-ng) ──► speaker  │
                        │ Modes: OCRMode | ObjectMode | CurrencyMode | ClaudeMode (online)                    │
                        │           │           │            │  └── haptic motor, BCM 13                     │
                        └───────────┼───────────┼────────────┼──────────────────────────────────────────────┘
                                    │           │            │ imports weights / code
            EasyOCR(bn,en)+CLAHE+   │  YOLOv8s  │            ▼
            lexicon repair, Ekush   │  (COCO)   │   realtime_bangla_taka_detection
            CNN, optional Tesseract │           │     models/best.pt (YOLOv8s, 9 Taka classes)
                                                │     roboeye/qduig  → PRMVT checkpoint (genuine / jaal)
                                                │     roboeye/authenticity (CNN+ViT fallback)
```

## B.3 Data flow and processing pipeline

**Currency mode (ACTION press).** The code path is `main.py:143-175` → `CurrencyMode.process_frame` → `detect_live` (`currency_mode.py:103-327`):

1. Frame → YOLOv8s predict (conf 0.25, imgsz 640).
2. If the frame mean is below 80 and no box was found, retry once on a brightened copy (gain 3, gamma 0.5).
3. For up to two boxes with confidence ≥ 0.35 and a crop of at least 24×24 px, run PRMVT on the crop as **one view** (128×128, ImageNet normalisation). The note is called genuine if p(genuine) ≥ 0.5, otherwise counterfeit.
4. Build a Bangla sentence, e.g. "একশত টাকার নোট। আসল" ("100 taka note. Genuine"). Play the haptic pattern: genuine = 2 short pulses, counterfeit = 3 long pulses.
5. Send the sentence to the TTS queue.
6. If the YOLO weights are missing, fall back to the older HSV-colour heuristic, with an optional MobileNetV3 classifier on top.

**OCR mode.** ACTION captures, runs OCR and stores the text; READ speaks it (`ocr_mode.py`, `main.py:177-194`). The OCR steps are:

1. Preprocess: upscale → denoise → CLAHE (clip 2.5, 8×8 tiles).
2. EasyOCR with Bangla and English.
3. Filter detections: confidence ≥ 0.4, at least 3 characters, at least 50 % letters or digits.
4. Sort into reading order.
5. Repair: NFC normalisation plus lexicon repair from `assets/ocr_lexicon.txt`.
6. Fallbacks: optional Tesseract, and an Ekush handwritten-letter CNN.
7. Split the text into Bangla and English segments so each is spoken by the matching voice.

**Object mode.** A background loop runs every 1.5 s:

1. Resize the frame to 320×240.
2. Run YOLOv8s (COCO), falling back to YOLOv8n, at confidence ≥ 0.5.
3. Translate labels to Bangla with `assets/labels_bn.json`.
4. Apply a 4 s cooldown per class. ACTION forces an immediate announcement.

**Claude mode (online, optional).** The frame is JPEG-encoded and sent to the Anthropic Messages API with a Bangla assistive prompt (`modes/claude_mode.py`). This mode needs `ANTHROPIC_API_KEY` and internet access.

---

# PART C — COMPLETE REPORT

## ABSTRACT

Visually impaired people in Bangladesh have few assistive devices that work offline, speak Bangla, and recognise local banknotes. This project implements *Savior Glass*, a Raspberry Pi 5-targeted smart-glass application with physical push-buttons and spoken output. It has three offline modes: Bangla/English text reading with EasyOCR, everyday-object announcement with YOLOv8, and Bangladeshi Taka recognition. A fourth mode is optional and online, using a cloud vision model.

For currency recognition, a copy-paste compositing pipeline places 5,073 cropped note photographs onto COCO backgrounds. It generated 22,333 labelled images, and a YOLOv8s detector was fine-tuned on them. On an independent set of 1,536 Bangladeshi note photographs (Bangla Money, Kaggle), which was not used for training or selection, the detector names the correct denomination for 91.5 % of notes. On hand-held close-ups from NSTU-BDTAKA it does so for only 18.5 %.

For counterfeit detection, a MobileNetV3-Small + small-ViT multi-view authenticator was trained on the JaalTaka collection (1,390 physical notes, six views each, note-disjoint split). A prefix-robust training protocol, which trains on the first k views for varying k, raises single-view test accuracy from 73.6 % (CNN+ViT baseline) to 97.1 % (202/208 notes, seed 42; McNemar p = 4.3×10⁻¹¹). Six-view accuracy is 98.1 % mean over three seeds. Learned view-selection policies did not beat a fixed view order. The authenticator is sensitive to heavy occlusion, strong low light and over-exposure.

Supporting modules were measured separately:

| Module | Test data | Result |
|---|---|---|
| Emotion recogniser | RAF-DB official test | 86.5 % accuracy |
| OCR | 80 synthetic rendered phrases | CER 3.2 % (English), 27.8 % (Bangla) |

Raspberry Pi 5 latency, real hardware operation, live-note accuracy, and a user study with visually impaired participants have **not** been measured. The project therefore demonstrates a working software prototype and a controlled authentication study, not a validated assistive device.

**Keywords:** assistive technology; smart glass; Bangla OCR; YOLOv8; Bangladeshi banknote recognition; counterfeit detection; multi-view learning; edge AI

## TABLE OF CONTENTS

1. Introduction
2. Literature Review / Related Work
3. Requirements and System Analysis
4. System Design
5. Methodology
6. Implementation
7. Experiments and Evaluation
8. Results and Discussion
9. Conclusion and Future Work
References · Appendices · Implementation Traceability · Claim Verification · Consistency Audit · Information Required

---

## CHAPTER 1 — INTRODUCTION

### 1.1 Background

The WHO blindness and vision-impairment fact sheet is cited in the thesis bibliography (`ref.bib`). Camera-based assistive devices usually combine text reading, object recognition and audio feedback. Currency handling is a specific daily difficulty. Bangladeshi Taka notes come in nine circulating denominations (2–1000 Taka), and counterfeit ("jaal") notes circulate. Many published assistive systems depend on a phone or a cloud service and target English. This point comes from the thesis abstract and the literature spreadsheet, not from a systematic review.

### 1.2 Problem statement

The project addresses three technical problems:

1. **Offline, Bangla-first assistance on low-cost hardware.** Text, object and currency information must be delivered by speech in Bangla on a Raspberry Pi-class device, without internet access.
2. **Taka denomination recognition without a large labelled detection dataset.** The available BanglaTaka images are tight crops of notes. Real use involves notes at arbitrary positions in cluttered scenes.
3. **Counterfeit authentication when the number of available views varies.** A classifier trained on six fixed views of a note loses accuracy when a user supplies fewer views. The measured drop is to 73.6 % at one view for the CNN+ViT baseline (`results/qduig/eval/seed42/baseline/`).

### 1.3 Motivation

**Documented motivation** (thesis abstract, `docs/00_OVERVIEW.md`, `README.md`): a Bangla-speaking, offline, affordable aid that "understands local banknotes".

**Reasonable technical interpretation** (this report's inference, not a documented statement): the research focus shifted to authentication because it is the harder and less-studied problem. The evidence is that the bulk of the commits and result files after 2026-09-21 concern JaalTaka authentication (`paper_evidence/`, `results/qduig/`, `results/novel*/`).

### 1.4 Objectives

**Primary**
- Build an offline smart-glass application with OCR, object and currency modes controlled by physical buttons and speech (`savior_glass/`).
- Detect and name all nine Taka denominations in real time (`realtime_bangla_taka_detection/`).

**Secondary**
- Add a genuine/counterfeit verdict to currency announcements (`currency_mode.py:195-315`).
- Add emotion-adaptive speech and haptic feedback (`roboeye/fer_emotion.py`, `roboeye/haptics.py`).

**Research**
- Make a multi-view authenticator accurate for any number of views from 1 to 6 (prefix robustness).
- Test whether learned view selection or adaptive stopping can reduce the number of views needed (`roboeye/qduig/policy.py`).
- Report calibration, robustness, multi-seed variation and negative results with artifact-level traceability (`paper_evidence/CLAIM_REGISTRY.json`).

### 1.5 Scope

**In scope:**
- nine Taka paper denominations
- genuine versus counterfeit on the JaalTaka collection
- Bangla/English printed text
- 80 COCO object classes
- a Raspberry Pi 5 software target plus a Windows test harness

**Out of scope, or not implemented:**
- coins
- foreign currency
- security-feature localisation (hologram, thread, watermark); JaalTaka has no such labels (`NOVELTY_DECLARATION.md`)
- navigation and obstacle avoidance
- bone-conduction audio hardware (`FEATURE_AUDIT_REPORT.md`)

### 1.6 Research questions (reconstructed from the pre-registered hypotheses in `paper_evidence/FINAL_AUDIT.md`)

- **H1.** Can the proposed method match the CNN+ViT baseline while using fewer views at a predefined operating point? **Result: FAIL.**
- **H2.** Is it better calibrated than the baseline? **Result: PASS** (early Q-DUIG model).
- **H3.** Does it degrade less under corruptions? **Result: FAIL** on occlusion and low light.
- **H4.** Does the learned policy reduce acquisition cost at comparable reliability? **Result: FAIL.**
- **Added later:** Does training on view prefixes remove the few-view accuracy drop? **Result: supported on this dataset** (Chapter 7).

### 1.7 Contributions (as demonstrated by the repository; see Part G)

1. A working, offline-capable smart-glass software stack with four modes, Bangla speech routing, and a currency-plus-authenticity pipeline.
2. A copy-paste compositing pipeline and a YOLOv8s Taka detector. The detector was evaluated on held-out composites and on two independent public datasets.
3. A controlled multi-view authentication study on JaalTaka. It shows that prefix-robust training removes the few-view accuracy drop. It reports the failure of learned view-selection policies, calibration, 13 corruption types, and three training seeds.
4. An evidence-tracking discipline: a claim registry, a claim validator, correction notes when evaluation bugs were found, and explicit NOT_MEASURED entries.

### 1.8 Report organisation

- Chapter 2 summarises related work available in the repository.
- Chapters 3–6 describe requirements, design, methodology and implementation.
- Chapter 7 reports experiments.
- Chapter 8 discusses them.
- Chapter 9 concludes.

---

## CHAPTER 2 — LITERATURE REVIEW / RELATED WORK

### 2.1 Existing approaches

The repository contains three literature sources:

- `docs/literature/Review and paper links.xlsx`
- `docs/RELATED_WORK_BIBLIOGRAPHY.md`: 93 KB of links and abstracts generated from the spreadsheet
- `ref.bib` in the thesis draft: 67 entries

The entries fall into four groups:

- **Assistive smart glasses and reading aids.** Examples: "Smart Glass System Using Deep Learning for the Blind and Visually Impaired", "FingerReader" (CHI), "Vision Voice: A Raspberry Pi-Based Text-to-Audio Converter", "AIris", "Divya-Drishti", and a bibliometric review of smart wearable mobility aids (2024). Most combine a camera, OCR and TTS. Several use Raspberry Pi. Several rely on cloud OCR (e.g. "Cloud based Text extraction using Google Cloud Vision").
- **Currency recognition.** Examples: "Currency recognition system using image processing" (two entries) and the NSTU-BDTAKA open dataset (Data in Brief, 2024).
- **Bangla OCR.** Example: "Bangla OCR Using Deep Learning Based Image Classification Algorithms".
- **Multi-view and missing-view learning.** These are cited in `paper_evidence/NOVELTY_DECLARATION.md` and `LITERATURE_NOVELTY_MATRIX.md`. Examples: Deep Sets, the Set Transformer (arXiv:1810.00825), focal loss (arXiv:1708.02002), generalized distillation (arXiv:1511.03643), multi-view information bottleneck (arXiv:2002.07017), sharpness-aware minimisation, and missing-view methods (Robust MVAE, RML).

### 2.2 Technologies and methods in prior work

Based on the titles and abstracts in the repository, prior assistive systems typically use Tesseract, EasyOCR or cloud OCR with a TTS engine. Currency systems use colour/texture features or CNN classifiers on cropped notes. The repository does not contain a quantitative comparison with any specific published system.

### 2.3 Existing limitations (as stated in repository documents)

The thesis abstract states that most published systems:

- depend on a phone or a cloud service
- target English
- do not handle Bangladeshi notes or counterfeit notes

These are the authors' summaries. The repository does not include a systematic review that would establish them quantitatively.

### 2.4 Research gap

The evidence supports one narrow gap statement, taken from `NOVELTY_DECLARATION.md`: "a classifier trained at a fixed view count of 6 drops to 0.5865 accuracy at 1 view on JaalTaka. The fix is supervised prefix coverage, not a new missing-view theory."

The literature search behind that statement was a web-index search. It is explicitly **not** a systematic search of Scopus, IEEE Xplore, ACM DL or Web of Science.

### 2.5 Position of the proposed project

- **Assistive system.** The project is a system integration of known components (YOLOv8, EasyOCR, Piper/espeak) for Bangla and Taka. It adds a counterfeit verdict.
- **Authentication research.** The repository's own labels for its methods are "EXTENDS" or "ADAPTED" prior work, never "new".

**A formal literature review should be added before publication.** It should compare quantitatively against published Taka recognition results on NSTU-BDTAKA.

---

## CHAPTER 3 — REQUIREMENTS AND SYSTEM ANALYSIS

The repository contains no formal requirements document. Requirements below are **inferred from the implementation** unless marked *(documented)*.

### 3.1 Functional requirements

| ID | Requirement | Source | Status |
|---|---|---|---|
| FR1 | Cycle modes with one button and announce the mode in Bangla | `main.py:131-141`, `config.MODE_NAMES_BN` | Implemented; simulated-GPIO test PASS |
| FR2 | OCR: capture on ACTION, speak on READ | `main.py:143-194` | Implemented |
| FR3 | Read Bangla and English printed text | `ocr_mode.py` (EasyOCR `["bn","en"]`) *(documented in README)* | Implemented; CER measured |
| FR4 | Recognise isolated handwritten Bangla letters | `ekush_letters.py` | Implemented; weights are local, not in git |
| FR5 | Announce nearby objects with a per-class cooldown | `object_mode.py` | Implemented |
| FR6 | Detect and name 9 Taka denominations | `currency_mode.detect_live` *(documented)* | Implemented |
| FR7 | Say genuine or counterfeit for a detected note | `currency_mode._authenticity` | Implemented; accuracy on live crops NOT MEASURED |
| FR8 | Haptic verdict pattern on BCM 13 | `currency_mode._buzz` | Implemented; hardware NOT MEASURED |
| FR9 | Volume up/down in steps of 10 %, clamped 0–100 | `main.py:196-204` | Implemented; simulated test PASS |
| FR10 | Optional online scene description | `modes/claude_mode.py` | Implemented; requires API key |
| FR11 | Emotion-adaptive speech (desktop harness) | `roboeye/fer_emotion.py`, `test_windows.py` | Implemented on Windows harness |

### 3.2 Non-functional requirements

| Requirement | Evidence | Status |
|---|---|---|
| Offline operation *(documented: "Fully Offline" in `main.py` docstring)* | `paper_evidence/edge/offline_validation.json` lists modules as offline, but its `full_pipeline_network_disabled_run` is "NOT MEASURED". Claude mode is online. | Partially verified |
| Real-time or interactive latency on Pi 5 *(documented target)* | `PI5_RESULTS.md`: NOT_MEASURED | Not verified |
| Robustness to lighting | Dark-frame retry in detector; low-light robustness of authenticator measured (Ch. 7) | Partially verified |
| Reliability of concurrent input | `_inferring` guard ignores overlapping ACTION presses (`main.py:162-165`); 250 ms debounce | Implemented |
| Maintainability | Central `config.py`; modes share a `BaseMode` interface | Implemented |
| Reproducibility *(documented in `CONTRIBUTING.md`)* | Saved configs, seeds, splits and checkpoints; see §8.6 | Partially verified |

### 3.3 System constraints

- Raspberry Pi 5 CPU only. No GPU acceleration is assumed on the device (`easyocr.Reader(..., gpu=False)`).
- EasyOCR needs about 600 MB of weights, which are lazy-loaded (`ocr_mode.py` docstring).
- A Bangla offline voice is available only via espeak-ng or Piper. The Windows harness has English SAPI voices only (`SPEECH_TEST.md`).
- Voice commands in Bangla are not available offline: "no Bangla Vosk model exists" (`savior_glass/requirements.txt`).

### 3.4 Hardware requirements

These are documented in `config.py`, `install.sh`, `README.md` and `smart_glass.service`:

- Raspberry Pi 5
- Pi camera via libcamera, or a USB camera (index 0, 640×480 at 30 fps)
- five momentary push-buttons on BCM 17 (MODE), 27 (ACTION), 24 (READ), 22 (VOL+), 23 (VOL−), with pull-ups and falling-edge interrupts
- a vibration motor on BCM 13
- an audio output device

A bill of materials, a wiring diagram, a power supply, an enclosure or glasses frame, and battery specifications are **not available in the current project repository**.

### 3.5 Software requirements

- Python ≥ 3.10 (`setup.py`)
- the packages in `savior_glass/requirements.txt` (easyocr, ultralytics, opencv-python-headless, torch, torchvision, rpi-lgpio, numpy, Pillow) and `realtime_bangla_taka_detection/requirements.txt`
- espeak-ng and optional Piper (installed by `install.sh`)
- systemd service `smart_glass.service`

### 3.6 Feasibility analysis

**Technical feasibility.**
- All models run on a laptop.
- Laptop-CPU timings exist (Ryzen 7 5800H). The Taka detector takes 105 ms (PyTorch) and 60.8 ms (INT8 ONNX); PRMVT takes 10.4–21.3 ms (`GPU_SPEED.md`).
- A Pi 5 CPU is slower than this laptop CPU, but no Pi measurement exists, so feasibility on the target device is **not established**.

**Economic feasibility.** Not available in the current project repository (no cost/BOM).

---

## CHAPTER 4 — SYSTEM DESIGN

### 4.1 Overall architecture

The system uses a thread-per-concern design around a single mode state (Part B.1):

- GPIO callbacks change the mode or trigger inference.
- A camera thread keeps only the freshest frames.
- A TTS worker serialises speech.
- An object-mode loop runs automatic scans.

### 4.2 System components

| Component | File | Responsibility |
|---|---|---|
| `SmartGlass` | `savior_glass/main.py` | lifecycle, mode switching, button callbacks, detection loop, SIGTERM shutdown |
| `ButtonHandler` | `savior_glass/button_handler.py` | rpi-lgpio setup, debounce, callbacks |
| `CameraManager` | `savior_glass/utils.py:305+` | capture thread, frame buffer |
| `TTSEngine` | `savior_glass/utils.py:127+` | Piper/espeak-ng synthesis, language segmentation, volume, queue |
| `OCRMode` | `savior_glass/modes/ocr_mode.py` | preprocessing, EasyOCR, repair, capture/read split |
| `ObjectMode` | `savior_glass/modes/object_mode.py` | YOLOv8 COCO, Bangla labels, cooldowns |
| `CurrencyMode` | `savior_glass/modes/currency_mode.py` | YOLOv8s Taka, authenticity, haptics, HSV/MobileNet fallback |
| `ClaudeMode` | `savior_glass/modes/claude_mode.py` | online vision API call |
| `roboeye.qduig` | `realtime_bangla_taka_detection/roboeye/qduig/` | authentication network, fusion, policies, calibration, corruptions |
| `roboeye.camva` | `realtime_bangla_taka_detection/roboeye/camva/` | JaalTaka data loading, splits, `ViewEncoder`, CAMVA model, metrics |

### 4.3 Data flow

See Part B.3.

### 4.4 Control flow

- **Mode button.** Lock → increment mode modulo 4 → stop current speech → deactivate old mode → activate new mode, which loads models lazily → announce the mode.
- **ACTION.** In object mode it forces an announcement. In OCR or currency mode it starts a short-lived inference thread, unless one is already running, in which case the press is ignored.
- **READ.** Speaks the stored OCR or Claude text.
- **Shutdown.** SIGTERM/SIGINT → speak "বন্ধ হচ্ছে" ("shutting down") → drain TTS for up to 3 s → clean up modes, camera and GPIO.

### 4.5 Module architecture

All modes implement `activate`, `deactivate`, `cleanup` and `process_frame` (`modes/base_mode.py`).

### 4.6 Database design

Not applicable. The system has no database. It stores OCR/Claude text only in memory and logs to a rotating file (5 MB × 3, `config.py`).

### 4.7 Hardware architecture

A block-level description is possible from the pin map (§3.4). A wiring diagram, a schematic or photographs of the assembled glass are **not available in the current project repository**.

### 4.8 Software architecture: authentication network

The `QDUIGNet` network (`roboeye/qduig/model.py`) is built from the following parts:

- **Encoder.** `ViewEncoder` (`roboeye/camva/model.py:14-31`) runs two branches per view and concatenates them into a 704-dimensional feature:
  - an ImageNet MobileNetV3-Small feature extractor, frozen, giving a 576-d feature
  - a small Vision Transformer (`TinyViT`, patch 16, width 128, depth 3, 4 heads, 128×128 input), giving a 128-d feature
- **Per-view heads.**
  - quality factor head
  - residual spectral quality attention (RSQA)
  - uncertainty head
  - information-gain surrogate
- **Diversity term.** A log-determinant (volume) over the views.
- **Fusion.** Hypernetwork-gated evidence fusion (HGEF).
- **Additions for prefix robustness.**
  - a per-view self-gate `z·(1+tanh(W z))`
  - separate batch normalisation for each view count 1–6 (`k_bn`)
- **Classifiers.** A plain classifier, and a single-view auxiliary classifier.

### 4.9 User interaction

The user holds a note or text in front of the camera and presses buttons. All feedback is spoken, mostly in Bangla; for example, a detected note is announced as "একশত টাকার নোট। আসল" ("100 taka note. Genuine") or "…। জাল টাকা" ("… counterfeit taka"). Currency verdicts also produce vibration patterns. There is no screen on the glass. The Windows harness (`test_windows.py`) shows an OpenCV window and maps buttons to keys.

---

## CHAPTER 5 — METHODOLOGY

### 5.1 Overall methodology

The project has three work streams:

1. **Engineering.** Build and test the glass application on a laptop harness.
2. **Detection.** Synthesise a detection dataset, fine-tune YOLOv8s, and evaluate on held-out composites and external datasets.
3. **Authentication.** Build a note-disjoint split, train a baseline and many variants, select only on validation, evaluate once on test, and record every result, negative results included (`CONTRIBUTING.md`, `FINAL_AUDIT.md`).

### 5.2 Data collection

No new images were collected by the project, as far as the repository shows. All datasets are external or pre-existing and are kept outside git (`.gitignore`). Their provenance is listed in §7.2.

### 5.3 Data preprocessing

**Detection data.** `scripts/generate_synthetic_dataset.py` builds the detection dataset:

1. Split source note images 80/10/10 per class, **by source image**, with seed 42.
2. Make 4 composites per source image on a 640×640 canvas.
3. Use COCO val2017 photographs as backgrounds, with synthetic backgrounds when those run out.
4. Augment each composite: perspective warp; rotation ±30°; scale 0.25–0.90; HSV and brightness jitter; motion blur; Gaussian noise; JPEG recompression.
5. Add 10 % background-only negatives.
6. Compute YOLO boxes from the paste position.

**Authentication data.**
- `roboeye/camva/data.py` resizes each view to 128×128, caches it, and applies ImageNet normalisation.
- The split (`scripts/make_camva_splits.py`) is by physical note ID with seed 42 and 15 % validation / 15 % test. Leakage is 0/0/0 (`results/camva/splits/split_metadata.json`).

**OCR input.** Upscale, denoise and CLAHE (`ocr_mode.py`).

### 5.4 Algorithms

| Algorithm | Purpose | Input → Output | Key parameters (verified) | Limitations |
|---|---|---|---|---|
| Copy-paste compositing | Create labelled detection data from crops | note crop + background → 640² image + YOLO box | 4 composites/image, 10 % negatives, seed 42 | Synthetic domain; source photos share acquisition conditions |
| YOLOv8s fine-tuning | Detect and classify 9 denominations | image → boxes + class + confidence | 80 epochs, patience 15, imgsz 640, `fliplr=0`, degrees 20, hsv_s 0.7 (`scripts/train.py`) | Has no "unknown note" output (`CROSS_DATASET_TAKA.md`) |
| Dark-frame retry | Recover detections in low light | frame (mean < 80) → brightened frame | gain 3, gamma 0.5; only if first pass finds nothing | Heuristic |
| PRMVT (prefix-robust Q-DUIG) | Genuine vs counterfeit from 1–6 views | k views → p(genuine) | Stage 1: 6 epochs mixed view-dropout. Stage 2: 3 epochs, lr 3e-4, batch 8, `full_view_prob` 0.5. Loss weights λ_auth 1.0, λ_q 0.25, λ_u 0.15, λ_d 0.1, λ_g 0.2, λ_c 0.05; contrastive 0.05; single-view auxiliary 0.5 (`results/qduig/prefix_ft/seed42/config.yaml`) | Single dataset; one capture collection |
| CNN+ViT baseline | Reference authenticator | 6 views → p(genuine) | same encoder, mean fusion | Loses accuracy at few views |
| Sequential policies (CRIQP, quality, confidence, diversity) | Choose next view and when to stop | partial views → next view / stop | λ_cost ∈ {0.02 … 0.35} | Did not beat fixed order (negative result) |
| Calibration (temperature, entropy map, HER, MC dropout) | Probability calibration | logits → calibrated p | fit on validation only | One split |
| EasyOCR + lexicon repair | Text reading | image → text | conf ≥ 0.4, ≥ 3 chars | Bangla CER 27.8 % |
| EfficientNet-V2-S FER | Facial expression, 7 classes | face crop → emotion | RAF-DB official split | Fear and Disgust recall ≈ 0.55–0.59 |
| Grad-CAM, solvePnP pose, FedAvg, EWC | Explainability, note pose, federated and continual demos | — | `roboeye/*.py` | Demonstrations; see Part E |

### 5.5 Model architecture

See §4.8 for the authenticator. The detector is Ultralytics YOLOv8s, 9 classes (`data/data.yaml`). The emotion model is EfficientNet-V2-S (`GPU_SPEED.md`; the weight file name `emotion_faces_efficientnet_v2s_865.pt` is in `savior_glass/models/`, which is local and not in git).

### 5.6 Training methodology

- **Detector:** `scripts/train.py` (§5.4).
- **Authenticator:** `scripts/train_qduig.py` with `configs/proposed_prefix.yaml`, then `proposed_prefix_ft.yaml` (resume). Multi-seed runs use `scripts/run_phase1_multiseed.py`. Variants use `scripts/train/train_novel.py` with `configs/v2/*.yaml`.
- **Selection rule** (`FINAL_RESULTS.md`, `CONTRIBUTING.md`): checkpoint by mean validation accuracy over 1–6 views; thresholds and calibrators fit on validation; the test split is scored once.
- **Emotion:** `savior_glass/scripts/train_emotion_rafdb.py`.
- **Ekush handwritten letters:** `savior_glass/scripts/train_ekush.py`, writer-disjoint split.

Complete hyperparameters for every detector or emotion run (optimizer, learning-rate schedule, batch size) could not be verified beyond what the scripts state.

### 5.7 Inference pipeline

See Part B.3. The glass calls the authenticator with **one view**: the YOLO crop of a live frame.

### 5.8 Hardware/software integration

The integration points are:

- `button_handler.py` (rpi-lgpio)
- `_buzz` in `currency_mode.py`, which uses the `RPi.GPIO` API; `rpi-lgpio` provides that API on the Pi 5
- `CameraManager`
- espeak-ng/Piper subprocesses

None was exercised on real Pi hardware (`GPIO_TEST.md`, `HAPTICS_TEST.md`).

### 5.9 Deployment

- **Pi:** `install.sh`, `scripts/deploy_pi5.sh` and the systemd unit `smart_glass.service`.
- **Windows:** `test_windows.py` / `run_windows_test.bat`.
- **Model export:** `scripts/export.py` (TorchScript, ONNX) and `scripts/quantize_int8.py` (INT8 ONNX; the head and first layer are kept in FP32, chosen on validation, `models/best_int8.selection.json`).
- **Phone:** a TFLite export script exists (`scripts/export/export_tflite.py`); phone latency is NOT_MEASURED.

---

## CHAPTER 6 — IMPLEMENTATION

### 6.1 Development environment

- Windows 11 laptop: AMD Ryzen 7 5800H CPU, NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB (`GPU_SPEED.md`, `edge/latency_this_machine.json`).
- Python 3.10.
- The PyTorch version differs between documents (see Consistency Audit, item 12).

### 6.2 Software implementation

About 29,000 lines of tracked Python exist outside `Unused/`, counted with `wc -l` over tracked `.py` files. The glass application is about 2,300 lines plus scripts. The rest is research, training and evaluation code.

### 6.3 Major modules

| Module | Purpose | Inputs → Outputs | Status |
|---|---|---|---|
| `savior_glass/main.py` | Pi entry point | GPIO, camera → speech | Implemented; not run on Pi |
| `savior_glass/modes/currency_mode.py` | Taka + jaal | frame → Bangla sentence, haptics | Implemented; laptop-tested |
| `savior_glass/modes/ocr_mode.py` | OCR | frame → stored text | Implemented; CER measured on synthetic text |
| `savior_glass/modes/object_mode.py` | Objects | frame → Bangla object list | Implemented; mAP50 on COCO subset |
| `savior_glass/modes/claude_mode.py` | Online description | frame → text | Implemented; not evaluated |
| `savior_glass/utils.py` | TTS, camera, language split | text → audio | Implemented; Bangla voice not tested on Windows |
| `savior_glass/ekush_letters.py` | Handwritten letters | 28×28 crop → letter | Implemented; test acc 0.900 |
| `realtime_bangla_taka_detection/scripts/realtime_detect.py` | Desktop webcam detector + SAPI | webcam → boxes, speech | Implemented; ran twice for about 5 min (`FEATURE_AUDIT_REPORT.md`) |
| `realtime_bangla_taka_detection/scripts/roboeye_live.py` | Desktop full HUD | webcam → HUD, speech, haptics | Implemented |
| `roboeye/qduig/*` | Authentication research library | views → verdict, policies, calibration | Implemented; extensively evaluated |
| `roboeye/fedavg.py`, `ewc.py`, `qduig/federated.py`, `qduig/continual.py` | Federated and continual learning | JaalTaka shards | Simulated in one process; QDW-Fed and QWER-VPC NOT_MEASURED |
| `roboeye/vlm.py` | Scene sentence | frame → sentence | Template by default; BLIP only if cached |
| `roboeye/clip_zero_shot.py` | Prototype matcher | crop → nearest prototype | Uses ImageNet MobileNet unless CLIP/DINOv2 is cached |
| `scripts/validate_claims.py` | Evidence check | claim registry → pass/fail | Implemented; exit 0 on 122 claims (`DIAGNOSIS_FULL_PROJECT.md`) |

### 6.4 Algorithms

See §5.4.

### 6.5 Database / API implementation

There is no database and no server API. The only external API is the Anthropic Messages API, called with `urllib` (`claude_mode.py`). The key is read from the environment or from a git-ignored `.env`.

### 6.6 AI/ML implementation

- 32 authentication-related model/algorithm files are in `realtime_bangla_taka_detection/models/*.py` and `roboeye/qduig/`.
- Trained weights are tracked in git: `best.pt` (22.5 MB), `best.onnx`, `best_int8.onnx` (11.5 MB), TorchScript, CNN+ViT, FedAvg, EWC, the prototype and dual-head authenticity weights, and Q-DUIG checkpoints under `results/`.
- The glass's own weights (`savior_glass/models/*.pt, *.onnx`) are git-ignored and exist only locally.

### 6.7 Hardware implementation

Only software interfaces exist (§5.8). Physical hardware was not available to the evaluation (`PI5_RESULTS.md`: "Pi 5 hardware was not attached").

### 6.8 User interface

The device has an audio-only interface with five buttons and a Bangla mode announcement. Mode names are in `config.MODE_NAMES_BN`. Error prompts are spoken, for example:

- "ক্যামেরা প্রস্তুত নয়" (camera not ready)
- "নোট সনাক্ত করা যায়নি। ক্যামেরার সামনে ধরুন।" (note not detected; hold it in front of the camera)

The Windows harness and `roboeye_live.py` provide visual HUDs for development. Accessibility has not been evaluated with users.

### 6.9 Error handling

- Camera failure at start: spoken prompt and exit (`main.py:75-80`).
- Missing models: each mode logs a warning and degrades. Currency falls back to HSV; authenticity falls back to CNN+ViT and then to "unknown".
- Non-finite probability: the verdict becomes "unknown" (`currency_mode.py:242`).
- Overlapping inference: the second press is ignored.
- Piper failure: fall back to espeak-ng.
- Missing espeak-ng: logged as an error.
- Object-loop exceptions are caught per frame.
- Haptic GPIO errors are silently ignored (`currency_mode.py:354`), so a hardware fault would not be reported.

---

## CHAPTER 7 — EXPERIMENTS AND EVALUATION

### 7.1 Experimental setup

All experiments ran on the development laptop: RTX 3050 Laptop GPU and Ryzen 7 5800H CPU, Windows 11. Some multi-seed CAMVA/baseline runs were on the laptop CPU (`WEAK_RESULTS_FIX.md`). Seeds are 42, 43 and 44 where stated.

### 7.2 Datasets

| Dataset | Location | Size (verified) | Labels | Use | Provenance |
|---|---|---|---|---|---|
| BanglaTaka "A Diverse Image Dataset for Bangladeshi Currency Recognition" | `data set/Bangladeshi_Paper_Currency_Raw` | 5,073 images (+5,073 label .txt). Per class: 2: 890, 5: 1,320, 10: 838, 20: 960, 50: 832, 100: 1,130, 200: 846, 500: 2,486, 1000: 844 files (image + txt) | denomination | compositing source | Mendeley Data 3cv2sypkkh (`ref.bib`) |
| COCO 2017 val | `data set/coco2017/val2017` | 5,000 images | — | backgrounds; object-mode test subset | COCO (`ref.bib`) |
| Synthetic composites | `data set/currency_yolo_data` | 22,333 images: train 17,839, valid 2,212, test 2,282 (incl. about 10 % negatives) | 9 classes + boxes | detector train/val/test | generated |
| JaalTaka | `data set/JaalTaka` | 1,390 notes (802 genuine, 588 counterfeit) × 6 views = 8,340 images. Split 974/208/208 notes | genuine/counterfeit | authentication | **Not documented.** `ref.bib` entry says only "Image dataset used in this work". Camera/session metadata unknown (`FINAL_RESULTS.md`: `camera_id=unknown`). License unknown. |
| Bangla Money (Kaggle) | `data set for comparison/` | 1,536 photos of 8 denominations + 101 one-taka photos | denomination | external detector test | Kaggle (no URL in repo) |
| NSTU-BDTAKA | `data set for comparison/` | detection test 186; recognition test 1,144 (88 originals overlap NSTU train) | box / denomination | external detector test | Data in Brief 2024 (`ref.bib`) |
| RAF-DB | `data set/RAF-DB` | official test 3,068 | 7 emotions | emotion | RAF-DB (license not in repo) |
| Ekush | `data set/OCR bangla data set/Ekush Data set` | test 38,608 images from 305 writers | 122 classes | handwritten letters | not documented |
| OCR phrases | generated at runtime | 80 (40 bn, 40 en) | text | OCR CER | rendered with Vrinda font (`savior_glass/scripts/publish_boost.py`) |
| ModelNet40 (10-class render subset) | `data set for comparison/` | 100 test objects | shape class | generality check of prefix training | public |

JaalTaka test set: 120 genuine and 88 counterfeit notes. A majority-class predictor therefore scores 0.5769, which is the accuracy of several collapsed runs.

### 7.3 Evaluation metrics

| Area | Metrics |
|---|---|
| Detection | mAP@0.5, mAP@0.5:0.95, precision, recall, IoU, top-1 denomination accuracy, detection rate |
| Authentication | accuracy per view count, macro-F1, balanced accuracy, ROC-AUC, PR-AUC, ECE (10 bins), Brier, NLL, McNemar, paired bootstrap CI, Cohen's h |
| OCR | CER, WER |
| Emotion | accuracy, macro-P/R/F1, per-class |
| Speed | median and P95 latency, FPS |

### 7.4 Experimental procedure

- The test split is used only for final scoring (`FINAL_AUDIT.md`, audit update 2026-09-28).
- On 2026-09-28, three evaluation bugs were found and fixed (`WEAK_RESULTS_FIX.md`):
  1. The sequential and oracle paths skipped the view self-gate.
  2. Fully confident notes produced a NaN entropy. This affected 23 of 208 test notes for PRMVT seed 42.
  3. Masked views changed the volume feature.
- 59 checkpoints were then re-evaluated. **Where pre-fix and post-fix numbers differ, this report uses the post-fix number and says so.**

### 7.5 Quantitative results

#### 7.5.1 Taka detection

| Evaluation | Data | Result | Source | Status |
|---|---|---|---|---|
| YOLO validation | composite val split (Ultralytics `val`) | mAP50 0.9939, mAP50-95 0.8440, P 0.9912, R 0.9917 | `savior_glass/results/currency_boost.json` → `yolo_val` | VERIFIED |
| Held-out composite test | 2,238 images per README | mAP50 0.995, mAP50-95 0.847, P 0.998, R 0.997 | `realtime_bangla_taka_detection/README.md`; only plots in `results/training_v2/test_eval/` | PARTIALLY VERIFIED (no metrics file; image count differs from disk, see Consistency Audit) |
| Box quality on composite test | 2,052 notes + 230 empty images | mean IoU 0.9083; IoU ≥ 0.5: 100 %; IoU ≥ 0.75: 98.39 %; class correct when detected 99.66 %; 0 false alarms | `results/bbox/bbox_eval.json` | VERIFIED |
| Multi-note composites | 2, 3 and 5 notes, 12 images each | all notes detected at IoU ≥ 0.5; mean IoU 0.946 / 0.936 / 0.937 | `results/bbox/bbox_multinote.json` via `FINAL_RESULTS.md` | VERIFIED |
| Raw BanglaTaka photos | 450 (50/class) | detection 98.4 %, top-1 96.9 % | `savior_glass/results/currency_boost.json` | VERIFIED as a number, **but not a held-out test**: the images are sampled from the compositing source folder with no split filter (`publish_boost.py:562-575`, `717`) |
| **Independent: Bangla Money** | 1,536 photos, 8 classes | **top-1 91.47 %**, no detection 3.91 %, 95.19 % correct when detected. Weakest: 1000 (78.4 %, mostly called 500) | `results/external/external_taka.json` | VERIFIED |
| Independent: NSTU detection test | 186 notes | class-agnostic AP50 0.796; P 0.832, R 0.694 at conf 0.25 | same | VERIFIED |
| Independent: NSTU hand-held close-ups | 1,144 | top-1 18.5 %; no detection 66.9 % | same | VERIFIED |
| Unknown denomination (1 taka) | 101 | 37 announced as 5 taka, 4 as 50, 3 as 2; 57 no detection | `CROSS_DATASET_TAKA.md` | VERIFIED |

#### 7.5.2 Multi-view authentication (JaalTaka test, n = 208 notes)

**Table 7.1: Accuracy by number of views (forced first-k views)**

| Views | CNN+ViT baseline (seed 42) | PRMVT seed 42, stored (pre-fix) | PRMVT seed 42, re-evaluated (post-fix) | PRMVT 3-seed mean (SD), post-fix |
|---:|---:|---:|---:|---:|
| 1 | 0.7356 (153/208) | 0.9712 | **0.9712 (202/208)** | 96.5 (1.5) |
| 2 | 0.8702 | 0.9760 | 0.9808 | 98.4 (0.6) |
| 3 | 0.9135 | 0.9760 | 0.9808 | 98.1 (0.5) |
| 4 | 0.9183 | 0.9760 | 0.9808 | 97.6 (0.8) |
| 5 | 0.8990 | 0.9808 | 0.9856 | 98.6 (0.5) |
| 6 | 0.9183 (191/208) | 0.9760 | **0.9808** | 98.1 (1.0) |

Sources:
- baseline: `results/qduig/eval/seed42/baseline/`
- PRMVT: `results/qduig/prefix_ft/seed42/test_views/` (stored) and `test_views_20260928/` (re-evaluated)
- 3-seed: `WEAK_RESULTS_FIX.md`, "fixed_order" row
- counts: `FINAL_RESULTS.md` "count check"

**Significance, seed 42, PRMVT versus baseline** (`STATISTICAL_ANALYSIS.md`; computed on the stored, pre-fix seed-42 values; the 1-view values are unchanged by the fix):

| Views | McNemar p | Bootstrap 95 % CI of accuracy difference |
|---:|---:|---|
| 1 | 4.30×10⁻¹¹ | [0.178, 0.298] |
| 6 | 0.0033 | [0.024, 0.096] |

The 6-view difference is not significant after Bonferroni correction over 36 comparisons (p = 0.118).

**Earlier Q-DUIG model versus baseline at 6 views** (`FINAL_RESULTS.md`, `tables/table02_baseline_vs_proposed.md`):

| Metric | Baseline | Q-DUIG |
|---|---:|---:|
| Accuracy | 0.9183 | 0.9663 |
| Macro-F1 | 0.9143 | 0.9655 |
| ROC-AUC | 0.9709 | 0.9891 |
| ECE | 0.0789 | 0.0303 |
| Brier | 0.0673 | 0.0230 |

McNemar p = 0.0244 (Holm 0.0489). This earlier model was **worse** than the baseline at 1–5 views; the post-fix re-evaluation gives 68.8–89.9 %.

**CAMVA over three seeds** (`WEAK_RESULTS_FIX.md`):

| Views | CAMVA | Baseline |
|---:|---:|---:|
| 1–5 | 52.1–64.6 % | 76.1–92.0 % |
| 6 | 97.1 % | 92.3 % |

CAMVA is better at 6 views in 3/3 seeds and worse at 1–5 views in 3/3 seeds.

**Adaptive view selection (negative result).**
- On the deployed PRMVT model, averaged over three seeds after the fixes, the plain fixed view order is best or tied-best at every view count.
- The learned `full_proposed` policy with adaptive stopping stops after 1.00 view at 89.3 % (4.7), against 96.5 % for the fixed first view.
- A prefix-stopping rule with λ = 0.02, chosen on validation, scores 0.9712 at 1.0096 mean views. It is effectively the 1-view classifier.
- The oracle subset search reaches 205/208 at a mean of 2.10 views, against 201/208 for the learned model in the same file. 3 notes are unsolvable by any subset (`oracle.json`, `final_pass_20260929.json`).

**Calibration, PRMVT 6 views, post-fix ECE** (`WEAK_RESULTS_FIX.md`):

| Method | ECE |
|---|---:|
| Raw confidence | 0.0146 |
| Temperature | 0.0168 |
| Entropy map | 0.0153 |
| HER | 0.0178 |
| MC dropout | 0.0136 |

Before the fix, HER looked best (0.0328 against 0.0608). After the fix, no calibrator clearly improves on raw confidence.

**Robustness, PRMVT 6-view accuracy, post-fix** (`WEAK_RESULTS_FIX.md` severity table). Clean accuracy is 98.1 %.

| Behaviour | Corruptions (accuracy) |
|---|---|
| Stable (≥ 96.6 %) | Gaussian and motion blur (all levels); JPEG 70/30/10; scale 0.85–0.5; sensor noise; contrast 0.6–2.5; brightness 0.7 |
| Moderate loss | glare 1.0: 85.1 %; perspective 0.2: 75.0 %; rotation 20°: 83.2 % |
| Severe loss | low light 0.35: 64.9 %; low light 0.2: 58.7 %; brightness 2.2: 48.6 %; occlusion 0.55: 51.9 %; rotation 40°: 62.5 % |

**Robustness fixes:**
- **Low light.** A robust fine-tune reaches 93.8 % at low light 0.2 (6 views) with clean 6-view 98.6 %.
- **Occlusion.** A three-seed occlusion ensemble reaches 0.8894 at occlusion 0.55. A rejection rule (confidence ≥ 0.99, chosen on validation) answers 46.6 % of occluded test notes with a wrong-verdict share of 0.48 % of notes (`OCCLUSION_ENSEMBLE.md`, `OCCLUSION_REJECTION.md`, `PAPER_FINAL.md`). The 90 % occlusion target was **not met**.

**Additional algorithms** (`FINAL_RESULTS.md`, `NOVEL_ALGORITHMS_COMPARISON.md`, `MULTI_SEED_RESULTS.md`, `PAPER_FINAL.md`). Fifteen further variants were trained at seed 42, some at seeds 43–44. The reported standing results are:

| Variant | Accuracy |
|---|---|
| NDAL v2 | 0.9567–0.9808 across 1–6 views |
| VAT | 0.9663–0.9760 |
| PRAVT | 0.9519–0.9760 |
| MTPT with a 6+3 epoch schedule | 0.9808 at 1 view, 0.9663 at 6 |
| VCIE, selected on 1-view validation | 0.9279 at 1 and 6 views |
| OGPD v1 | 0.9087–0.9279 (v2 regressed) |
| SFPL v2 | below baseline from 2 views |

- Three v1 runs collapsed to the majority class (0.5769) before being repaired in v2.
- The multi-seed table in `MULTI_SEED_RESULTS.md` predates the NaN fix for Q-DUIG-based models.
- The repository labels none of these as first or state of the art.

**Auxiliary-loss finding** (`ABLATION_RESULTS.md`, `AUXILIARY_FIX_RESULTS.md`). A prefix-trained encoder with the auxiliary losses switched off (`her_base`) reaches 0.9904 at 1 view, which is higher than full PRMVT. The auxiliary stack is therefore not what produces 1-view accuracy on this split.

**Theory** (`PAC-Bayes` files):
- A McAllester bound on the published weights under a zero-mean prior is vacuous (penalty 57.59).
- A separate network trained on 487 training notes has a non-vacuous bound of 0.1127 on the other 487 training notes. This does not bound the published model.

**Generality check** (`FINAL_RESULTS.md`, ModelNet40 10-class render, n = 100, seed 42):

| Training | 1 view | 2 views | 3 views | 6 views |
|---|---:|---:|---:|---:|
| Fixed-view | 0.64 | 0.70 | 0.73 | 0.77 |
| Prefix | 0.64 | 0.73 | 0.77 | 0.77 |

#### 7.5.3 Assistive modules

| Module | Data | Result | Source | Status |
|---|---|---|---|---|
| OCR (EasyOCR bn+en + CLAHE) | 80 synthetic rendered phrases | CER 0.155, WER 0.342. English: CER 0.032, WER 0.104. Bangla: CER 0.278, WER 0.579 | `paper_evidence/ocr/ocr_cer.json` | VERIFIED; synthetic images only |
| Handwritten Ekush CNN | 38,608 test images, writer-disjoint | acc 0.900 (conjuncts 0.877, digits 0.943) | `savior_glass/results/ekush_letters.json` | VERIFIED |
| Emotion EfficientNet-V2-S | RAF-DB official test, 3,068 | acc 0.8654, macro-F1 0.795. Recall: Fear 0.554, Disgust 0.588 | `paper_evidence/emotion/emotion_rafdb.json` | VERIFIED |
| Emotion (earlier FER+ pipeline) | 5,741 test faces | acc 0.579, macro-F1 0.493 | `savior_glass/results/assistive_eval_core.json` | VERIFIED; superseded model |
| Object YOLOv8s (COCO-pretrained, not fine-tuned) | 250 random COCO val2017 images, 77 classes scored | mAP50 0.604 (project's own AP computation) | `paper_evidence/object/object_coco.json` | VERIFIED; not the official COCO evaluator |

### 7.6 Qualitative results

Existing figures include:

- sample detections, Grad-CAM, a webcam detection and the compositing process (`docs/figures/ieee/fig2, fig7, fig8, fig9`)
- the confusion matrix and training curves (`docs/papers/currency_detection/`)
- failure cases (`paper_evidence/figures/fig11_failures.png`)

In a simulated-GPIO run, one real 100-taka photograph produced the spoken output "একশত টাকার নোট। আসল" ("100 taka note. Genuine") 989 ms after the button press (`GPIO_TEST.md`). This is a single example, not an accuracy estimate.

### 7.7 Performance evaluation

**Table 7.2: Latency on the development laptop** (`GPU_SPEED.md`, `results/speed/gpu_speed.json`; 50 runs, median). None of these are Raspberry Pi numbers.

| Model | GPU (RTX 3050) | CPU (Ryzen 7 5800H) |
|---|---:|---:|
| Taka YOLOv8s, PyTorch (incl. pre/post-processing) | 19.1 ms (52 FPS) | 105.2 ms |
| Taka YOLOv8s, FP32 ONNX | 24.0 ms | 92.5 ms |
| Taka YOLOv8s, INT8 ONNX | 37.1 ms | 60.8 ms |
| PRMVT, 1 view | 15.5 ms | 10.4 ms |
| PRMVT, 6 views | 16.3 ms | 21.3 ms |
| Emotion EfficientNet-V2-S (face + flip) | 29.4 ms | 83.0 ms |

**Live camera loop** (`LIVE_CAMERA_TEST.md`; 20.1 s, 150 frames, dark room, no note in view):

| Stage | Median |
|---|---:|
| Whole loop | 7.5 FPS (128.9 ms per frame) |
| Camera capture | 73.6 ms |
| Taka detection + jaal check | 31.1 ms |
| Object mode | 23.8 ms |

**Not measured:** Raspberry Pi 5 latency, FPS, temperature and throttling; energy; phone latency.

### 7.8 Comparison with baselines

- **Authentication.** The CNN+ViT baseline was retrained on the same note-disjoint split and evaluated with the same first-k view protocol (§7.5.2). This is a verified in-project baseline.
- **Detection.** Comparison with published methods on NSTU-BDTAKA is not reported. The project's detector was evaluated on NSTU without training on it. **No verified comparison with published Taka recognisers was found in the repository.**

### 7.9 Ablation study

The prefix-protocol ablations are in `ABLATION_RESULTS.md`, with post-fix values in `WEAK_RESULTS_FIX.md`. After the fix, most prefix-trained variants reach 95–99 % at every view count. The early fixed-recipe ablations in `results/qduig/ablations/` remain weak (56–95 %). `WEAK_RESULTS_FIX.md` states they "must not be presented as working methods".

---

## CHAPTER 8 — RESULTS AND DISCUSSION

### 8.1 Results

The strongest verified results are:

1. **Prefix-robust authentication.** It removes the few-view accuracy drop on JaalTaka. The 1-view gain over the baseline is +23.6 points at seed 42 (202 against 153 of 208 notes).
2. **Independent-photo detection.** The Taka detector generalises to independently collected full-note photographs (Bangla Money: 91.5 %). It does **not** generalise to hand-held close-ups (NSTU: 18.5 %).
3. **Integrated application.** The glass software integrates these models and runs end-to-end on a laptop. Its button logic passes a simulated-GPIO test.

### 8.2 Analysis

- **Synthetic versus real.** Composite-test scores (mAP50 ≈ 0.99, 100 % IoU ≥ 0.5) are much higher than independent-data scores, as expected: composites are statistically similar to training data. The 450-photo "raw" figure (96.9 %) comes from the same source images as training, so it measures fit to the source domain, not generalisation.
- **Close-up failure.** NSTU close-ups show partially covered, folded notes filling the frame. The detector was trained on whole notes at scale 0.25–0.90 of the canvas. An NSTU-trained classifier scored 99.1 % on NSTU test but only 48.2 % on Bangla Money (`CROSS_DATASET_TAKA.md`). In-domain scores therefore overstate real-world performance.
- **View selection.** Useful 2-view subsets exist: the oracle reaches 205/208 at a mean of 2.10 views. No learned policy found them better than taking views in the fixed capture order.
- **Evaluation bugs.** The NaN-entropy bug changed conclusions: several variants reported as "degrading with more views" did not degrade after the fix. This shows the value of the project's re-evaluation practice. It also means older write-ups must be read with their correction notes.

### 8.3 Interpretation

Within this dataset, the main scientific finding is simple. **Training with random view prefixes is sufficient for view-count robustness.** The more elaborate components (quality, diversity, information gain, gated fusion) are not needed for 1-view accuracy, as the `her_base` result shows. This is a narrower result than the original Q-DUIG framing (quality-, diversity-, uncertainty- and information-gain-aware acquisition). The repository's own audit records H1 and H4 as failed.

### 8.4 Strengths

- Note-disjoint split with zero measured leakage. Validation-only selection. Multi-seed results for the main model.
- Negative results are reported: failed policies, collapsed variants, occlusion target not met, vacuous bounds.
- Every headline number traces to a JSON file; `validate_claims.py` exits 0 on 122 claims.
- External detector evaluation on two public datasets, including an MD5 check that excluded a dataset identical to the training source.
- The glass degrades gracefully when models are missing.

### 8.5 Limitations

See Part F.

### 8.6 Reliability and reproducibility

**Present in the repository:**
- configs, seeds, splits (`test_note_ids.json`) and per-run `environment.txt` / `git_commit.txt`
- trained checkpoints, tracked in git
- step-by-step commands (`paper_evidence/REPRODUCIBILITY.md`)

**Missing:**
- JaalTaka, BanglaTaka composites, RAF-DB and Ekush are not distributed. JaalTaka's source and license are undocumented.
- The documented PyTorch versions conflict (Consistency Audit, item 12).
- The glass's own model files are git-ignored.
- There is no automated unit-test suite and no continuous integration.

**Assessment:** a developer with access to the local datasets could reproduce the authentication tables with the provided commands. An outside researcher could not reproduce them without JaalTaka.

### 8.7 Practical implications

- The glass announces a genuine/counterfeit verdict from **one live-frame crop** at a 0.5 threshold, with no abstain option.
- The authenticator was validated on JaalTaka close-up views, not on YOLO crops from the glass camera.
- The occlusion rejection rule developed in the research code is **not** used in the glass path.
- A wrong "genuine" verdict on a counterfeit note has direct financial consequences for the user.

Until live-crop accuracy is measured, the verdict should be treated as experimental.

---

## CHAPTER 9 — CONCLUSION AND FUTURE WORK

### 9.1 Conclusion

The project set out to build an offline, Bangla-first smart glass that reads text, announces objects and recognises Bangladeshi Taka with a counterfeit check. The software for all of these functions is implemented and has been exercised on a laptop.

The Taka detector, trained only on synthetic composites, identifies 91.5 % of independently collected full-note photographs. It fails on hand-held close-ups. A prefix-robust multi-view authenticator achieves 97.1 % single-view and about 98 % six-view accuracy on a note-disjoint JaalTaka test set. Learned view-selection policies did not improve on a fixed order. The authenticator remains sensitive to heavy occlusion, strong low light and over-exposure.

Raspberry Pi 5 performance, physical hardware behaviour, live-note accuracy and user acceptance have not been measured. The work therefore demonstrates a software prototype and a controlled authentication study, not a validated assistive device.

### 9.2 Contributions

See Part G.

### 9.3 Limitations

See Part F.

### 9.4 Future work, in priority order

1. **Pi 5 measurement.** Run `savior_glass/scripts/benchmark_pi5.py`, including the 30-minute sustained run.
2. **Live-crop authenticity.** Collect glass-camera crops of known genuine and counterfeit notes. Measure accuracy, then add the validated rejection rule to `CurrencyMode`.
3. **User study.** Run the protocol in `paper_evidence/user_study/study_protocol.md` with visually impaired participants.
4. **Detector generalisation.** Add hand-held close-ups and partial notes to training. Add an "unknown note" rejection. Re-test on newly collected photographs before adopting the P2 fallback (`CROSS_DATASET_TAKA.md`).
5. **JaalTaka provenance.** Document its source, capture devices, sessions and license. Test camera- or session-disjoint splits.
6. **Bangla OCR.** Improve it (CER 27.8 %) and evaluate on real photographed signs rather than rendered text.
7. **Hardware documentation.** Add the BOM, wiring, power and enclosure, and test the buttons and haptics on real switches.
8. **Engineering hygiene.** Add a unit-test suite and continuous integration. Resolve the documentation inconsistencies listed in the Consistency Audit.

---

## REFERENCES

References are limited to those present in the repository (`Thesis Report and paper/thesis report/My report/ref.bib`, 67 entries; `docs/RELATED_WORK_BIBLIOGRAPHY.md`; arXiv identifiers in `paper_evidence/NOVELTY_DECLARATION.md`). The entries below are the ones this report relies on. Details are reproduced as they appear in the repository files; several `ref.bib` entries lack authors or years and should be completed before submission.

1. BanglaTaka: "A Diverse Image Dataset for Bangladeshi Currency Recognition", Mendeley Data, https://data.mendeley.com/datasets/3cv2sypkkh/1
2. "NSTU-BDTAKA: An open dataset for Bangladeshi paper currency detection and recognition", *Data in Brief*, 2024. https://www.sciencedirect.com/science/article/pii/S2352340924006681
3. T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context" (`ref.bib` key `coco`).
4. "JaalTaka: Genuine and Counterfeit Bangladeshi Banknote Image Collection (six views per note)", image dataset used in this work. **Incomplete citation; source not documented.**
5. J. Lee et al., "Set Transformer", ICML 2019, arXiv:1810.00825.
6. T.-Y. Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017, arXiv:1708.02002.
7. D. Lopez-Paz et al., "Unifying distillation and privileged information", arXiv:1511.03643.
8. M. Federici et al., "Learning Robust Representations via Multi-View Information Bottleneck", arXiv:2002.07017.
9. "Blindness and Vision Impairment: Fact Sheet", WHO (`ref.bib`).
10. "Advancements in Smart Wearable Mobility Aids for Visual Impairments: A Bibliometric Narrative Review", 2024 (`ref.bib`).

The remaining assistive-system references (FingerReader, Vision Voice, AIris, Divya-Drishti and others) are listed in `ref.bib` and `docs/RELATED_WORK_BIBLIOGRAPHY.md`.

## APPENDICES

**Appendix A: Important configuration**
- `savior_glass/config.py`: pins, thresholds, TTS settings, timing
- `realtime_bangla_taka_detection/data/data.yaml`
- `results/qduig/prefix_ft/seed42/config.yaml` (quoted in §5.4)

**Appendix B: Dataset structure**
- JaalTaka: `real_notes/note_XXX/note_XXX_{1..6}.jpg` and `fake_notes/note_XXX/…`
- Composites: YOLO format, `{train,valid,test}/{images,labels}`

**Appendix C: Test scripts**

| Script | What it checks |
|---|---|
| `savior_glass/scripts/test_buttons_haptics.py` | buttons and haptics on simulated GPIO |
| `savior_glass/scripts/test_speech.py` | speech routing |
| `savior_glass/scripts/test_live_camera.py` | live camera loop |
| `realtime_bangla_taka_detection/scripts/test_roboeye_modules.py` | smoke test of every module |
| `realtime_bangla_taka_detection/scripts/validate_claims.py` | evidence check |
| `realtime_bangla_taka_detection/scripts/random_test_check.py` | random test-split spot check |

**Appendix D: Installation**
- Root `README.md` (Quick start)
- `savior_glass/install.sh`
- `paper_evidence/REPRODUCIBILITY.md`

**Appendix E: User manual (buttons)**

| Button | Function |
|---|---|
| MODE | next mode |
| ACTION | capture / announce |
| READ | speak stored OCR text |
| VOL+ / VOL− | ±10 % volume |

Windows harness keys: see `savior_glass/test_windows.py`. Desktop detector: SPACE to speak, q to quit.

---

# PART D — TECHNICAL INVENTORY

### D.1 Technologies

| Technology | Version (if stated) | Purpose | Where used |
|---|---|---|---|
| Python | ≥ 3.10 | all code | everywhere |
| PyTorch / torchvision | ≥ 2.1 (requirements); conflicting pins, see audit | training and inference | both code bases |
| Ultralytics YOLOv8 | ≥ 8.3 | detector training and inference | `train.py`, `currency_mode.py`, `object_mode.py` |
| ONNX / ONNX Runtime | ≥ 1.15 / ≥ 1.17 | export, INT8 inference | `export.py`, `quantize_int8.py` |
| OpenCV | ≥ 4.8 | capture, preprocessing, solvePnP, Haar/YuNet faces | modes, `roboeye/pose.py` |
| EasyOCR | ≥ 1.7.1 | Bangla/English OCR | `ocr_mode.py` |
| Tesseract (optional) | — | OCR fallback | `ocr_repair.py` |
| espeak-ng, Piper | — | offline TTS on Pi | `utils.TTSEngine` |
| Windows SAPI / pyttsx3 | — | desktop TTS | `roboeye/speaker.py`, `test_windows.py` |
| rpi-lgpio | ≥ 0.5 | Pi 5 GPIO | `button_handler.py` |
| SpeechRecognition | ≥ 3.10 | desktop voice commands | `roboeye/asr.py` |
| scikit-learn, SciPy, pandas, matplotlib | see requirements | metrics, statistics, figures | evaluation scripts |
| Anthropic Messages API | model from `CLAUDE_MODEL` env, default `claude-sonnet-4-5` | optional online mode | `claude_mode.py` |
| systemd | — | autostart on Pi | `smart_glass.service` |

### D.2 Models

| Model | File | Tracked in git | Evidence |
|---|---|---|---|
| YOLOv8s Taka (9 classes) | `realtime_bangla_taka_detection/models/best.pt` (22.5 MB), `.onnx`, `_int8.onnx` (11.5 MB), `.torchscript` | yes | §7.5.1 |
| PRMVT authenticator | `results/qduig/prefix_ft/seed{42,43,44}/checkpoint.pt` | yes (seed 42 confirmed) | §7.5.2 |
| CNN+ViT, FedAvg, EWC, prototype and dual-head authenticity | `models/authenticity_*.pt`, `dual_head_auth.pt` | yes | legacy; early CNN+ViT printed 1.00 on a same-collection split, which `FEATURE_AUDIT_REPORT.md` says must not be reported |
| FER+ ONNX, YuNet face detector | `models/emotion-ferplus-8.onnx`, `face_detection_yunet_2023mar.onnx` | yes | desktop emotion |
| EfficientNet-V2-S emotion, Ekush CNN, MobileNetV3 currency, YOLOv8s/n COCO | `savior_glass/models/` | **no** (git-ignored) | §7.5.3 |

### D.3 Hardware

| Item | Specification in repo |
|---|---|
| Compute | Raspberry Pi 5 (target; not tested) |
| Camera | Pi camera (libcamera) or USB; 640×480 at 30 fps |
| Input | 5 push-buttons, BCM 17/27/24/22/23, 250 ms debounce |
| Output | speaker (device not specified); vibration motor on BCM 13 |
| Power, enclosure, frame | Not available in the current project repository |

### D.4 APIs / interfaces

- There is no REST or server API.
- The external API is the Anthropic Messages API (online mode only).
- Internal interface: `BaseMode.process_frame(frame) -> Optional[str]`.

### D.5 Configuration files

- `savior_glass/config.py`
- `realtime_bangla_taka_detection/configs/*.yaml`: 35 top-level, 20 in `v2/`, 20 in `ablation_prefix/`, 11 in `aux_fixes/`
- `data/data.yaml`
- `smart_glass.service`

---

# PART E — RESULTS

### E.1 Verified quantitative results

These are the tables in §7.5 marked VERIFIED. The headlines are:

| Result | Value |
|---|---|
| Bangla Money external top-1 | 91.47 % |
| NSTU close-up top-1 | 18.5 % |
| PRMVT 1-view accuracy, seed 42 | 97.12 % (202/208) |
| PRMVT 6-view accuracy | 98.08 % post-fix at seed 42; 98.1 ± 1.0 % over 3 seeds |
| Baseline 1-view / 6-view | 73.56 % / 91.83 % |
| Emotion, RAF-DB | 86.5 % |
| OCR CER (English / Bangla) | 3.2 % / 27.8 % |
| Ekush handwritten letters | 90.0 % |
| Laptop GPU/CPU latencies | Table 7.2 |

### E.2 Demonstrated functionality (works, no quantitative evaluation)

- Mode cycling, Bangla announcements, volume and debounce (simulated GPIO)
- the OCR capture/read split
- Piper→espeak fallback (code)
- the dark-frame retry
- the haptic patterns (simulated)
- Claude online mode
- the Grad-CAM live overlay
- solvePnP "straighten" prompt
- the voice-command intent parser
- the template VLM sentence
- simulated FedAvg/EWC

### E.3 Unverified claims found in repository documents

| Claim | Where | Problem |
|---|---|---|
| 135 FPS / 7.3 ms on RTX 4060 | `realtime_bangla_taka_detection/README.md` | No artifact; the measured GPU is an RTX 3050 (19.1 ms) |
| Test set of 2,238 images with P 0.998 / R 0.997 | same README | No metrics file; the disk test split has 2,282 images |
| "Detects 98.4 % of 450 raw phone photographs … 96.9 % top-1" presented as evaluation | thesis abstract | Not held out; drawn from the training source folder |
| Earlier draft figures 96.7 % / 98.1 % / 96.5 % authenticity, Pi 15 FPS, CLIP 96.5 %, Qwen-VL | older IEEE draft (`FEATURE_AUDIT_REPORT.md`) | Explicitly never measured |
| "Fully Offline" | `savior_glass/main.py` docstring | The mode cycle includes an online Claude mode |

### E.4 Missing experiments

- Raspberry Pi 5 latency, FPS, thermal behaviour and 30-minute run
- energy consumption
- live-note detection and authenticity accuracy from the glass camera
- camera- or session-disjoint authentication
- the user study (protocol ready, no data)
- real-hardware button and haptic tests
- Bangla TTS intelligibility
- a full-pipeline run with the network disabled
- phone/TFLite latency
- seeds 45–46
- security-feature labels (SFAQ)
- comparison with published Taka recognisers

---

# PART F — LIMITATIONS

### F.1 Confirmed limitations (evidence in repository)

1. The detector fails on hand-held close-ups (18.5 %, NSTU) and has no unknown-note class (1-taka notes are announced as 5 taka 37 % of the time).
2. The authenticator degrades under heavy occlusion (51.9 % at 0.55), strong low light (58.7 % at 0.2) and over-exposure (48.6 % at brightness 2.2). The occlusion ensemble reaches 88.9 %, below the 90 % target.
3. Learned view selection and adaptive stopping do not beat a fixed order (H1 and H4 failed).
4. Bangla OCR is weak: CER 27.8 %, WER 57.9 %, even on rendered text.
5. The emotion model's recall is low for Fear (0.554) and Disgust (0.588).
6. The PAC-Bayes bound on the published model is vacuous.
7. The glass applies the authenticator to a single live-frame crop with no rejection option.

### F.2 Missing validation

These items are NOT MEASURED: Pi 5 operation, real GPIO/haptics, live-note accuracy, the user study, energy, and offline operation with the network disabled.

### F.3 Technical risks

- **Domain shift** between JaalTaka close-ups and glass-camera crops, for the authenticity verdict.
- **Pi 5 compute budget.** EasyOCR (about 600 MB), YOLOv8s and the authenticator on CPU. INT8 is the fastest CPU format measured on the laptop.
- **Silent haptic failure.** Exceptions are swallowed in `_buzz`.
- **Unsafe model loading.** `torch.load(..., weights_only=False)` for the MobileNet classifier would execute pickled code from an untrusted weights file.
- **Privacy.** Claude mode sends camera frames, possibly containing faces or documents, to an external API.

### F.4 Research limitations

- One authentication dataset, with unknown provenance, license and capture conditions. All notes come from one collection.
- A single note-disjoint split, with three training seeds at most.
- JaalTaka lacks security-feature labels, so feature-level claims cannot be tested.
- Novelty was assessed by web-index search, not a systematic review.
- The ModelNet check is small (n = 100) and shows no difference at 1 or 6 views.

---

# PART G — RESEARCH CONTRIBUTION

Based on the available repository evidence:

1. **System integration.** The implementation contributes a complete, offline-first software stack for a Bangla assistive glass. It combines OCR, object announcement, Taka denomination recognition and a counterfeit verdict with speech and haptic output. This is an application and integration contribution. It has not yet been validated on hardware or with users.
2. **Synthetic-to-real Taka detection, honestly evaluated.** The project demonstrates that a detector trained only on copy-paste composites reaches 91.5 % on an independent full-note photo set, and quantifies where it fails (close-ups, unknown notes). It also shows that in-domain test scores overstate generalisation (NSTU-trained classifier: 99.1 % in-domain, 48.2 % external).
3. **Prefix-robust multi-view authentication.** The project demonstrates, on JaalTaka, that training over random view prefixes removes the drop from 6-view to 1-view accuracy (73.6 % → 97.1 % at 1 view). It also shows that the more complex acquisition machinery (quality, diversity, information gain, learned stopping) adds no measured benefit. The repository labels this approach as extending existing missing-view training. **Novelty cannot be conclusively established without comparison against the broader literature.**
4. **Negative and corrective results.** Failed view-selection policies, collapsed variants, vacuous bounds, and documented evaluation-bug corrections are reported alongside positive results. The claim registry links each result to a file.

---

# PART H — RECOMMENDED FIGURES AND TABLES

### Figures (existing files where available)

| # | Figure | Existing file |
|---|---|---|
| 1 | Overall system architecture (glass + models) | `paper_evidence/figures/fig01_roboeye_architecture.png`, `fig12_assistive_pipeline.png`. Should be redrawn to match Part B. |
| 2 | Hardware block diagram / wiring | **missing**; to be created from the §3.4 pin map |
| 3 | Photograph of the assembled glass | **missing** |
| 4 | Compositing pipeline | `docs/figures/ieee/fig2_compositing.png` |
| 5 | Detector training curves and confusion matrix | `docs/papers/currency_detection/fig_training_curves.png`, `fig_confusion_matrix.png` |
| 6 | Sample detections / webcam | `docs/figures/ieee/fig7_sample_detections.png`, `fig9_webcam_detection.png` |
| 7 | Grad-CAM | `docs/figures/ieee/fig8_gradcam.png` |
| 8 | JaalTaka six views of one note | `Thesis Report and paper/…/DIagram fig tab/jaal_real_v1…v6.jpg` |
| 9 | Authenticator architecture | `paper_evidence/figures/fig02_qduig_architecture.png` |
| 10 | Accuracy vs number of views (baseline vs PRMVT) | `paper_evidence/figures/fig4_accuracy_vs_views.png` (check that it uses post-fix numbers) |
| 11 | Multi-seed accuracy | `fig11_multiseed.png` (regenerate with post-fix values) |
| 12 | Calibration / reliability | `fig6_calibration.png` (regenerate post-fix) |
| 13 | Robustness severity curves | `fig8_robustness.png` |
| 14 | Failure cases | `fig11_failures.png` |
| 15 | View-selection policy comparison | `fig7_view_selection.png` (shows a collapsed policy; `FINAL_AUDIT.md` notes this) |
| 16 | External-dataset detection results | **missing** |
| 17 | Emotion confusion matrix | `savior_glass/results/emotion_confusion_matrix.png` |
| 18 | Pi 5 latency | `fig10_pi5_latency.png` exists, but **no Pi data exists**; do not use it until measured |

### Tables

| # | Table | Source |
|---|---|---|
| 1 | Technology stack | D.1 |
| 2 | Functional / non-functional requirements | §3.1–3.2 |
| 3 | Datasets | §7.2; `paper_evidence/tables/table01_dataset.md` |
| 4 | Detector configuration | `scripts/train.py` |
| 5 | Authenticator configuration | `config.yaml` |
| 6 | Hardware components and pins | §3.4 (+ BOM when available) |
| 7 | Detection results: synthetic vs external | §7.5.1 |
| 8 | Authentication accuracy 1–6 views | Table 7.1 |
| 9 | Significance tests | `STATISTICAL_ANALYSIS.md` |
| 10 | Calibration | §7.5.2 |
| 11 | Robustness | `WEAK_RESULTS_FIX.md` |
| 12 | Latency | Table 7.2 |
| 13 | Assistive module results | §7.5.3 |
| 14 | Test cases (GPIO, speech, live camera) | `GPIO_TEST.md`, `SPEECH_TEST.md`, `LIVE_CAMERA_TEST.md` |
| 15 | Limitations and NOT_MEASURED items | `paper_evidence/tables/table10_limitations.tex` |

---

# PART I — CLAIM VERIFICATION

### I.1 Implementation traceability

| Report claim | Evidence / file | Status |
|---|---|---|
| Four modes cycled by one button | `savior_glass/main.py:45,131-141`, `config.py` | Verified (code; simulated test) |
| Currency mode uses YOLOv8s first, HSV only as fallback | `currency_mode.py:103-143` | Verified (code) |
| Jaal verdict from PRMVT on one crop, threshold 0.5 | `currency_mode.py:195-248` | Verified (code) |
| OCR capture/read split, CLAHE, lexicon repair | `ocr_mode.py`, `main.py:177-194` | Verified (code) |
| Piper → espeak-ng fallback | `utils.py:227-295` | Verified (code) |
| 22,333 composite images, source-image split 80/10/10 | disk counts; `generate_synthetic_dataset.py:27,303-316` | Verified |
| Detector val mAP50 0.9939 / mAP50-95 0.8440 | `savior_glass/results/currency_boost.json` | Verified |
| Detector test mAP50 0.995 / 0.847 | README; plots only | Partially verified |
| Bangla Money 91.47 % | `results/external/external_taka.json` | Verified |
| NSTU close-ups 18.5 % | same | Verified |
| JaalTaka 1,390 notes, 974/208/208, zero leakage | `results/camva/splits/split_metadata.json` | Verified |
| PRMVT 1-view 0.9712, 6-view 0.9808 (post-fix) | `results/qduig/prefix_ft/seed42/test_views_20260928/` | Verified |
| Baseline 1-view 0.7356, 6-view 0.9183 | `results/qduig/eval/seed42/baseline/` | Verified |
| Learned policies worse than fixed order | `WEAK_RESULTS_FIX.md` | Verified |
| Occlusion ensemble 0.8894; rejection 46.6 % answered, 0.48 % wrong | `results/robustness/occlusion_decision.json` via `PAPER_FINAL.md` | Verified |
| Emotion RAF-DB 0.8654 | `paper_evidence/emotion/emotion_rafdb.json` | Verified |
| OCR CER 0.155 | `paper_evidence/ocr/ocr_cer.json` | Verified (synthetic data) |
| Laptop latencies | `results/speed/gpu_speed.json` via `GPU_SPEED.md` | Verified |
| Pi 5 performance | `PI5_RESULTS.md` | NOT MEASURED |
| User study | `user_study/statistical_report.json` | NOT MEASURED (protocol ready) |
| Real GPIO / haptics | `GPIO_TEST.md`, `HAPTICS_TEST.md` | Simulated only |
| QDW-Fed, QWER-VPC | `FINAL_RESULTS.md` | NOT MEASURED |
| Security-feature detection (SFAQ) | `NOVELTY_DECLARATION.md` | Not implemented (no labels) |
| Bone-conduction audio, VR headset | `FEATURE_AUDIT_REPORT.md` | Not implemented |

### I.2 Claim verification report (major claims in this report)

| Claim | Evidence | Status |
|---|---|---|
| The glass works offline | `offline_validation.json`: modules offline; full network-off run not measured; Claude mode online | PARTIALLY VERIFIED |
| Prefix training removes the few-view drop | Table 7.1, 3 seeds | VERIFIED (on JaalTaka only) |
| Auxiliary components are unnecessary for 1-view accuracy | `her_base` 0.9904 | VERIFIED (seed 42) |
| Detector generalises to independent full-note photos | Bangla Money 91.5 % | VERIFIED |
| Detector generalises to real use | Only full-note photos and NSTU close-ups tested; no glass-camera test | UNVERIFIED |
| Authenticity verdict is reliable on the glass | No live-crop evaluation | UNVERIFIED |
| System suits visually impaired users | No user study | UNVERIFIED |
| Pi 5 real-time performance | No measurement | UNVERIFIED |
| Novelty of PRMVT | Repository labels it EXTENDS | INFERRED (not established) |
| The research focus moved to authentication | Commit and file history | INFERRED |

---

# PART J — REPORT COMPLETENESS SCORECARD

| Category | Status | Evidence / missing information |
|---|---|---|
| Architecture | **Complete** | Code fully inspected; Part B |
| Implementation | **Complete (software)** / **Missing (hardware)** | All modes implemented; no BOM, wiring or Pi run |
| Dataset | **Partial** | Counts and splits verified; JaalTaka provenance and license missing; Kaggle URL missing |
| Methodology | **Complete** | Scripts and configs present |
| Testing | **Partial** | Scripted functional tests and simulated GPIO; no unit-test suite, no hardware tests |
| Results | **Strong for authentication and detection; weak for OCR** | OCR on synthetic text only; live-note accuracy missing |
| Performance | **Partial** | Laptop only; Pi 5 missing |
| Literature | **Partial** | 67 references, but no systematic review and several incomplete citations |
| Reproducibility | **Partial** | Commands, seeds and checkpoints present; datasets not distributed; environment pins conflict |
| Limitations | **Complete** | Extensively documented in `paper_evidence/` |
| Research contribution | **Partial** | Clear measured finding; novelty not established |

---

# CONSISTENCY AUDIT

The following contradictions were found. They are reported, not resolved. Where the code or a saved artifact settles the question, that is stated.

| # | Statement A | Statement B | What the evidence shows |
|---|---|---|---|
| 1 | Root `README.md`: Object mode is "YOLOv8n" | `config.py`, `object_mode.py`: `yolov8s.pt` first, `yolov8n.pt` as fallback | Code uses YOLOv8s when present |
| 2 | `savior_glass/README.md`, `docs/00_OVERVIEW.md`, and the `currency_mode.py` module docstring: currency = HSV + optional MobileNetV3 | `currency_mode.process_frame`: YOLOv8s "Stage 0" plus authenticity; HSV only if YOLO is absent | Code uses YOLOv8s. Documents are outdated. |
| 3 | `main.py` docstring: "Fully Offline" | `MODE_CLAUDE` is in the MODE button cycle; it calls an internet API | The fourth mode is online |
| 4 | Detector README: test set 2,238 images | Disk: `currency_yolo_data/test` has 2,282 images | Cannot be resolved from files; the dataset was probably regenerated |
| 5 | Thesis abstract: 22,315-image dataset | `data.yaml` comment and disk: 22,333 | `data.yaml` explains that the 22,315 draft used a different negative draw |
| 6 | README/thesis: backgrounds are 1,500 COCO images (`download_backgrounds.py`) | `generate_synthetic_dataset.py`: `BG_DIR = …\coco2017\val2017`, 5,000 on disk | Which set produced the current composites cannot be verified |
| 7 | Thesis abstract presents the 450 raw photographs as an evaluation | `publish_boost.py` samples them from the compositing source folder with no split filter; `CROSS_DATASET_TAKA.md` confirms that folder is the training source | Not a held-out test |
| 8 | Detector README: 7.3 ms (135 FPS) on RTX 4060 | `GPU_SPEED.md`: RTX 3050, 19.1 ms; `edge/latency_this_machine.json`: RTX 4060 "NOT MEASURED" | The 135 FPS claim has no artifact |
| 9 | `PAPER_FINAL.md` (2026-09-29): PRMVT 6-view 0.9760 | `test_views_20260928/`: 0.9808 after the NaN fix; thesis: 98.1 % | 0.9760 is the pre-fix stored value |
| 10 | `MULTI_SEED_RESULTS.md`: PRMVT 6-view mean 0.9792 | `WEAK_RESULTS_FIX.md`: 98.1 (post-fix) | `MULTI_SEED_RESULTS.md` predates the fix |
| 11 | `STATISTICAL_ANALYSIS.md`: "McNemar for PRMVT vs baseline at 6 views remains … p = 0.0244" | `FINAL_RESULTS.md`: p = 0.0244 belongs to the earlier Q-DUIG model (0.9663); the same `STATISTICAL_ANALYSIS.md` table gives PRMVT 6-view p = 0.0033 | 0.0244 is misattributed |
| 12 | `REPRODUCIBILITY.md`: torch 1.12.1+cu113 on Python 3.10 | `requirements.txt`: torch 2.14.0 + cu126; `edge/latency_this_machine.json`: 2.6.0+cu124 | Three different documented environments |
| 13 | `FEATURE_AUDIT_REPORT.md` (10 Sep): full composite set not on disk; INT8 is 11.0 MB | Disk now has 22,333 images; `fps_benchmark.json` gives 11.53 MB | The audit is dated; the newer state holds |
| 14 | `fps_benchmark.json`: CPU 179 ms (PT), 199 ms (INT8) | `GPU_SPEED.md`: CPU 105 ms (PT), 60.8 ms (INT8) | Different runs and runtimes; the INT8 ranking reverses. Cite the newer file with its date. |
| 15 | `FINAL_RESULTS.md`: occlusion 88.5 % and 0.5 % | `occlusion_decision.json`: 0.8894 and 0.0048 | Same experiment, rounded (noted in `DIAGNOSIS_FULL_PROJECT.md`) |
| 16 | `FINAL_AUDIT.md` header: MULTI-SEED and ABLATIONS NOT_MEASURED | Later sections of the same file: PASS | The header is superseded by later updates |
| 17 | `docs/00_OVERVIEW.md` listed other repository links | Root README: `aflam330/...` | Resolved: all documents now point to the `aflam330` repository |
| 18 | `FINAL_RESULTS.md` (2026-09-28): calibration "HER lowest ECE" | `WEAK_RESULTS_FIX.md` post-fix: MC dropout and raw confidence lower than HER | Post-fix supersedes |

---

# INFORMATION REQUIRED TO COMPLETE THE REPORT

In priority order:

1. **Raspberry Pi 5 benchmark results.** Run `benchmark_pi5.py`, including the sustained run.
2. **Live authenticity and detection accuracy from the glass camera.** Use known genuine and counterfeit notes.
3. **User-study data** (protocol ready), or an explicit statement that the study is out of scope.
4. **JaalTaka provenance.** Creator, capture devices, sessions, license, and how the counterfeit notes were obtained.
5. **Hardware documentation.** BOM, wiring diagram, power supply and battery life, enclosure, and photographs of the prototype.
6. **Author names, supervisor, institution and acknowledgements.**
7. **A detector test-set metrics file** matching the reported mAP. Alternatively, re-run `scripts/evaluate.py` and save its JSON, and resolve the 2,238 versus 2,282 image count.
8. **Screenshots and a demo video** of the glass in use.
9. **Complete bibliographic data** for `ref.bib` entries (authors, venues, years). Add the Kaggle Bangla Money URL.
10. **Real-photograph Bangla OCR evaluation.**
11. **A single pinned environment** (lock file) for reproducibility.
12. **A decision on the online Claude mode**: keep it in the device mode cycle, or document it as optional, with a privacy note.
