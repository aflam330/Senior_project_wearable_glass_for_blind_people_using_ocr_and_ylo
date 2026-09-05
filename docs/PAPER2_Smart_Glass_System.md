# Paper 2 Documentation — Offline Smart Glass for Blind People (Full System)

**Primary source:** `e:\Final SP\savior_glass`  
**Duplicate clone:** `e:\Final SP\smart-glass` (identical code; use one name in the paper)  
**GitHub:** https://github.com/Mukut313/savior_glass (or smart-glass)  
**Initial commit theme:** “Offline Smart Glass for Blind People (RPi 5)”

In-repo READMEs are stubs — this file is the documentation base for the system paper.

---

## 1. Working title options

1. Savior Glass: An Offline Bangla-Capable Smart Glass for Visually Impaired Users on Raspberry Pi 5  
2. A Fully Offline Multimodal Assistive Smart Glass with Bangla OCR, Object Detection, and BDT Currency Recognition  
3. Edge-Deployed Assistive Vision Glasses for Bangladesh: Design, Implementation, and Accessibility UX

**Product / system name (choose one for consistency):** Savior Glass **or** Smart Glass

---

## 2. Draft abstract (system paper)

Visually impaired users in Bangladesh often lack affordable assistive devices that work offline, speak Bangla, and recognise local currency. We present an offline smart-glass prototype built on Raspberry Pi 5 that provides three complementary modes — Bangla/English text reading (OCR), continuous everyday object announcement (YOLOv8n), and Bangladeshi Taka note recognition — controlled by five physical GPIO buttons and spoken through Piper neural TTS with espeak-ng fallback. The system requires no cloud connectivity after installation, uses a Pi Camera (or USB webcam), and ships as a systemd service for boot-time operation. OCR uses EasyOCR with an accessibility-oriented capture/read split; object detection announces Bangla COCO class names with per-class cooldowns; currency recognition combines HSV colour profiling with an optional MobileNetV3-Small classifier. A Windows test harness supports development without hardware. We describe the hardware interface, software architecture, Bangla-first design choices, and deployment pipeline, and discuss limitations and a path to integrate a high-accuracy YOLOv8 currency detector.

**Keywords:** Assistive Technology, Smart Glasses, Raspberry Pi, Bangla OCR, Object Detection, Currency Recognition, Offline TTS, Accessibility, Edge AI

---

## 3. Problem statement & target users

| Aspect | Detail |
|--------|--------|
| Users | Blind / visually impaired people, Bangla-speaking context |
| Barriers addressed | Cloud dependency, English-centric commercial tools, lack of BDT support, phone-screen UX |
| Design goals | Fully offline; physical buttons; auditory-only feedback; low-cost DIY/edge hardware |
| Locale | Bangladesh — Bangla mode names, Bangla object labels, BDT denominations |

**Boot UX:** speaks “স্মার্ট গ্লাস চালু হচ্ছে”, then the active mode name in Bangla.

---

## 4. Contributions (system paper)

1. **Fully offline** Bangla + English assistive vision stack on **Raspberry Pi 5**.  
2. **Three-mode** wearable UX with dedicated physical controls (mode / action / read / volume ±).  
3. **OCR capture ≠ playback** — ACTION stores text; READ speaks it (steadies camera, then listen hands-free).  
4. **Mixed-script TTS segmentation** (Bangla Unicode blocks vs English) for correct voice routing.  
5. **BDT-aware currency mode** with HSV graceful degradation when CNN weights are absent.  
6. **Edge packaging:** install script, systemd service, Windows development harness.

---

## 5. System architecture

```
[5 GPIO buttons] → ButtonHandler (rpi-lgpio)
                         ↓
                   SmartGlass (main.py)
          ┌──────────────┼──────────────────┐
          ↓              ↓                  ↓
   CameraManager    Mode plugins      TTSEngine
   (OpenCV thread)  OCR / Object /     (queue + worker)
          ↓         Currency                 ↓
     frame buffer   EasyOCR / YOLO /    Piper → aplay
                    HSV (+MobileNet)    else espeak-ng
```

**Threads:** main, camera grabber, TTS worker, object-detection loop, ACTION inference (OCR/currency), GPIO ISR threads.

**Concurrency:** `_inferring` Event prevents overlapping heavy OCR/currency jobs while object scanning continues on its interval.

---

## 6. Hardware

| Component | Specification / assumption |
|-----------|----------------------------|
| Compute | Raspberry Pi 5 (CPU-only ML; `gpu=False`) |
| Camera | Pi Camera 3 (IMX708 overlay) or USB (`CAMERA_INDEX`) |
| Resolution | 640×480 @ 30 FPS |
| Buttons | 5× push-buttons to GND, BCM 17/27/24/22/23 |
| Debounce | 250 ms |
| Audio | ALSA (`aplay`, `amixer`); headphones/speaker |
| User | systemd runs as `pi` with groups `gpio audio video` |

### Button map

| BCM pin | Role | Behaviour |
|---------|------|-----------|
| 17 | MODE | Cycle OCR → Object → Currency |
| 27 | ACTION | OCR capture / object force-announce / currency detect |
| 24 | READ | Speak stored OCR text |
| 22 | VOL_UP | +10% |
| 23 | VOL_DOWN | −10% |

---

## 7. Software modules

| Path | Responsibility |
|------|----------------|
| `main.py` — `SmartGlass` | Lifecycle, mode cycle, routing, detection loop, signals |
| `config.py` | GPIO, camera, models, thresholds, TTS, logging |
| `button_handler.py` — `ButtonHandler` | GPIO interrupts + debounce |
| `utils.py` | Logging, language detect/split, `TTSEngine`, `CameraManager` |
| `modes/base_mode.py` | Mode interface |
| `modes/ocr_mode.py` — `OCRMode` | Bangla/EN OCR + store/read |
| `modes/object_mode.py` — `ObjectMode` | YOLO + cooldown + Bangla labels |
| `modes/currency_mode.py` — `CurrencyMode` | HSV ± MobileNet |
| `scripts/train_currency.py` | Train MobileNetV3-Small for BDT |
| `assets/labels_bn.json` | COCO English → Bangla (80 classes) |
| `install.sh` | Apt, venv, Piper EN, YOLO, camera overlay, systemd |
| `smart_glass.service` | Boot autostart |
| `test_windows.py` / `run_windows_test.bat` | Desktop webcam + keyboard harness |

---

## 8. Modes of operation

| Index | Mode | Bangla announcement |
|------:|------|---------------------|
| 0 | OCR | টেক্সট রিডিং মোড |
| 1 | Object | অবজেক্ট ডিটেকশন মোড |
| 2 | Currency | কারেন্সি ডিটেকশন মোড |

### 8.1 OCR mode (`OCRMode`)

- **Lazy-loads** `easyocr.Reader(["bn", "en"], gpu=False)` (~600 MB RAM noted in code).  
- **ACTION:** capture → preprocess → OCR → filter → **store** → short confirmation (“text saved, press READ”).  
- **READ:** speak stored text with prefix “পড়া হচ্ছে:” / “Reading:”.  
- Preprocess: upscale ~1200 px width → grayscale → bilateralFilter(9,75,75) → CLAHE(clip=2.5, 8×8).  
- **No binarization** (comments: measured conf ~0.84 vs ~0.69 with thresholding).  
- Post: reading-order sort; conf ≥ 0.4; min 3 chars; text-likeness filter.

### 8.2 Object mode (`ObjectMode`)

- Continuous scan every **1.5 s**.  
- YOLOv8n on **320×240**, conf ≥ **0.50**.  
- Top-3 unique classes; per-class cooldown **4.0 s**.  
- Announcement: `"সামনে আছে: " + Bangla names`.  
- ACTION: force announce (bypass cooldown on next scan).

### 8.3 Currency mode (`CurrencyMode`)

- **ACTION-triggered** only (CPU contention management).  
- **Stage 1 — HSV:** note-like contours (aspect 1.5–2.8, area ≥ 8000) + colour profile match (accept ≥ 0.30).  
- **Stage 2 — CNN (optional):** MobileNetV3-Small if `models/currency_mobilenet.pt` exists; override if softmax ≥ **0.65**.  
- Denominations in glass code: **10, 20, 50, 100, 200, 500, 1000** Taka (7 classes) with Bangla spoken names.  
  - *Note vs Paper 1:* YOLO currency work covers **9** classes including **2** and **5** Taka. Mention this difference if integrating.

---

## 9. Models & assets

| Asset | Path / config | Role |
|-------|---------------|------|
| YOLOv8n | `models/yolov8n.pt` | COCO object detection |
| Optional NCNN | `yolov8n_ncnn_model` | ~40% faster ARM (mentioned in install) |
| Currency MobileNet | `models/currency_mobilenet.pt` | Optional 7-class BDT |
| Bangla labels | `assets/labels_bn.json` | 80 COCO → Bangla |
| Piper EN | `models/piper/en_US-amy-low.onnx` | Neural TTS |
| Piper BN | `models/piper/bn_BD-medium.onnx` | Optional (not auto-downloaded) |
| Piper binary | `/usr/local/bin/piper` | Runtime |
| EasyOCR | runtime cache | bn + en |
| espeak-ng | system | Fallback (`bn` / `en-us`) |

**Repo status:** model weights are gitignored (placeholders only). Document download/train steps for reproducibility.

### Key thresholds (`config.py`)

| Parameter | Value |
|-----------|-------|
| OCR_CONFIDENCE | 0.4 |
| OCR_MIN_CHARS | 3 |
| OBJECT_CONFIDENCE | 0.50 |
| CURRENCY_CONFIDENCE | 0.65 |
| OBJECT_SCAN_INTERVAL | 1.5 s |
| OBJECT_ANNOUNCE_COOLDOWN | 4.0 s |
| Default volume | 80% |
| TTS queue max | 5 |

---

## 10. TTS pipeline (Bangla-first)

1. Detect language: if >25% of chars are Bangla Unicode (U+0980–U+09FF) → `bn`, else `en`.  
2. Split mixed-script segments and sentences for natural pauses (`TTS_INTER_UTTERANCE_PAUSE = 0.15`).  
3. Prefer Piper if binary + voice ONNX exist; else espeak-ng.  
4. Piper PCM: 22050 Hz, S16_LE, mono → `aplay`.  
5. Mode switch calls `stop_current()` to clear queued speech.

---

## 11. Deployment

### Install (`install.sh` — 7 steps)
1. System packages (libcamera, espeak-ng, alsa-utils, OpenBLAS, …)  
2. Python venv `~/smart-glass-env`  
3. Pip packages from `requirements.txt`  
4. Piper ARM64 + English Amy-low voice  
5. Copy / fetch YOLOv8n  
6. Camera overlay (`dtoverlay=imx708`)  
7. Enable `smart_glass.service`

### Service
```
ExecStart=/home/pi/smart-glass-env/bin/python3 /home/pi/smart-glass/main.py
Restart=on-failure
After=sound.target
```

### Windows development
- `run_windows_test.bat` / `test_windows.py`  
- Keys: M / A / R / + / − / Q  
- TTS: Piper or Windows SAPI (SAPI cannot speak Bangla well)

### Currency CNN training
```bash
python3 scripts/train_currency.py --data-dir /path/to/taka_images
```
- ImageFolder layout: `10/`, `20/`, … `1000/`  
- Suggested ≥500 images/class (1000+ best)  
- Defaults: 25 epochs, batch 8, lr 1e-3, freeze backbone, Adam, 80/20 split, seed 42

---

## 12. Related work

**Full annotated list (your spreadsheet):** see [`RELATED_WORK_BIBLIOGRAPHY.md`](RELATED_WORK_BIBLIOGRAPHY.md)  
**Source:** `Review and paper links.xlsx` (52 usable papers tagged for smart-glass / OCR / object / TTS).

### 12.1 High-priority from your review sheet

| Sheet # | Paper | Cites (sheet) | Use for |
|--------:|-------|--------------:|---------|
| 2 | Smart Glass System Using Deep Learning for the Blind and Visually Impaired (MDPI Electronics) | 77 | Closest smart-glass system comparator |
| 55 | A Google Glass Based Real-Time Scene Analysis for the Visually Impaired | 39 | Wearable scene analysis |
| 49 | Deep Learning Reader for Visually Impaired | 34 | OCR/reader assistive baseline |
| 3 | Design and Implementation of Smart Glass with Voice Detection… | 16 | Hardware smart-glass design |
| 11 | Cloud based Text extraction using Google Cloud Vision… | 16 | Contrast: cloud OCR vs your offline EasyOCR |
| 32 | AI Based Real-Time Blind Assistance System with Voice Feedback | 11 | Multi-function voice assistive |
| 12 / 14 / 40 | Raspberry Pi reading / obstacle / OCR systems | — | Edge/RPi positioning |
| 23 | Bangla OCR using deep learning | — | Bangla OCR related work |
| 10 | NSTU-BDTAKA BDT dataset | — | Currency mode related work / Paper 1 bridge |

### 12.2 Extra categories to expand

| Category | Examples to cite |
|----------|------------------|
| Commercial assistive | OrCam, Envision Glasses, Seeing AI (contrast: cloud/phone, English-centric, cost) |
| Bangla OCR | Sheet #23 + EasyOCR literature |
| Currency for VI | Paper 1 YOLO work; sheet currency papers; MobileNet/CNN BDT papers |
| Offline TTS | Piper, espeak-ng; sheet TTS/speech papers (#24, #25, …) |

**Differentiation table:**

| Typical systems | This system |
|-----------------|-------------|
| Cloud / phone companion | Fully offline on Pi |
| English-centric | Bangla announcements + bn OCR + BDT |
| Touch / voice / app UI | Physical GPIO buttons |
| Rich LLM scene description | Fixed modes: OCR / COCO YOLO / currency |
| Continuous OCR streaming | OCR & currency on-demand; object periodic |
| Proprietary form factor | DIY RPi 5 + Camera Module 3 class prototype |

---

## 13. Evaluation plan (needed for a strong paper)

The codebase has **no published accuracy tables or user study**. Recommended experiments before submission:

| Experiment | Metric |
|------------|--------|
| OCR on Bangla/English printed samples | CER / WER; latency on Pi 5 |
| Object mode indoors | Precision of announcements; false-alarm rate |
| Currency mode vs lighting | Accuracy per denomination; HSV-only vs MobileNet vs YOLO (Paper 1) |
| End-to-end UX | Task completion time; NASA-TLX or SUS with VI participants |
| Power / thermal | Battery life estimate; CPU load per mode |
| Ablations | Without CLAHE; with/without capture-read split; Piper vs espeak |

---

## 14. Limitations (Discussion)

1. README / academic docs were missing in-repo (now covered here).  
2. Currency CNN optional and lighting-sensitive HSV stage.  
3. Glass currency is **7-class MobileNet/HSV**, not yet the **9-class YOLOv8s** from Paper 1.  
4. Object detector is generic COCO — not navigation/obstacle-specific.  
5. EasyOCR is heavy and slow on Pi — OCR not continuous by design.  
6. Piper Bangla not installed by default.  
7. Voice commands deferred (Vosk commented — no Bangla model).  
8. No formal evaluation yet in the repository.

---

## 15. Future work

- Integrate Paper 1 `best.pt` into `CurrencyMode` (replace or cascade with HSV/MobileNet).  
- Continuous soft audio cues for currency/object.  
- Custom Bangla scene / obstacle classes.  
- Voice commands when Bangla ASR is available.  
- NCNN/TFLite/INT8 for lower latency.  
- Formal user study with visually impaired participants.  
- Mechanical enclosure / spectacle form-factor BOM.

---

## 16. Relationship of the two glass repos

| Check | Result |
|-------|--------|
| Source files | Identical (23 tracked files) |
| `smart-glass` HEAD | `8bcd7f2` |
| `savior_glass` HEAD | `8d49720` (README-only delta on top of same history) |
| Unique features | **None** |

**Recommendation for publication:** cite a single repository URL and product name; mention the other as a mirror if needed.

---

## 17. Link to Paper 1

Paper 1 provides the **high-accuracy BDT detector** and synthetic-data methodology. Paper 2 provides the **end-to-end assistive system**. Together they form a coherent thesis narrative:

> Data engineering → robust currency detector → offline wearable that also reads text and announces objects in Bangla.

If the YOLO model is not yet wired into the glass at submission time, state that clearly and present MobileNet/HSV as the current deployed currency path.
