# SARA: Smart Assistive Reading Aid — full project information

This document is for writing a journal or conference paper about SARA. It gathers in one place what the project is, what was built, which data was used, how each method works, every measured result and its conditions, what failed, what was corrected, and what is still unmeasured.

**Sources and rules:**
- Every number is taken from the 124-page thesis (`documents/SARA_Thesis_Report.pdf`) or from the dated evidence note named next to it (`evidence_notes/`). The thesis copies each value from a saved result file (`results_data/`).
- The thesis is the reference wherever an older note disagrees.
- `02_RESULTS_MASTER_TABLE.md` lists every result with its test set, conditions and status.

---

## 1. The project in brief

| | |
|---|---|
| Name | SARA (Smart Assistive Reading Aid); earlier internal names *Savior Glass*, *RoboEye* |
| What it is | A wearable smart glass for blind and low-vision people in Bangladesh that speaks Bangla, reads Bangla and English, names objects, recognises Bangladeshi Taka notes and screens them for counterfeits without ever accusing a note, announces facial expressions, and offers an optional online scene description |
| Where it runs | On a Raspberry Pi 5 (8 GB) on the glass; everything except the online description works without the internet |
| Team | MD Abdun Fattah Lam (2220014), Mukut Protim Memo (2020086), Tanjila Hossain Tapti (2130561) |
| Supervisor | Mohammad Shidujaman, PhD, Assistant Professor |
| Institution | Department of Computer Science and Engineering, Independent University, Bangladesh (IUB); CSE Senior Project |
| Period | 2026; thesis consolidated October 2026 |
| Code (glass, public) | https://github.com/aflam330/SARA |
| Code and experiments (research) | https://github.com/aflam330/Senior_project_wearable_glass_for_blind_people_using_ocr_and_ylo (branch `RPi5_deploy`) |
| Status | Prototype version 2 built and worn; engineering tests done part by part; first field test informal; trials with a visually impaired participant started; no completed user study |

## 2. Problem and motivation

A blind person in Bangladesh usually has to ask someone for three everyday things:
- **Money:** which note was handed back, and whether it is real.
- **Reading:** what a medicine label, sign or document says.
- **Surroundings:** what or who is in front of them.

The answer comes from another person and cannot be checked.

**Global context:** WHO estimates at least 2.2 billion people with a near or distance vision impairment, most of them in low- and middle-income regions (thesis ref. `who2023vision`).

**Gaps in existing help:**
- **Language:** OrCam MyEye, the best-known commercial reading glass, has no Bangla language support and no Taka detection, and is too expensive for most users in Bangladesh (thesis literature review).
- **Internet:** the research glasses reviewed (Google Glass scene analysis `googleglass2021`, AIris `airis2024`) send pictures to a server, so without a network there is no answer.
- **Banknotes:** none of the reviewed wearables names Taka notes.
- **Counterfeit research:** existing Bangladeshi counterfeit work (`uddin2016counterfeit`, `pathak2024counterfeit`, `chowdhury2022uvink`, the JaalTaka dataset `jaaltaka`/`piyas2026jaaltaka`) is not in a wearable. None of it checks whether test counterfeits are copies of prints seen in training.

**Why safety matters:** for a blind user a wrong verdict is the worst error. A genuine note called counterfeit causes an accusation; a counterfeit called genuine loses money.

## 3. Research questions (thesis wording, shortened)

1. **System:** can a multi-function Bangla glass run on a Raspberry Pi 5, and how fast?
2. **Synthetic data:** can a Taka detector trained without hand labels work on real, hand-held photographs?
3. **Number of views:** why do multi-view counterfeit models fail with fewer views, and can training fix it?
4. **Unseen prints:** how much does accuracy drop on counterfeit prints never seen in training?
5. **Safety:** can the glass give counterfeit advice that never accuses a genuine note?
6. **Bangla OCR:** which part of a Bangla OCR pipeline matters most for a wearable camera?
7. **Interaction:** which interface choices make the glass usable without sight? (Design, not yet measured with users.)

## 4. Hardware

| Part | Details | Role |
|---|---|---|
| Raspberry Pi 5 with active cooler | 4-core 2.4 GHz, 8 GB; no accelerator used | Runs all models and the program |
| Camera Module 2 | Sony IMX219, 8 MP, fixed-focus; used at 640×480 | Pictures of text, notes, surroundings |
| 5 push-buttons | On a perforated board along the temple; GPIO 17, 27, 24, 22, 23: MODE, ACTION, READ, VOLUME UP, VOLUME DOWN | Control without a screen |
| Vibration motor | GPIO 13 | Short buzz when something is found (for example a note) |
| Bone-conduction headphone | Open-ear | Speech while the ears stay free |
| Power bank | USB-C (official Pi 5 supply is 5 V, 5 A) | Portable power; battery life not measured |
| Spectacle frame + UV lamp | 3D-printed frame with camera housing and a small UV-A lamp, 360–395 nm (seen as blue-violet light); head strap | Holds the camera where the user looks; lamp for the UV fibre check |
| Rear enclosure | 3D-printed, ventilated, at the back of the head | Carries the Pi 5, balances the frame, keeps heat from the face |

**Prototypes:**
- **Version 1:** board and fan on one temple.
- **Version 2:** camera housing and UV lamp at the front, five buttons along the side, bone-conduction headphone, and the Pi 5 in a rear enclosure.
- **Photos:** `figures/prototype_and_test_photos/` and `figures/thesis/sara_v2_collage.jpg`.

**UV background (for the paper):**
- **Bands:** UV-A 315–400 nm, UV-B 280–315 nm, UV-C 100–280 nm (WHO 2002, `who2002uvi`).
- **365 nm:** the most widely used fluorescent security features are excited at 365 nm (Mutanen et al. 2003, `mutanen2003luminescent`).
- **Paper:** banknote paper does not glow while its fibres glow red, blue and green (ECB 2010, `ecb2010quickguide`). Ordinary paper contains optical brighteners that glow bluish (`mutanen2003luminescent`, patent `thierauf2005patent`).
- **Caveat:** a genuine note washed in detergent can glow too.

## 5. Software and modes

**Language and platform:**
- Python on Raspberry Pi OS (Debian 13). The app is `savior_glass/main.py`, run as a systemd service (`smart_glass.service`).
- Models run with PyTorch or ONNX Runtime (INT8 where it keeps accuracy).

**The five modes:**

| Mode | Models | Offline? |
|---|---|---|
| Text reading (Bangla + English) | CRAFT text detector (ONNX) + EasyOCR recogniser, **blended** with a version fine-tuned on real words; main-text-block selection; CLAHE; clean-up rules; read width 1,200 px; line-by-line speech for pages | Yes |
| Objects | YOLOv8s (COCO, 80 classes), automatic scan every 1.5 s (thermal governor slows it when hot), at most 3 objects per scan, confidence 0.50 | Yes |
| Taka notes | Taka YOLOv8s trained on synthetic composites and fine-tuned on synthetic hand-held close-ups; announce thresholds 0.75 (laptop) / 0.80 (Pi INT8); count and total of several notes; safe counterfeit verdict (500 and 1,000 Taka); guided watermark check; optional UV fibre check (off by default) | Yes |
| Emotion | YuNet face detector + EfficientNet-V2-S (RAF-DB) | Yes |
| Online description | Cloud vision model through an API key kept in a git-ignored `.env` | **No** (the only online mode) |

**Interaction:**
- **Buttons:** MODE cycles the modes, ACTION captures, READ reads, plus two volume buttons.
- **Voice commands:** say "glass" first, for example "glass, check money" or "glass, read text". Recognition uses Vosk small English with a closed grammar, unknown words stripped at the edges, minimum confidence 0.5, and is deaf while the glass speaks. A noise filter, level control and VAD end-of-speech are available but off by default.
- **Speech output:** each sentence is rendered fully, then played at its voice's own sample rate with 150 ms lead-in. Engines: Piper (Bangla and English) → Piper binary → espeak-ng.
- **Feedback:**
  - prefetch of the next sentence
  - a progress word "পড়ছি" and a soft tick during long captures
  - spoken capture guidance
  - mixed-script speech
  - queue clearing on mode change
  - cooldowns
  - vibration for a found note
- **Thermal governor:** checks the temperature every 1 s and the firmware throttle flags every 5 s.
  - The object scan slows to every 3 s above 72 °C and every 6 s above 77 °C, and pauses above 81 °C.
  - The live preview stops above 77 °C.
  - Button presses and voice commands are never delayed.
- **Field log and power logger:** `power_log.py` reads `vcgencmd pmic_read_adc` every 2 s. Battery life has not been measured yet.

**Spoken currency verdict (exact Bangla, from `modes/currency_mode.py`):**
- note names such as "পাঁচশত টাকার নোট" (500 Taka note)
- "সম্ভবত আসল" (likely genuine) only above the safe threshold
- "আসল কিনা হাতে যাচাই করুন" (check by hand whether it is genuine) otherwise
- **"জাল" (counterfeit) is never spoken.**

## 6. Datasets

| Dataset | Content | Use | Split |
|---|---|---|---|
| BanglaTaka (Mendeley, `banglataka`) | 5,073 background-free photos of all 9 notes (2: 445, 5: 660, 10: 419, 20: 480, 50: 416, 100: 565, 200: 423, 500: 1,243, 1000: 422) | Foregrounds for compositing | 80/10/10 by physical note |
| COCO val2017 (`coco`) | Everyday photos; 1,500 downloaded for backgrounds; 5,000 for the object test | Backgrounds; object test | — |
| Synthetic Taka set (this work) | 22,333 composites (20,292 with a note, 2,041 background-only), automatic YOLO labels | Detector training/testing | 17,839 / 2,212 / 2,282 |
| Fresh synthetic test (this work) | 566 composites, test notes on unseen backgrounds | Second synthetic test | Test only |
| Synthetic hand-held close-ups (this work) | 6,000, from training-split notes only: warped, any rotation, folded in 40%, filling/overflowing the frame, 1–3 finger shapes, blur, low resolution, colour/JPEG changes | Fine-tuning the detector | Training |
| Bangla Money (Kaggle) | 1,536 photos of 8 notes (no 200 Taka) + 101 photos of 1 Taka (unknown class) | Independent test | Test only |
| NSTU-BDTAKA (`nstubdtaka2024`) | Detection part: 186 photos with boxes ("Taka", no value). Recognition part: 256×256 hand-held close-ups of 9 values, with the dataset's own splits | Independent test; selection of the hand-held detector | First evaluation: 1,144 test close-ups. Hand-held study: 1,611 validation photos (one per original) and 1,127 test close-ups. The notes do not explain the difference between 1,144 and 1,127; check `results_data/taka_and_counterfeit/closeup_ft/` before quoting both |
| JaalTaka (`jaaltaka`, `piyas2026jaaltaka`) | 1,390 genuine (802) and counterfeit (588) 500/1,000 Taka notes, 6 ordered views each (8,340 photos); genuine notes from government banks, counterfeits seized by the Rapid Action Battalion | Counterfeit check | Note-disjoint 974 / 208 / 208 (split seed 42; leakage 0); print-disjoint 946 / 222 / 222 (test: 121 genuine, 101 counterfeit, about 24 prints) |
| Bangladeshi Counterfeit Currency Image Dataset | Whole-note photos: 1,199 genuine, 87 counterfeit (about 4 physical counterfeit notes) | Safety check of the deployed verdict | Test only |
| MVP-N (`mvpn`), ModelNet (`modelnet`) | Multi-view objects | Does the view-count finding hold outside currency? | Own splits |
| RAF-DB (`rafdb`) | 15,339 faces, 7 expressions | Emotion | Official 12,271 / 3,068 |
| OCR benchmark (this work) | Bangla and English phrases rendered with correct Bangla shaping (HarfBuzz), some placed in photos; 432 images per set | OCR selection and test | Different phrases and fonts for validation and test |
| ICDAR 2019 MLT (`mlt2019`) | Real scene-text word images | Real-photo OCR test; real-word fine-tuning | Training words; validation 150 + 150; test 500 Bangla + 500 Latin |
| Ekush (`ekush`) | 367,018 handwritten Bangla characters, 122 classes | Fall-back handwritten-letter reader | By writer 2,429 / 303 / 305 |
| UV note photos (this work) | 118 phone photos of genuine 200/500/1,000 Taka under a UV lamp | UV fibre check | Blocked 5-fold by capture order |
| ESC-50 (`esc50`) | Environmental sounds | Noise for voice-command tests | — |
| Synthetic voices | 20 neural voices: 8 for design, 12 held out (4 Bangla voices reading English commands); 39 command phrases + 30 ordinary sentences each | Voice-command test | Design vs held-out |

**Leakage checks:**
- An image-level JaalTaka split gave accuracy 1.00. That was leakage and is not a result.
- A third public Taka collection is byte-identical to BanglaTaka, so it was excluded.
- The 450 "raw" BanglaTaka photos come from the training source, so they are a source-domain check, not a test.

## 7. Methods

### 7.1 Synthetic training data for Taka detection

Copy-paste compositing (thesis Algorithm "Copy-paste compositing with automatic labels"):
- **Pasting:** each source note is pasted onto a random crop of a COCO photo (640×640), four composites per note.
- **Transforms:**
  - perspective tilt (70% of pictures, corners moved up to 8%)
  - rotation ±30°
  - scale so the note fills 25–90% of the picture
  - HSV jitter (hue ±18°, saturation ×0.5–1.5, brightness ×0.4–1.6) and contrast ×0.7–1.3
  - motion blur (20%), Gaussian noise (50%), JPEG quality 55–92
  - 10% background-only pictures
- **Labels:** come from the paste geometry: x_c=(x+w/2)/S, y_c=(y+h/2)/S, b_w=w/S, b_h=h/S. No manual annotation.
- **Detector:** YOLOv8s from COCO weights, 80 epochs, 640 px, batch 16, SGD (lr 0.01, momentum 0.937, weight decay 0.0005, 3 warm-up epochs), mosaic 1.0, mixup 0.15, no flips (a mirrored note is not a real note).

### 7.2 Hand-held fine-tuning

- **Data and training:** 6,000 synthetic close-ups (Section 6), 12 epochs on the original training pictures plus the close-ups.
- **Choices fixed on validation before any test photo was read:**
  - **Epoch:** most correct minus wrong answers on 1,611 NSTU validation close-ups, among epochs keeping synthetic validation mAP@0.5 within 0.005 of the deployed detector. Chosen: epoch 8.
  - **Announce threshold:** open-set rule; 0.75 here (0.60 for the earlier detector).
  - **Pi version:** INT8 calibration (scene calibration) with threshold 0.80.
- **Test:** read once.

### 7.3 Prefix training for multi-view counterfeit detection (view-count collapse)

- **Problem:** JaalTaka gives 6 ordered views per note. A fusion network trained always with 6 views collapses when a user captures fewer, and unpredictably across seeds.
- **Prefix training:** during training, keep a random prefix of 1–6 views per note. This is the whole fix.
- **Isolating the cause:** the same network was trained four ways (fixed or prefix sampling × shared or per-view-count batch norm, 3 seeds each). The simplest variant, prefix + shared BN, is the most accurate and was chosen on validation.
- **PRMVT** (prefix-robust multi-view transformer) is the earlier full model with per-view-count BN and auxiliary losses.
- **Numerical fault fixed:** the entropy of a fully confident prediction (0·log 0) produced NaN and a probability of 0.5. It affected 23 of 208 six-view test notes; all figures are after the fix (6-view 97.6% → 98.1%).

### 7.4 Serial-number audit and a print-disjoint split

- **Audit:** serial numbers were read by OCR (Bangla digits mapped to 0–9, first 6 digits compared).
  - 279 of 322 readable 500 Taka counterfeits carry the same serial.
  - Genuine notes have almost one serial each.
  - On the usual note-disjoint test, 68 of 87 counterfeits have a serial that occurs among training counterfeits.
  - A rule "counterfeit if the serial occurs among training counterfeits" scores 90.2%. This is a shortcut that affects every result published on this split; the data article does not mention it.
- **Print-disjoint split:** no test counterfeit shares its serial with a training or validation counterfeit. Sizes 946 / 222 / 222; test has 121 genuine and 101 counterfeit notes from about 24 prints, so there are about 24 independent units.

### 7.5 Watermark check with guided capture and a learned localizer

- **Capture guide:** brightness ≥ 73.87 (mean grey) and sharpness ≥ 106.75 (Laplacian variance), each the 2nd percentile of validation back-lit photos. It gives a spoken reason at most every 2.5 s and gives up after 10 s.
- **Localizer:** a MobileNetV3-Small regresses the 4 watermark-window corners from a 320×320 image. Labels come from SIFT+RANSAC registration. Non-convex or implausible quadrilaterals are rejected. Registration remains a fall-back.
- **Classifier:** MobileNetV2, INT8 (2.6 MB).
- **Hybrid** (research): p_hyb = σ(β0 + β1·ℓ_net + β2·ℓ_wm + β3·m). Here ℓ is the log-odds, m = 1 when no window was found (then p_wm = 0.5), and β is fitted on validation only.

### 7.6 DMWR — denomination-matched watermark residual (new method)

Physics: a back-lit note gives I ≈ L·t, with L the lamp and t the paper transmittance. The watermark lives in t.

1. **Remove the lamp (quotient image):** Q̃ = I / (G_σ * I), σ = 16 px, on a 128×128 crop; Q = (Q̃ − mean)/std.
2. **Genuine portrait per note value:** T_d = pixel-wise median of Q over **genuine training** notes of value d (235 × 500 Taka, 260 × 1,000 Taka). No counterfeit is needed, and validation and test notes are never used.
3. **Residual:** R = Q − T_d.
4. **Four numbers:** s = [ρ(Q, T_d), std(R), Var(∇²R), mean|Q − G_2 * Q|], standardised on training notes.
5. **Decision:** two small CNN streams (3 stride-2 convolutions, 16/32/64 channels) read Q and the standardised R. Their features and ŝ are joined by one linear layer; softmax ≥ 0.5 means "watermark clear".
6. **Size and training:** 47,818 parameters (0.21 MB). Trained 12 epochs with AdamW (lr 1e-3) and ±5° / ±3 px augmentation; epoch chosen by validation AUC; threshold 0.5.

### 7.7 Safe verdict policy (selective prediction with a conformal guarantee)

- **Input:** the detector's note box is resized to width 640 and turned to landscape. Four windows are cut at the positions of JaalTaka views 1–4 (measured on 200 training notes with SIFT+RANSAC), and the authenticator returns p.
- **Rule:**
  - If p > τ, the glass says "likely genuine".
  - Otherwise it says "check by hand whether it is genuine".
  - It never says "counterfeit".
- **Threshold:** τ = 0.99959, the highest p of any of the n = 88 validation counterfeits.
- **Guarantee:** for an exchangeable new counterfeit, P(p_new > τ) = 1/(n+1) = 1.1%. Under a change of camera or print, the bound weakens by at most the total-variation distance between the two distributions.
- **Scope:** 500 and 1,000 Taka only. For other notes the glass says the check was not done.
- **Quality gate:** an image-quality gate runs before any verdict.

### 7.8 UV fibre check (ENFR: excitation-normalised fluorescence residual)

1. **Lamp light as reference:** the lamp's light reaches the camera as blue. For one note's paper, a = median(G/B) and b = median(R/B) are measured in the same picture.
2. **Fluorescence:** E = max(G − aB, R − bB) is the light the lamp does not explain. Then r = tophat(E) / (G_6 * B + 1), using a white top-hat with a 9-px disc.
3. **Fibres:** connected parts of r > 0.12, 5–70 px long and thin, in a crop of width 600.
4. **Colour condition:** each fibre pixel must satisfy max(G − aB, R − bB)/B > c with c = 0.22, and lie on lit paper (B > ½ median B).
5. **Decision:** accept if at least T = 1 fibre is found.
6. **Properties:** lamp brightness and distance cancel out. The camera's colour balance does not, so T and the r threshold belong to one camera.
7. **Deployment:** off by default, with cautious wording.

### 7.9 Bangla/English text reading pipeline

- **Steps,** each chosen on validation (thesis Table "ocr_steps"):
  - order the words into lines
  - drop unsure words
  - keep only the dominant text block (the largest gain)
  - CLAHE
  - tuned EasyOCR settings (best of 31)
  - clean-up rules
- **Not adopted:** binarisation, deskew, Tesseract, TrOCR, PaddleOCR, dictionaries, KenLM, ByT5, multi-scale, rotation retry.
- **Real-word recogniser:**
  - EasyOCR's recogniser was fine-tuned on ICDAR 2019 MLT training words mixed with rendered words (3 seeds).
  - The full fine-tune failed the 0.5-point rendered-text rule, so the weights were blended: θ = ½θ_released + ½θ_fine-tuned. This rule was fixed after the first model failed and before any other was scored.
  - Seed 44 was chosen on real validation. The weights ship as fp16 parts.
- **Speed:** the CRAFT detector runs in ONNX; read width stays 1,200 px.

### 7.10 Voice commands

- **Evaluated:** 138 design combinations of front end (Wiener filter, level control), early partial decisions, Silero VAD endpointing, confidence threshold, wake word and edge-unknown rule, compared on 8 design voices.
- **Chosen:** wake word "glass" + edge rule, no front end.
- **Test:** read once on 12 held-out voices.
- **Conditions:**
  - clean
  - soft (−20 dB)
  - simulated whisper (the voice's spectral envelope excited by noise)
  - whisper with people talking nearby
  - ESC-50 street and home sounds at 10 to −10 dB SNR
  - babble of 3 speakers

## 8. Results (headlines; full list in `02_RESULTS_MASTER_TABLE.md`)

**Taka detection**
- **Synthetic composites:** 0.995 mAP@0.5, 0.849 mAP@0.5:0.95 on 2,282 held-out composites.
- **Independent photos (Bangla Money, 1,536):** 91.5% right (first detector), 94.1% (hand-held detector, laptop).
- **Hand-held close-ups (NSTU, 1,127):** 11.8% → 47.5% on the laptop and 8.7% → 33.5% on the Pi.
- **Costs:**
  - 1-Taka notes announced as a value 22.8% → 36.6%
  - three-note total 99.2% → 97.0%
  - on the Pi, about one hand-held answer in seven is a wrong value

**Counterfeit, note-disjoint split (208 notes)**
- **One view:** 71.2% (fixed six-view training) → 98.2% (prefix training), mean of 3 seeds.
- **Significance:** McNemar p ≤ 9.3e-10 per seed at one view; no difference at six views.
- **Seed stability:** fixed training is unstable across seeds (131–194 correct at one view); prefix training gives 197–206.
- **No state-of-the-art claim:** frozen ImageNet probes reach 98–99%.

**Counterfeit, unseen prints (222 notes, about 24 prints)**
- **Prefix network:** about 89–90%.
- **Network + watermark hybrid:** 94.4% (1 view) / 95.0% (6 views), stable across seeds.
- **Full-res fine-tuned ResNet-50:** 94.4–94.9%. This is a tie, not a win.
- **Uncertainty:** about ±12 points at print level.

**DMWR:** 91.5 ± 0.3% (AUC 0.950) with 47,818 parameters. Not significantly different from a fine-tuned MobileNetV2 (92.9%, 2.23 M parameters, 9.15 MB): McNemar p = 0.375 / 0.688 / 0.375. Both streams are needed: each alone gives 85.8–88.0% with a 3–5 point spread.

**Safe verdict**
- **Counterfeits passed:** 0 of 88 test counterfeits (Wilson 0–4.2%) and 0 of 20 whole-note counterfeit photos.
- **Genuine notes confirmed:** 57.5% of close-up genuine notes, but only 0.4–16.9% on whole-note photos.
- **Wording:** "counterfeit" never spoken in 1,889 app runs.
- **Rejected alternative:** a direct genuine/counterfeit verdict at p = 0.5 on whole notes was wrong on 20–59% of genuine notes and 40% of counterfeits, so it was switched off.

**UV fibre check**
- **Genuine notes accepted:** 85.6% (95 of 111).
- **Simulated fibre-free sheets rejected:** 88.6%.
- **Superseded figure:** the earlier 92.8% counted JPEG colour smear as fibres.
- **Not yet tested** on real counterfeits.

**OCR**
- **Rendered test:** CER 9.3% → 2.4% (Bangla 11.9% → 3.2%), and 2.27% with the blended recogniser.
- **Real word photos (ICDAR 2019 MLT):** Bangla 41.7% → 39.2% → 28.7%; Latin 64.9% → 62.5% → 50.6%.
- **Ekush handwriting:** 90.0%.

**Objects:** YOLOv8s 0.551 mAP@0.5 on COCO val2017 (YOLOv8n 0.453). Removing a frame shrink improved the glass path from 0.493 to 0.521.

**Emotion:** RAF-DB 86.5%, macro-F1 0.795 (SOTA about 92%).

**Voice commands (synthetic voices, laptop)**
- **Quiet conditions:** clean 100%, whisper 98.9%.
- **Street noise:** 82.5% at 0 dB.
- **People talking nearby:** 54.3% at 10 dB (was 3.2%).
- **False commands:** 212 → 13 of 5,040 sentences.
- **Response:** median 0.86 s after the speech ends.
- **Bangla command words (Whisper):** 45–62%, not deployed.

**Raspberry Pi 5**

| | First run, 1 Oct (cool board) | Second run, 13 Oct (hot, throttled) |
|---|---|---|
| Currency check | 327 ms | 399 ms |
| Taka detector | 254 ms | 306 ms |
| Watermark check | 50 ms | 64 ms |
| Authenticator, 1 / 6 views | 37 / 90 ms | 43 / 98 ms |
| Objects | YOLOv8n 207 ms | YOLOv8s 562 ms |
| Emotion | — (the 48 ms was a frame without a face) | 680 ms with a face |
| Text reading, one capture | 18.7 s | 18.3 s |
| Memory | peak 1.48 GB | — |
| Temperature | peak 77.1 °C, no throttling | peak 88.7 °C, throttled in 76% of text-test samples, clock 1.5 GHz |

- **Thermal model:** τ ≈ 7 s, T∞ ≈ 74.8 + 0.105·u. An average load above about 50% keeps the board above 80 °C.
- **Response time:** a currency answer takes about 2.3–2.9 s with speech.

## 9. What did not work (keep these in a paper — they are evidence)

- **Direct verdict on whole notes:** the genuine/counterfeit verdict at p = 0.5 was wrong on 20–59% of genuine notes, so it was switched off. Five repair attempts failed:
  - views cut from the whole note
  - a frozen ResNet-50 on the cut views
  - a classifier trained on the counterfeit set
  - PRMVT retrained on synthetic whole notes (it passed 14 of 20 real counterfeits at its own validation threshold)
  - plus the safe policy as the replacement
- **Sequential policies:** learned view-selection and stopping rules did not beat the fixed view order.
- **Simulated federated training:** 74.0% at one view.
- **Prefix training outside currency:** it is not universal. It helped MVP-N with an attention head, not with concatenation, and made no difference on ModelNet.
- **Occlusion:** occluding 55% of the note drops PRMVT to 51.9%. Fine-tuning raised it to 88.5%, still below 90%. Fingers are the main open failure mode.
- **OCR methods that did not help:** binarisation, deskew, Tesseract (about 7× error), TrOCR (unrelated words for Bangla), PaddleOCR (no Bangla), spelling dictionary and KenLM (replace correct rare words), ByT5 (rewrote text, 105.4% CER), multi-scale reading, a rotation retry, INT8 CRAFT (+0.65 pt), and read width 880 (validation said yes, the single test read said no; the width stays 1,200).
- **Voice:**
  - The VAD end-of-speech cut commands at the pause after "glass".
  - The 200 ms response target was not met (0.86 s).
  - At −10 dB noise only 39.5% of commands get through.
  - Offline Bangla commands reached only 45–62%.
- **UV:** the first check was fooled by JPEG colour smear on simulated printed sheets (accepted 93%).

## 10. Corrections and audit history (part of the method)

All claims were audited against their result files (`evidence_notes/CORRECTIONS.md`, `CLAIM_VALUE_AUDIT.md`, `CLAIM_REGISTRY.json`: 107 of 136 claims value-checked). The main corrections:

| Old figure | Corrected to |
|---|---|
| Detector test: 2,238 images, P 0.998 / R 0.997, 7.3 ms on an RTX 4060 (no result file) | Re-run on 2,282 images: P 0.995, R 0.996, 13.9 ms on an RTX 3050 |
| 96.9% on 450 "wild" photos | Source-domain check only; the independent result is 91.5% |
| PRMVT 6-view 97.6% | 98.1% after the NaN-entropy fix |
| Spoken genuine/counterfeit verdict | Switched off, then replaced by the safe policy |
| Bangla OCR CER 27.8% | Withdrawn: test images drawn without Bangla shaping, and a lexicon that contained the test phrases |
| Hybrid's apparent win over a fine-tuned ResNet-50 | That baseline had been trained at quarter resolution; at full resolution it is a tie |
| Object mAP@0.5 0.604 (250-image subset) | 0.551 on all 5,000 images |
| UV 92.8% | 85.6% after the JPEG colour-smear repair |
| Pi emotion 48 ms | A frame without a face; 680 ms with a face |
| Currency answer 6.5 s | About 2.3–2.9 s; the old figure used a longer sentence's duration |

Lesson stated in the thesis: the largest corrections came from looking at the data and the protocol, not from new models. Examples: the serial numbers, the Bangla rendering, a quarter-resolution cache, the source of the "wild" set.

## 11. Evaluation rules used throughout

- **Splits:** disjoint by physical unit (note, writer, print).
- **Choices:** made on validation only; each test set read once; trained models repeated with seeds 42, 43, 44.
- **Statistics:** exact or χ² McNemar for paired comparisons, Bonferroni or Holm where many comparisons are made, and Wilson 95% intervals for proportions.
- **OCR admissibility:** a change is kept only if rendered validation CER rises by less than 0.5 points. The one departure was read width 1,200, and it went towards accuracy.
- **Honesty:** nothing is estimated; a missing measurement is written "not measured".

## 12. Limitations (state them in any paper)

1. **No data through the glass camera:** every accuracy figure comes from datasets or a laptop webcam. The first field test with the glass camera (640×480, fixed focus) was poor: currency unreliable, Bangla not read.
2. **No user study:** trials with a visually impaired participant have started, but there are no results. The protocol (SUS, NASA-TLX, task time) is ready.
3. **Few counterfeit prints:** about 24 unseen prints of one currency, giving about ±12 points of uncertainty. The whole-note safety evidence rests on about 4 physical counterfeits, and the highest counterfeit score (0.99567) is close to τ (0.99959).
4. **Low usefulness on whole notes:** the safe verdict confirms only 0.4–16.9% of genuine whole notes.
5. **UV check:** never tested on a real counterfeit; its fibre-free side is simulated.
6. **Real-photo OCR is weak:** CER 28.7% for Bangla and 50.6% for Latin. Text reading is slow on the Pi (about 18 s per capture).
7. **Voice:** command results come from synthetic voices on a laptop, not people. Nearby talkers remain hard, and Bangla commands are not deployed.
8. **Heat:** the Pi throttles under sustained load. The governor's effect, battery life and the new code's speed on the Pi are not measured.
9. **Objects:** limited to the 80 COCO classes.

## 13. Novelty (what may be claimed) and what may not

**May be claimed:**
- A complete offline Bangla assistive glass on a Raspberry Pi 5 with five modes and measured per-module accuracy and on-device latency.
- Zero-annotation Taka detection from composites, plus synthetic hand-held fine-tuning (11.8% → 47.5% on independent hand-held photos).
- The view-count collapse of jointly trained multi-view fusion networks, its cause (fixed-view sampling), and a one-line fix (prefix sampling) with controlled evidence.
- A serial-number audit that exposes a shortcut in the standard JaalTaka split, and a print-disjoint split.
- DMWR: a physics-based watermark checker that matches a 46× larger network.
- A safe, never-accusing verdict policy with a distribution-free bound.
- A Bangla OCR benchmark with correct shaping and a step-by-step ablation; a blended real-word recogniser.

**Must not be claimed:**
- State of the art on JaalTaka (frozen probes reach 98–99%).
- That the hybrid beats a properly fine-tuned ResNet-50.
- Real-world counterfeit detection, or a validated UV check.
- Results with users or with the glass camera.
- That the glass "detects counterfeits"; it is a screening aid.

## 14. Ethics and safety

- **Wording:** the glass never accuses a note.
- **Privacy:**
  - The online mode is optional and clearly separated.
  - API keys are kept out of the code (`.env`, git-ignored).
  - Field logs and study data stay private. This package contains no participant data; `evidence_notes/user_study/` holds only the protocol, consent form and empty templates.
- **Third-party material:** device photos in the literature figures are Wikimedia Commons images with their licences (see `figures/FIGURE_INDEX.md`). The two cartoon infographics are illustrations, not evidence.

## 15. Package contents

See `README.md`.
