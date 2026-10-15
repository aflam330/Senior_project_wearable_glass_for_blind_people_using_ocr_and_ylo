# New features, 13 October 2026

Each feature below is implemented in the glass application, has a test or a measurement, and says what was not measured. Features from the requested list that were not built are at the end, with the reason.

## 1. Several notes in one picture: count and total

**What it does.** When the camera sees two or more notes it is sure about, the glass says how many, names them from the largest, and says the total, for example "৩টি নোট: পাঁচশত টাকার নোট, একশত টাকার নোট, পঞ্চাশ টাকার নোট। মোট ৬৫০ টাকা" (three notes: 500, 100, 50; total 650 Taka). With one note it behaves as before, including the counterfeit answer.

**How.** `CurrencyMode.distinct_notes` keeps the boxes at or above the announce confidence (0.60) and treats a box that overlaps a more confident one by more than 0.5 as the same note seen twice. Both numbers were fixed before the test; nothing was tuned.

**Measurement** (`savior_glass/scripts/eval_multi_note.py` → `results/multi_note_total.json`). 900 pictures built for this test from 513 source notes of the detector's test split, on COCO train2017 backgrounds that were not used in training, with the training generator's tilt, rotation, colour, blur, noise and JPEG steps.

| Notes in the picture | Pictures | Count right | Every note right, so the total is right |
|---:|---:|---:|---:|
| 1 | 300 | 300 (100 %) | 299 (99.7 %) |
| 2 | 300 | 298 (99.3 %) | 298 (99.3 %) |
| 3 | 300 | 299 (99.7 %) | 296 (98.7 %) |
| all | 900 | 99.7 % | 99.2 % |

No note was counted twice. Three times a note was missed.

**Limits.** Synthetic pictures; notes overlap by at most 5 % of the smaller note, so a fanned stack of notes is not covered. Laptop path with the PyTorch weights, not the INT8 graph of the Pi. Not measured through the glass camera. Given the 18.5 % on close-ups with fingers, a hand full of notes will be harder than these pictures.

## 2. Text standing on its side: rotation retry

**What it does.** When nothing at all is read, the picture is tried turned by 90° each way and the more confident reading is kept (`OCR_ROTATE_FALLBACK`). It costs time only when nothing was read.

**Why.** On real word photographs, rotated words were not read at all (84 % and 99 % error).

**Measurement and decision.** In `OCR_REAL_IMPROVED.md`. Off until that study has decided.

## 3. A recogniser fine-tuned on real words

A loader for fine-tuned recogniser weights (`ocr_recognizer_ft.py`, `OCR_RECOGNIZER_PATH`). Training, rule and results in `OCR_REAL_IMPROVED.md`.

## 4. Earlier in this pass (11–13 October)

| Feature | Measured | Where |
|---|---|---|
| Whole pages spoken line by line | First line after about 5.5 s; validation error unchanged | `OCR_SPEED_IMPROVED.md` |
| Text detector through ONNX Runtime | 2.1 times faster, identical output | `OCR_SPEED_IMPROVED.md` |
| Bangla speech kept whole; long texts in order; failing voice tried once | 7 checks in `scripts/test_tts_queue.py` | `QUALITY_LOOP.md` |
| Object mode without the needless shrink | mAP@0.5 0.493 → 0.521 | `OBJECT_IMPROVED.md` |
| UV fibre check with a colour condition (off by default) | 85.6 % of genuine photographs accepted; 88.6 % of simulated fibre-free sheets rejected | `UV_SYNTHETIC_TEST.md` |

## 5. Already in the application before this pass

Session logging (`field_log.py`), error recovery (every model in a `try` block with a fall-back; the service restarts after a crash), offline operation with no picture stored or sent in the offline modes, a retry on a brightened copy for dark frames, and a check that the back-lit picture is bright and sharp enough. These were not rebuilt.

## 6. Requested and not built

| Item | Why not |
|---|---|
| Damaged, folded and partly visible notes | The data that shows the problem (1,144 close-ups, 18.5 %) is a test set. Training on it would remove the only independent measure, and no other labelled photographs of such notes are on disk |
| Adaptive threshold per user; learning from user feedback | The safe threshold is a guarantee because it is fixed. A threshold that moves with feedback from a user who cannot check the answer would lose it |
| Battery optimisation; thermal management | Need the device. The Pi stayed below throttling in the 30-minute run; battery life has never been measured |
| Offline model updates | No need has appeared; models are replaced by pulling the repository |
| Note orientation | The detector already names notes at any angle within its training range (±30° plus tilt); upside-down notes were not tested |

---

# Added 14-15 October 2026

| Feature | What the user gets | Evidence | Measured on the Pi? |
|---|---|---|---|
| Natural Bangla voice | Piper Bangla voice instead of espeak-ng; every sentence rendered before it plays (no broken sound); next sentence prepared during playback; fixed prompts cached | `VOICE_2026-10-14.md` | No |
| Voice commands with a wake word | "glass, read text", "glass, check money", "glass, stop" ... instead of buttons; whispering works; other people's talk rarely triggers anything | `VOICE_COMMANDS_FAST.md` | No |
| "Reading" cue | "পড়ছি" at once and a soft tick every 4 s during a text capture | `scripts/test_tts_queue.py` | No |
| Hand-held notes | Notes held close, folded or under fingers are recognised far more often | `HANDHELD_FINETUNE.md` | Pi version measured on photos, not on the Pi |
| Thermal governor | Automatic object scans and preview boxes slow down or pause when the board is hot; button presses never wait | `THERMAL.md` | No |
| Power and battery logger | `scripts/power_log.py` estimates battery hours from the Pi 5's own supply rails | `THERMAL.md` | Not run yet |
| One-command Pi check | `scripts/pi_check_all.sh` runs every test and times text reading and the voice | `TEST_RESULTS.md` | Waits for a run |
| Start at boot that works | `scripts/install_service.sh` installs a user service for the real folder | `BUGS_FIXED.md` | No |

Not built from the requested list, with the reason: voice biometrics and custom user profiles (one user per glass; no recordings of users to build or test them on); emotion-voice fusion (no data with both); offline updates (updates arrive with `git pull`); note orientation and damaged-note handling as separate features (the hand-held fine-tune covers turned, folded and partly hidden notes in one model).
