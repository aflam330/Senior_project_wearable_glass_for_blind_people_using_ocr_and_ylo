"""
Smart Glass Configuration
All tunable constants in one place.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _raspberry_pi_5() -> bool:
    """True only when this process is running on a Raspberry Pi 5."""
    try:
        text = open("/proc/device-tree/model", encoding="utf-8", errors="ignore").read()
    except OSError:
        return False
    return "Raspberry Pi 5" in text


ON_RASPBERRY_PI_5 = _raspberry_pi_5()

# ---------------------------------------------------------------------------
# GPIO Pin Numbers (BCM mode — works with rpi-lgpio on RPi 5)
# ---------------------------------------------------------------------------
BUTTON_MODE     = 17   # Cycle through modes
BUTTON_ACTION   = 27   # Capture frame / trigger detection (object & currency: announce immediately;
                       #   OCR: snap + OCR + store text in the session — does NOT speak it yet)
BUTTON_READ     = 24   # Speak back the OCR text most recently stored by ACTION (OCR mode only)
BUTTON_VOL_UP   = 22   # Volume up
BUTTON_VOL_DOWN = 23   # Volume down
BUTTON_DEBOUNCE_MS = 250

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
# Camera. CAMERA_BACKEND: auto (Pi ribbon camera through picamera2 if one is detected, else OpenCV),
# picamera2, or opencv. CAMERA_INDEX is the first OpenCV device tried; other /dev/video* devices are tried
# after it. Both can be set from the environment. Check what the Pi sees with: python scripts/check_camera.py
CAMERA_BACKEND = os.environ.get("CAMERA_BACKEND", "auto")
CAMERA_INDEX  = int(os.environ.get("CAMERA_INDEX", "0"))
# Capture size. 640 x 480 is what the Pi was benchmarked at. A larger size gives text and notes more pixels
# (try CAMERA_WIDTH=1280 CAMERA_HEIGHT=720 on a webcam, 1640 x 1232 on the Pi Camera v2) at the cost of speed.
CAMERA_WIDTH  = int(os.environ.get("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "480"))
CAMERA_FPS    = 30
FRAME_BUFFER_SIZE = 2  # deque maxlen — keep only the freshest frames

# ---------------------------------------------------------------------------
# Application Modes
# ---------------------------------------------------------------------------
MODE_OCR      = 0
MODE_OBJECT   = 1
MODE_CURRENCY = 2
MODE_CLAUDE   = 3   # Online Claude vision (internet + API key required)
MODE_EMOTION  = 4   # Facial expression of the nearest face (modes/emotion_mode.py)

# Bangla mode announcements spoken on switch
MODE_NAMES_BN = [
    "টেক্সট রিডিং মোড",        # Text Reading Mode (offline EasyOCR)
    "অবজেক্ট ডিটেকশন মোড",     # Object Detection Mode
    "কারেন্সি ডিটেকশন মোড",    # Currency Detection Mode
    "অনলাইন ক্লড মোড",         # Online Claude (Windows: C key)
    "ইমোশন ডিটেকশন মোড",       # Emotion Detection Mode
]

# ---------------------------------------------------------------------------
# Model Paths
# ---------------------------------------------------------------------------
# Object model. YOLOv8s is more accurate (mAP@0.5 0.551 against 0.453 for YOLOv8n on all of COCO val2017) and slower:
# on the Raspberry Pi 5, 562 ms against 227 ms per scan (benchmark_report/REPORT.md, 13 October 2026, a hot board).
# YOLO_MODEL=yolov8n.pt chooses the faster one, for example when the board runs hot.
YOLO_MODEL_PATH     = os.path.join(BASE_DIR, "models", os.environ.get("YOLO_MODEL", "yolov8s.pt"))
YOLO_MODEL_FALLBACK = os.path.join(BASE_DIR, "models", "yolov8n.pt")
CURRENCY_MODEL_PATH = os.path.join(BASE_DIR, "models", "currency_mobilenet.pt")
# On a Pi 5 the measured device model is the 18 MB INT8 detector. Elsewhere keep the
# PyTorch weights; ONNX is slower to start when onnxruntime is missing.
_DETECTOR_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "realtime_bangla_taka_detection", "models"))
CURRENCY_YOLO_PATH = os.path.join(_DETECTOR_DIR, "best_int8.onnx" if ON_RASPBERRY_PI_5 else "best.pt")
HAPTIC_PIN          = 13   # BCM — vibration motor for note confirmation

# Genuine/jaal verdict on the detected note. Off: on whole-note photos of genuine notes the
# JaalTaka-trained checker said "jaal" for 20-59 % of them and missed 40 % of counterfeits
# (paper_evidence/JAAL_VERDICT_FIXED.md). The app says the check was not done instead.
# Turn on only with a checker validated on whole-note photos.
JAAL_VERDICT_ENABLED = False

# Safe counterfeit check (policy E, paper_evidence/JAAL_VERDICT_FIXED.md, 2026-09-29).
# 500 / 1000 Taka only. Four JaalTaka-style views are cut from the note crop and scored by PRMVT.
# The glass says "সম্ভবত আসল" (likely genuine) only if p(genuine) > JAAL_SAFE_TAU, otherwise
# "হাতে যাচাই করুন" (check by hand). It never says "জাল" (counterfeit).
# JAAL_SAFE_TAU is the highest score of any JaalTaka VALIDATION counterfeit note. Measured once:
# 0 / 88 JaalTaka test counterfeits and 0 / 25 whole-note counterfeit photos were passed.
JAAL_SAFE_POLICY_ENABLED = True
JAAL_SAFE_TAU = 0.9995918869972229
JAAL_SAFE_DENOMINATIONS = ("500_taka", "1000_taka")
# View windows (x0, y0, x1, y1) on the landscape note crop, medians over 200 JaalTaka TRAIN notes
# (realtime_bangla_taka_detection/results/jaal_whole/view_geometry.json).
# Back-lit watermark check. On for the Pi 5, where the INT8 model is the device path.
# Off elsewhere: it has not been measured on the glass camera. Set WATERMARK_CHECK_ENABLED=1 or 0 to override.
# It never says "counterfeit".
_wm_env = os.environ.get("WATERMARK_CHECK_ENABLED")
WATERMARK_CHECK_ENABLED = ON_RASPBERRY_PI_5 if _wm_env is None else _wm_env == "1"
WATERMARK_MODEL_PATH = os.path.abspath(os.path.join(
    BASE_DIR, "..", "realtime_bangla_taka_detection", "models", "watermark_mobilenetv2_int8.onnx"))  # MobileNetV2 INT8, chosen on VAL AUC; same decisions as FP32
WATERMARK_CLEAR_THRESHOLD = 0.5
# Learned watermark localizer (no template, no SIFT). Seed chosen on VALIDATION corner error (seed 42).
# On unseen-print test notes it ties the SIFT path (92.4 vs 92.9 %, McNemar p = 1.0) and also covers the 25 / 222
# photos SIFT cannot register (realtime_bangla_taka_detection/results/watermark_localizer/). SIFT stays the fallback.
WATERMARK_LOCALIZER_PATH = os.path.abspath(os.path.join(
    BASE_DIR, "..", "realtime_bangla_taka_detection", "models", "watermark_localizer_seed42.onnx"))
# Guided capture for the watermark check (modes/capture_guide.py): after a 500 / 1,000 Taka note is
# announced, ask the user to hold it to the light, reject dark / blurry frames, then check.
# Thresholds: 2nd percentile of JaalTaka VALIDATION back-lit photos
# (realtime_bangla_taka_detection/results/capture_guide/thresholds.json); on test photos they accept
# 213 / 222 clean and 0 / 222 darkened or defocused. Recalibrate on the glass camera before the study.
_cg_env = os.environ.get("CAPTURE_GUIDE_ENABLED")
CAPTURE_GUIDE_ENABLED = WATERMARK_CHECK_ENABLED if _cg_env is None else _cg_env == "1"
CAPTURE_GUIDE_MIN_MEAN = 73.87
CAPTURE_GUIDE_MIN_LAPVAR = 106.75
CAPTURE_GUIDE_TIMEOUT_S = 10.0
# User-study condition (STUDY_CONDITION): "guided" uses the thresholds above; "unguided" asks once for the
# light and checks the first frame with a note in it (no dark / blur rejection). Logged with every check.
STUDY_CONDITION = os.environ.get("STUDY_CONDITION", "guided")
CAPTURE_GUIDE_PROMPT_GAP_S = 2.5

# UV security-fibre check (modes/uv_check.py) for 200, 500 and 1,000 Taka notes: a genuine note shows short glowing
# fibres under a UV lamp, on paper that stays dull. OFF by default: it was measured on phone photographs only, not
# through the glass camera, and never on a counterfeit note (none were available).
# Measured on 118 phone photographs of genuine notes (results/uv_check/):
#  * 2026-10-12, one split by capture order (uv_fibre_check.json): 29 of 43 test notes accepted (67 %). The threshold
#    (14) had been set on front-side photographs and was tested on back-side ones taken from further away.
#  * 2026-10-13, blocked 5-fold cross-validation over capture order, threshold from the other four blocks, same rule
#    (uv_blocked_cv.json): 103 of 111 accepted (92.8 %, Wilson 95 % 86.4-96.3; 200 Taka 45/45, 500 Taka 36/40,
#    1,000 Taka 22/26); all 71 patches of the plain surface beside the note rejected.
#  * 2026-10-13, evening: a SIMULATION with fibre-free printed sheets (scripts/uv_stress_sim.py) showed that the check
#    above also accepted them: with JPEG compression 93 % of 120 simulated sheets passed, because the colour smear at
#    print edges was counted as fibres. Most of what had been counted on genuine notes was of the same kind.
#    A brightness condition did not help (balanced accuracy 0.50). A COLOUR condition does: a fibre pixel must be
#    less blue than the paper by more than UV_COLOUR_THRESHOLD and lie on lit paper. Nested blocked 5-fold with
#    development and held-out sheets (scripts/uv_robust_study.py, results/uv_check/uv_robust_study.json):
#    95 of 111 genuine photographs accepted (85.6 %, Wilson 77.9-90.9) and 88.6 % of held-out simulated sheets
#    rejected. With the setting below (chosen on all genuine photographs and the development sheets) 223 of 240
#    held-out sheets are rejected (JPEG 50: 48 of 60). The old check rejected 4 of 240.
# Still not measured: a real counterfeit note, real printed paper under the lamp, and the glass camera. The fibre-free
# side is a simulation. Slight blur hides fibres, so the UV picture must be close and sharp.
UV_CHECK_ENABLED = os.environ.get("UV_CHECK_ENABLED", "0") == "1"
UV_DENOMINATIONS = ("200_taka", "500_taka", "1000_taka")
UV_MIN_FIBRES = int(os.environ.get("UV_MIN_FIBRES", "1"))       # see above; set again for the glass camera
UV_REL_THRESHOLD = 0.12                                        # fibre brightness relative to the local lamp level
UV_COLOUR_THRESHOLD = float(os.environ.get("UV_COLOUR_THRESHOLD", "0.22"))   # how much less blue than the paper a fibre must be
UV_LUM_THRESHOLD = 0.0                                         # brightness condition: tried, no gain, off
_uv_pin = os.environ.get("UV_LIGHT_PIN", "")
UV_LIGHT_PIN = int(_uv_pin) if _uv_pin else None               # BCM pin that switches the lamp; None = lamp switched by hand
UV_SETTLE_S = 0.4
UV_TIMEOUT_S = 6.0
# The word "counterfeit" is not spoken unless this is switched on by hand. A worn or washed genuine note can lose its
# fibres, and the user cannot check the answer; the default wording is "UV marks not found, check by hand".
UV_SAY_COUNTERFEIT = os.environ.get("UV_SAY_COUNTERFEIT", "0") == "1"
JAAL_VIEW_WINDOWS = (
    (0.0, 0.0, 0.4758, 1.0),
    (0.2843, 0.0, 0.8256, 1.0),
    (0.5725, 0.0, 1.0, 1.0),
    (0.5094, 0.0, 1.0, 1.0),
)

# Announce a Taka denomination only when the top detector box reaches this confidence.
# Chosen on validation only (NSTU validation notes + demonetized notes and coins as unknowns);
# on the Bangla Money test it cut 1-taka notes announced as another value from 43.6 % to 22.8 %
# and kept 89.6 % of known notes correct (was 91.5 %). paper_evidence/OPEN_SET_REJECTION.md
CURRENCY_ANNOUNCE_CONF = 0.60
# Several notes in one picture: count the sure ones and speak the total ("3 notes: ..., total 650 Taka").
# Both numbers were fixed before the test (the announce confidence above and an overlap of 0.5 for "same note");
# measured in scripts/eval_multi_note.py, results/multi_note_total.json. Counterfeit checks stay single-note.
CURRENCY_COUNT_NOTES = os.environ.get("CURRENCY_COUNT_NOTES", "1") == "1"
CURRENCY_COUNT_IOU = 0.5
LABELS_BN_PATH      = os.path.join(BASE_DIR, "assets", "labels_bn.json")

# Piper TTS models. Binary is project-local (models/piper/engine/), not /usr/local/bin, because
# installing there needs sudo and this deploy has none; the binary's RUNPATH is $ORIGIN so its
# bundled .so files are found alongside it without an installer step.
PIPER_EN_MODEL = os.path.join(BASE_DIR, "models", "piper", "en_US-amy-low.onnx")
PIPER_BN_MODEL = os.path.join(BASE_DIR, "models", "piper", "bn_BD-google-medium.onnx")
PIPER_BINARY   = os.path.join(BASE_DIR, "models", "piper", "engine", "piper")

# ---------------------------------------------------------------------------
# Text-to-Speech
# ---------------------------------------------------------------------------
ESPEAK_VOICE_BN  = "bn"      # espeak-ng Bangla voice code
ESPEAK_VOICE_EN  = "en-us"   # espeak-ng English voice code
# Words per minute. 135 is slower than espeak's 150 for clarity. On the Pi 175 wpm would cut speech time by ~27 %;
# that trades clarity for speed and is for blind users to judge, so the default stays 135. Try it with ESPEAK_SPEED=175.
ESPEAK_SPEED     = int(os.environ.get("ESPEAK_SPEED", "135"))
ESPEAK_PITCH     = 45         # 0-99, default 50; slightly lower reads less shrill/robotic
ESPEAK_WORD_GAP  = 4          # 1/100s pause between words — improves intelligibility
DEFAULT_VOLUME   = 80         # percent (0–100)
VOLUME_STEP      = 10         # percent per button press
TTS_QUEUE_MAXSIZE = 5         # drop old items if queue fills
TTS_INTER_UTTERANCE_PAUSE = 0.15  # seconds between queued utterances (sentences/segments)
# Which voice speaks: "auto" = Piper where its model is installed (natural voice), espeak-ng otherwise;
# "espeak" = espeak-ng only (the lightest; use it if Piper is too slow on a hot board).
TTS_VOICE = os.environ.get("TTS_VOICE", "auto")
# Piper speaking rate: 1.0 is the voice's own pace, above 1 is slower. For blind listeners to judge.
PIPER_LENGTH_SCALE = float(os.environ.get("PIPER_LENGTH_SCALE", "1.0"))
# Silence played before and after each piece, so the sound device does not swallow the first or last syllable.
TTS_LEAD_IN_MS = int(os.environ.get("TTS_LEAD_IN_MS", "150"))
TTS_TAIL_MS = 80
# Text reading takes many seconds on the Pi. So that the user knows the press was taken, the glass says "reading"
# at once and gives a soft tick every OCR_PROGRESS_TICK_S seconds until the answer comes (0 = no ticks).
OCR_PROGRESS_CUE = os.environ.get("OCR_PROGRESS_CUE", "1") == "1"
OCR_PROGRESS_TICK_S = float(os.environ.get("OCR_PROGRESS_TICK_S", "4"))

# ---------------------------------------------------------------------------
# Voice commands (voice_commands.py): offline, English command words, Vosk small model
# ---------------------------------------------------------------------------
# "auto" = on when the recogniser, its model and a microphone are there; "0" = off; "1" = on.
VOICE_COMMANDS = os.environ.get("VOICE_COMMANDS", "auto")
VOICE_CMD_MODEL = os.path.join(BASE_DIR, "models", "vosk", "vosk-model-small-en-us-0.15")
# Every word of a command must be recognised with at least this confidence, or the command is ignored.
VOICE_CMD_MIN_CONF = float(os.environ.get("VOICE_CMD_MIN_CONF", "0.7"))
VOICE_CMD_WAKE = os.environ.get("VOICE_WAKE_WORD", "")          # e.g. "glass": then say "glass read text"
VOICE_CMD_MIC = os.environ.get("VOICE_MIC", "")                 # part of the microphone's name; empty = system default
VOICE_CMD_DEAF_AFTER_SPEECH_S = 0.4                             # the room still rings a little after the glass spoke
VOICE_CMD_SETTLE_S = 1.2                                        # after a mode change by voice, before the capture

# ---------------------------------------------------------------------------
# Online Claude (optional — EasyOCR stays the offline text reader)
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")
CLAUDE_MAX_TOKENS = 600
# Second online provider (modes/claude_mode.py). Used when only OPENAI_API_KEY is set, or with ONLINE_PROVIDER=openai.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")

# Emotion-adaptive feedback (E key in Windows test)
EMOTION_ADAPTIVE_DEFAULT = True

# ---------------------------------------------------------------------------
# Inference Thresholds
# ---------------------------------------------------------------------------
# EasyOCR decoder. beamsearch (beamWidth=5) is what scripts/eval_ocr_offline.py measured CER
# with (paper_evidence) and is more accurate on mixed Bangla/English text. Tried greedy on this
# Pi 5 expecting a speedup (2026-10-01): median went from 18.7 s to 20.6 s (n=8 each) — slightly
# *slower*, not faster. The ~19 s cost is not the beam-search decode step; it is EasyOCR's text
# detector running on CPU before decoding ever starts, so switching decoders doesn't touch the
# bottleneck. Kept beamsearch as the default for its measured accuracy. Override with
# OCR_DECODER=greedy only if you want to re-check that result yourself.
OCR_DECODER = os.environ.get("OCR_DECODER", "beamsearch")
OCR_BEAM_WIDTH = 5
# EasyOCR's detector scales the input toward canvas_size before running CRAFT. The glass frame is 640 px wide.
# Pi 5 timing (2026-10-01, one frame): 2560 -> 20.2 s, 640 -> 6.9 s.
# Choice made on a SELECTION set (24 phrases not in the lexicon, new image seeds, 80 images), rule fixed before
# the run: smallest canvas whose raw CER is within 0.5 points of 2560 (results/ocr_select_canvas*.json).
#   raw CER 2560: 13.29 %, 640: 13.84 % (+0.55, Bangla 24.5 -> 25.6 %), 480: 15.37 %, 320: 19.15 %.
# 640 misses the margin by 0.05 points (paired Wilcoxon p = 0.48, not significant), so the default stays 2560.
# OCR_CANVAS_SIZE=640 is the fast option: ~2.5-3x faster for about one point more Bangla character error.
OCR_CANVAS_SIZE = int(os.environ.get("OCR_CANVAS_SIZE", "2560"))
# EasyOCR group_text_box: a box whose slope is below this value is merged as a horizontal line.
# The default 0.1 is about tan(6 degrees), which is the rotation used in the offline renderer,
# so tilted lines were sent to the free-box path instead of the line merger.
# Chosen on a validation phrase list that is not the 24-phrase set and not the 80-image phrase
# list (12 Bangla + 12 English, image seeds 9000/9100/9200, 144 images, GPU). Rule fixed first:
# lowest raw character error, then Bangla error, then seconds. slope_ths=0.2 won
# (raw 0.0926, Bangla 0.1713) against the default 0.1 (raw 0.1417, Bangla 0.2168).
# The 24-phrase set and the 80-image set were read once after that choice
# (results/ocr_improve_summary.json). They were not used to pick the value.
# 24-phrase raw 0.1162 (baseline 0.1329), Bangla 0.2324 (baseline 0.2449).
# 80-image raw 0.0831 (baseline 0.0990), Bangla 0.1428 (baseline 0.1746).
OCR_SLOPE_THS = float(os.environ.get("OCR_SLOPE_THS", "0.2"))

# OCR pipeline v2 (paper_evidence/OCR_TEXT_REGION.md and the other OCR_*.md files, 2026-10-02).
# Chosen on a validation set of 48 new phrases drawn with correct Bangla shaping (HarfBuzz) in two fonts,
# three image seeds, clean / distorted / photo-scene images. Held-out test (48 other phrases, three other fonts),
# read once: character error 9.3 % (pipeline below "legacy") -> 2.4 %; Bangla 11.9 -> 3.2 %; English 6.6 -> 1.6 %.
#   CLAHE preprocessing, EasyOCR text_threshold 0.8 / low_text 0.3 / slope_ths 0.4 / adjust_contrast 0.7,
#   keep the dominant text block (boxes with confidence >= 0.2), lines in reading order, text cleanup rules.
#   The lexicon repair is off in v2: on phrases outside its 24-phrase list it raised error (val 2.83 -> 4.59 %).
# OCR_PIPELINE=legacy restores the earlier path (bilateral + CLAHE, confidence >= OCR_CONFIDENCE, lexicon repair).
OCR_PIPELINE = os.environ.get("OCR_PIPELINE", "v2")
OCR_V2_PARAMS = {"text_threshold": 0.8, "low_text": 0.3, "link_threshold": 0.4, "mag_ratio": 1.0, "contrast_ths": 0.1,
                 "adjust_contrast": 0.7, "slope_ths": 0.4}
# Words the recogniser is less sure about than this are dropped. 0.2 was chosen on rendered text. On real word
# photographs (ICDAR 2019 MLT) a quarter to a half of the words were then not read at all, so four settings were
# compared on a real validation sample (150 Bangla + 150 Latin-script words, disjoint from the test words;
# ocr_bench/real_words_tune.py, results/ocr_bench/real_words_tune.json, 2026-10-13). Rule fixed beforehand: lowest
# real-validation CER among settings whose rendered-validation CER stays within 0.5 points. 0.1 won: real validation
# 47.8 % against 51.1 %, rendered validation 2.58 % against 2.14 %. On the 1,000 real test words, read once:
# Bangla 39.2 % (was 41.7 %), Latin script 62.5 % (was 64.9 %); nothing read on 17 % and 38 % (was 25 % and 48 %).
OCR_V2_MIN_CONF = 0.1
# When nothing is read, try the picture turned by 90 degrees each way (text standing on its side). Off until the
# study in ocr_bench/real_final_eval.py has decided; its evidence is recorded there and here afterwards.
OCR_ROTATE_FALLBACK = os.environ.get("OCR_ROTATE_FALLBACK", "0") == "1"
# Fine-tuned letter recogniser (ocr_recognizer_ft.py). Empty = EasyOCR's released weights.
OCR_RECOGNIZER_PATH = os.environ.get("OCR_RECOGNIZER_PATH", "")
# Decoder for pipeline v2. Whole-page benchmark (ocr_bench/page_bench.py, laptop CPU limited to 4 threads like the
# Pi, results/ocr_bench/page_speed_compare.json, 2026-10-11): greedy reads a 12-line page in 6.5 s against 8.9 s
# for beam search, with the same validation error (48 val phrases x 3 seeds: 2.14 % vs 2.18 %; Bangla 1.59 vs
# 1.64 %). Rule as for the canvas: keep the faster setting only if validation error stays within 0.5 points.
# Batch size 16 gave no speed-up; reading at the camera's own size was 41 % faster but +0.74 points error (rejected).
# The legacy pipeline keeps OCR_DECODER.
OCR_V2_DECODER = os.environ.get("OCR_V2_DECODER", "greedy")
# A picture narrower than this is enlarged to this width before reading (the camera gives 640). The detector's time
# grows with the number of pixels. 1,200 is the width the pipeline was tuned at; smaller widths are being measured in
# ocr_bench/width_study.py (results/ocr_bench/width_study.json) and the default changes only by that study's rule.
OCR_READ_WIDTH = int(os.environ.get("OCR_READ_WIDTH", "1200"))
# Text detector on the CPU: "auto" uses the ONNX Runtime copy of CRAFT when models/craft_fp32.onnx exists (about twice
# as fast on four CPU threads with the same validation error, ocr_detector_onnx.py), otherwise PyTorch.
# "onnx_int8" is faster again but costs 0.65 points of validation error, over the 0.5-point rule: opt-in only.
# "torch" forces the original detector. Not yet timed on the Raspberry Pi.
# ON THE RASPBERRY PI 5 THE DEFAULT IS "onnx_int8" (since 14 October 2026). This is a deliberate exception to the
# 0.5-point rule, made for speed: on the Pi one pass of the full-precision detector took 11 to 18 s
# (benchmark_report/REPORT.md, 13 October), and the 8-bit Taka detector is 1.9 times faster than its full-precision
# version on the same board. Cost, measured on the laptop with the settings now in use: rendered validation CER
# 3.15 % against 2.58 % (+0.57 points; results/ocr_bench/detector_int8_variants.json). The speed of the 8-bit text
# detector on the Pi itself has NOT been measured yet. To go back: OCR_DETECTOR=onnx.
OCR_DETECTOR = os.environ.get("OCR_DETECTOR", "onnx_int8" if ON_RASPBERRY_PI_5 else "auto")
OCR_DETECTOR_ONNX = os.path.join(BASE_DIR, "models", "craft_fp32.onnx")
OCR_DETECTOR_ONNX_INT8 = os.path.join(BASE_DIR, "models", "craft_int8_minmax128.onnx")   # MinMax calibration, 128 images
# Pages: when the main text block holds at least OCR_STREAM_MIN_LINES lines, each line is spoken as soon as it is
# read, while the rest of the page is still being read. Measured (results/ocr_bench/page_streaming.json, same 4-thread
# laptop): first line after 5.5 s on a 12-line page; validation error unchanged (2.14 %, 432 images). Fewer lines
# (signs, labels) take the normal path. Without the main-text-block rule on pages, validation error rose to 4.95 %
# (results/ocr_bench/page_streaming_control.json), so the rule is applied on both paths. Not yet timed on the Pi.
OCR_STREAM_PAGES = os.environ.get("OCR_STREAM_PAGES", "1") == "1"
OCR_STREAM_MIN_LINES = int(os.environ.get("OCR_STREAM_MIN_LINES", "4"))

# Speak the captured text immediately after ACTION (READ repeats it). 0 = only confirm the capture, as before.
# Test preview (preview.py): run the current mode's own detector on the live picture every
# PREVIEW_LIVE_INTERVAL_S seconds so its boxes follow the camera (currency notes, faces). Only the
# window for sighted testers uses this; the blind user's headless glass never does. Set to 0 on a busy Pi.
PREVIEW_LIVE_BOXES = os.environ.get("PREVIEW_LIVE_BOXES", "1") == "1"
PREVIEW_LIVE_INTERVAL_S = float(os.environ.get("PREVIEW_LIVE_INTERVAL_S", "1.5" if ON_RASPBERRY_PI_5 else "0.7"))
PREVIEW_CAPTURE_BOX_S = 15.0   # how long the boxes of one capture (text, online picture) stay on screen
OCR_SPEAK_ON_CAPTURE = os.environ.get("OCR_SPEAK_ON_CAPTURE", "1") == "1"
OCR_CONFIDENCE      = 0.4    # EasyOCR minimum confidence
OBJECT_CONFIDENCE   = 0.50   # YOLO minimum confidence
# Stricter minimum for animals only. In dim indoor tests (2026-10-06) the COCO model called dark clutter
# "cat" (0.39-0.48) and "dog" (0.48) while the real desk objects scored 0.55-0.65; 0.65 drops the false animals
# and keeps those objects. Any class not listed uses OBJECT_CONFIDENCE.
OBJECT_CLASS_CONFIDENCE = {name: 0.65 for name in (
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "teddy bear")}
CURRENCY_CONFIDENCE = 0.65   # MobileNetV3 minimum softmax score
OCR_MIN_CHARS       = 3      # Discard OCR hits shorter than this

# ---------------------------------------------------------------------------
# Timing / Cooldowns (seconds)
# ---------------------------------------------------------------------------
OBJECT_SCAN_INTERVAL      = 1.5   # Seconds between automatic object scans
OBJECT_ANNOUNCE_COOLDOWN  = 4.0   # Per-class cooldown to avoid repetition
DETECTION_THREAD_SLEEP    = 0.1   # Main loop sleep when not in object mode

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE        = os.path.join(BASE_DIR, "logs", "smart_glass.log")
# Field-test log (field_log.py): logs/field/<run>/events.csv + system.csv + session.json, written by a background
# thread in batches. FIELD_LOG_ENABLED=0 turns it off; FIELD_LOG_SAVE_FRAMES=1 also keeps one 640 px JPEG per
# ACTION press (for checking answers later; capped at 2,000 frames and stops when under 500 MB is free).
FIELD_LOG_ENABLED = os.environ.get("FIELD_LOG_ENABLED", "1") == "1"
FIELD_LOG_DIR = os.environ.get("FIELD_LOG_DIR", os.path.join(BASE_DIR, "logs", "field"))
FIELD_LOG_SYS_INTERVAL_S = float(os.environ.get("FIELD_LOG_SYS_INTERVAL_S", "10"))
FIELD_LOG_SAVE_FRAMES = os.environ.get("FIELD_LOG_SAVE_FRAMES", "0") == "1"
LOG_MAX_BYTES   = 5 * 1024 * 1024   # 5 MB
LOG_BACKUP_COUNT = 3
