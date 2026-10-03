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
YOLO_MODEL_PATH     = os.path.join(BASE_DIR, "models", "yolov8s.pt")
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
OCR_V2_MIN_CONF = 0.2

OCR_CONFIDENCE      = 0.4    # EasyOCR minimum confidence
OBJECT_CONFIDENCE   = 0.50   # YOLO minimum confidence
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
