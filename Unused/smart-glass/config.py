"""
Smart Glass Configuration
All tunable constants in one place.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
CAMERA_INDEX  = 0      # 0 = Pi Camera via libcamera-apps, 1+ = USB
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30
FRAME_BUFFER_SIZE = 2  # deque maxlen — keep only the freshest frames

# ---------------------------------------------------------------------------
# Application Modes
# ---------------------------------------------------------------------------
MODE_OCR      = 0
MODE_OBJECT   = 1
MODE_CURRENCY = 2

# Bangla mode announcements spoken on switch
MODE_NAMES_BN = [
    "টেক্সট রিডিং মোড",        # Text Reading Mode
    "অবজেক্ট ডিটেকশন মোড",     # Object Detection Mode
    "কারেন্সি ডিটেকশন মোড",    # Currency Detection Mode
]

# ---------------------------------------------------------------------------
# Model Paths
# ---------------------------------------------------------------------------
YOLO_MODEL_PATH     = os.path.join(BASE_DIR, "models", "yolov8n.pt")
CURRENCY_MODEL_PATH = os.path.join(BASE_DIR, "models", "currency_mobilenet.pt")
LABELS_BN_PATH      = os.path.join(BASE_DIR, "assets", "labels_bn.json")

# Piper TTS models
PIPER_EN_MODEL = os.path.join(BASE_DIR, "models", "piper", "en_US-amy-low.onnx")
PIPER_BN_MODEL = os.path.join(BASE_DIR, "models", "piper", "bn_BD-medium.onnx")
PIPER_BINARY   = "/usr/local/bin/piper"

# ---------------------------------------------------------------------------
# Text-to-Speech
# ---------------------------------------------------------------------------
ESPEAK_VOICE_BN  = "bn"      # espeak-ng Bangla voice code
ESPEAK_VOICE_EN  = "en-us"   # espeak-ng English voice code
ESPEAK_SPEED     = 135        # words per minute — slower than default 150 for clarity
ESPEAK_PITCH     = 45         # 0-99, default 50; slightly lower reads less shrill/robotic
ESPEAK_WORD_GAP  = 4          # 1/100s pause between words — improves intelligibility
DEFAULT_VOLUME   = 80         # percent (0–100)
VOLUME_STEP      = 10         # percent per button press
TTS_QUEUE_MAXSIZE = 5         # drop old items if queue fills
TTS_INTER_UTTERANCE_PAUSE = 0.15  # seconds between queued utterances (sentences/segments)

# ---------------------------------------------------------------------------
# Inference Thresholds
# ---------------------------------------------------------------------------
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
LOG_MAX_BYTES   = 5 * 1024 * 1024   # 5 MB
LOG_BACKUP_COUNT = 3
