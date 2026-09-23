"""
Windows Development / Testing Script
=====================================
Tests all three Smart Glass modes on Windows using:
  - Webcam via OpenCV
  - tkinter window for live camera display
  - TTS: Piper if installed, else gTTS for Bangla, else Windows SAPI (English only)
  - Keyboard keys instead of GPIO buttons

Keyboard controls (click the window first):
  M  ->  Cycle mode    (OCR -> Object -> Currency -> Online Claude)
  A  ->  CAPTURE       (OCR: snap + EasyOCR + store text — offline
                        | Object/Currency: detect and announce immediately
                        | Claude: send the frame to Claude online)
  C  ->  CLAUDE        (always online: send current camera frame to Claude API)
  R  ->  READ          (OCR: speak last EasyOCR text | Claude: speak last online result)
  +  ->  Volume up
  -  ->  Volume down
  Q  ->  Quit

Install:
  pip install easyocr ultralytics torch torchvision numpy Pillow gtts pygame
"""

import os
import re
import sys
import queue
import logging
import subprocess
import tempfile
import threading
import time
import tkinter as tk
import warnings
from tkinter import ttk

# EasyOCR sets pin_memory=True internally; suppress the GPU-not-found noise
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)

import cv2
import numpy as np
from PIL import Image, ImageTk

_BANGLA_RE = re.compile(r"[ঀ-৿]")

# ---------------------------------------------------------------------------
# Logging with UTF-8 so Bangla prints in terminal
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
        )
    ],
)
logger = logging.getLogger("smart_glass_win")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODE_OCR, MODE_OBJECT, MODE_CURRENCY, MODE_CLAUDE = 0, 1, 2, 3
MODE_NAMES_EN = [
    "Text Reading Mode",
    "Object Detection Mode",
    "Currency Detection Mode",
    "Online Claude Mode",
]
MODE_NAMES_BN = [
    "টেক্সট রিডিং মোড",
    "অবজেক্ট ডিটেকশন মোড",
    "কারেন্সি ডিটেকশন মোড",
    "অনলাইন ক্লড মোড",
]

YOLO_MODEL_PATH     = os.path.join(BASE_DIR, "models", "yolov8n.pt")
CURRENCY_MODEL_PATH = os.path.join(BASE_DIR, "models", "currency_mobilenet.pt")
LABELS_BN_PATH      = os.path.join(BASE_DIR, "assets", "labels_bn.json")

# Piper TTS — drop piper.exe + the .onnx models into models/piper/ to enable
# natural neural TTS. Falls back to Windows SAPI automatically if not present.
# Download: https://github.com/rhasspy/piper/releases  (piper_windows_amd64.zip)
PIPER_EXE      = os.path.join(BASE_DIR, "models", "piper", "piper.exe")
PIPER_EN_MODEL = os.path.join(BASE_DIR, "models", "piper", "en_US-hfc_female-medium.onnx")
PIPER_BN_MODEL = os.path.join(BASE_DIR, "models", "piper", "bn_BD-medium.onnx")

OCR_CONFIDENCE           = 0.4
OBJECT_CONFIDENCE        = 0.50
CURRENCY_CONFIDENCE      = 0.65
OCR_MIN_CHARS            = 3
OBJECT_SCAN_INTERVAL     = 2.0
OBJECT_ANNOUNCE_COOLDOWN = 4.0
VOLUME_STEP              = 10
CAMERA_INDEX             = 0
DISPLAY_W, DISPLAY_H    = 640, 480


# Known Bangla phrases → English, used only if gTTS/Piper are unavailable.
_BN_EN_FALLBACK = (
    ("সামনে আছে: ", "In front: "),
    ("সামনে আছে:", "In front:"),
    ("দুই টাকার নোট", "2 taka"),
    ("পাঁচ টাকার নোট", "5 taka"),
    ("দশ টাকার নোট", "10 taka"),
    ("বিশ টাকার নোট", "20 taka"),
    ("পঞ্চাশ টাকার নোট", "50 taka"),
    ("একশত টাকার নোট", "100 taka"),
    ("দুইশত টাকার নোট", "200 taka"),
    ("পাঁচশত টাকার নোট", "500 taka"),
    ("এক হাজার টাকার নোট", "1000 taka"),
    ("লেখা সংরক্ষণ করা হয়েছে। শুনতে রিড বাটন চাপুন", "Text saved. Press R to listen."),
    ("পড়া হচ্ছে: ", "Reading: "),
    ("কোনো লেখা পাওয়া যায়নি", "No text found."),
    ("কোনো লেখা সংরক্ষিত নেই। প্রথমে ক্যাপচার করুন", "Nothing saved yet. Capture first."),
    ("নোট সনাক্ত করা যায়নি। ক্যামেরার সামনে ধরুন।", "No note found. Hold it in front of the camera."),
    ("নোট নিশ্চিত করা যায়নি। আরও কাছে ধরুন।", "Could not confirm the note. Hold it closer."),
    ("ক্যামেরা প্রস্তুত নয়", "Camera not ready."),
    ("অনলাইন ক্লড মোড", "Online Claude Mode"),
    ("কোনো অনলাইন বিবরণ নেই। আগে C চাপুন।", "No online description yet. Press C first."),
)


def _english_fallback(text: str) -> str:
    out = text
    for bn, en in _BN_EN_FALLBACK:
        out = out.replace(bn, en)
    if _BANGLA_RE.search(out):
        return "Detection complete. Bangla is on the screen."
    return out


# ---------------------------------------------------------------------------
# TTS Engine — Piper (offline) → gTTS (Bangla) → Windows SAPI (English)
# ---------------------------------------------------------------------------

class WindowsTTS:
    """
    Background TTS worker.

    Windows SAPI has no Bangla voice, so Bangla detections used to be replaced
    with a short English cue and were not spoken. Order now:

      1. Piper, if piper.exe + voice models are present
      2. gTTS (needs internet) for Bangla — speaks the real detected text
      3. Windows SAPI for English, or an English gloss of known Bangla phrases
    """

    def __init__(self, volume: int = 85):
        self._volume     = max(0, min(100, volume))
        self._q: queue.Queue = queue.Queue(maxsize=5)
        self._stop_event = threading.Event()
        self._proc       = None
        self._proc_lock  = threading.Lock()
        self._mixer      = None

        self._worker = threading.Thread(
            target=self._run, daemon=True, name="tts-worker"
        )
        self._worker.start()
        logger.info("[TTS] Worker started (Piper / gTTS Bangla / SAPI English)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str, emotion: str = "neutral", adaptive: bool = True) -> None:
        if not text or not text.strip():
            return

        try:
            from assistive import adapt_feedback
            bangla = len(_BANGLA_RE.findall(text.replace(" ", ""))) / max(len(text.replace(" ", "")), 1) > 0.25
            adapted = adapt_feedback(text, emotion, enabled=adaptive, bangla=bangla)
            text = adapted["text"]
            rate_delta = int(adapted.get("rate") or 0)
        except Exception:
            rate_delta = 0

        stripped     = text.replace(" ", "")
        bangla_ratio = len(_BANGLA_RE.findall(stripped)) / max(len(stripped), 1)
        lang         = "bn" if bangla_ratio > 0.25 else "en"

        logger.info("[TTS] Queuing (%s, emo=%s, adapt=%s): %s", lang, emotion, adaptive, text[:70])

        if self._q.full():
            try:
                self._q.get_nowait()
                self._q.task_done()
            except queue.Empty:
                pass
        self._q.put_nowait((text.strip(), lang, rate_delta))

    def stop(self) -> None:
        self._stop_event.set()
        if self._mixer is not None:
            try:
                self._mixer.music.stop()
            except Exception:
                pass
        with self._proc_lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
        while not self._q.empty():
            try:
                self._q.get_nowait()
                self._q.task_done()
            except queue.Empty:
                break
        self._stop_event.clear()

    def set_volume(self, pct: int) -> None:
        self._volume = max(0, min(100, pct))

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame
            pygame.mixer.init()
            self._mixer = pygame.mixer
        except Exception as exc:
            logger.warning("[TTS] pygame mixer unavailable (%s) — gTTS playback limited", exc)

        while True:
            item = self._q.get()
            if isinstance(item, tuple) and len(item) == 3:
                text, lang, rate_delta = item
            else:
                text, lang = item
                rate_delta = 0
            try:
                if os.path.isfile(PIPER_EXE):
                    self._piper_speak(text, lang)
                elif lang == "bn":
                    if not self._gtts_speak(text, "bn"):
                        self._sapi_speak(_english_fallback(text), "en", rate_delta)
                else:
                    self._sapi_speak(text, lang, rate_delta)
            except Exception as exc:
                logger.error("[TTS] Error: %s", exc)
            finally:
                self._q.task_done()

    # ------------------------------------------------------------------
    # Piper TTS backend (natural neural voice, fully offline)
    # ------------------------------------------------------------------

    def _piper_speak(self, text: str, lang: str) -> None:
        model = PIPER_BN_MODEL if lang == "bn" else PIPER_EN_MODEL
        if not os.path.isfile(model):
            logger.warning("[TTS] Piper model not found for lang=%s, falling back", lang)
            if lang == "bn" and self._gtts_speak(text, "bn"):
                return
            self._sapi_speak(text, lang, 0)
            return

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name

            # Render text → WAV
            piper_proc = subprocess.Popen(
                [PIPER_EXE, "--model", model, "--output_file", tmp_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            piper_proc.stdin.write(text.encode("utf-8"))
            piper_proc.stdin.close()
            piper_proc.wait()

            if self._stop_event.is_set():
                return

            # Play WAV via PowerShell SoundPlayer (blocking, killable)
            escaped = tmp_path.replace("\\", "\\\\")
            play_script = (
                f"$p = New-Object System.Media.SoundPlayer '{escaped}'; $p.PlaySync()"
            )
            proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive",
                 "-WindowStyle", "Hidden", "-Command", play_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self._proc_lock:
                self._proc = proc

            while proc.poll() is None:
                if self._stop_event.is_set():
                    proc.terminate()
                    break
                time.sleep(0.05)

            logger.info("[TTS] Piper done")

        except Exception as exc:
            logger.warning("[TTS] Piper failed (%s), falling back to SAPI", exc)
            self._sapi_speak(text, lang, 0)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _gtts_speak(self, text: str, lang: str) -> bool:
        """Speak via Google TTS. Returns False if download/playback fails."""
        if self._mixer is None:
            return False
        tmp_path = None
        try:
            from gtts import gTTS
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            gTTS(text=text, lang=lang).save(tmp_path)
            if self._stop_event.is_set():
                return True
            logger.info("[TTS] gTTS (%s): %s", lang, text[:60])
            self._mixer.music.load(tmp_path)
            self._mixer.music.set_volume(self._volume / 100.0)
            self._mixer.music.play()
            while self._mixer.music.get_busy():
                if self._stop_event.is_set():
                    self._mixer.music.stop()
                    break
                time.sleep(0.05)
            try:
                self._mixer.music.unload()
            except Exception:
                pass
            logger.info("[TTS] Done (gTTS)")
            return True
        except Exception as exc:
            logger.warning("[TTS] gTTS failed (%s)", exc)
            return False
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Windows SAPI backend (offline English fallback)
    # ------------------------------------------------------------------

    def _sapi_speak(self, text: str, lang: str, rate_delta: int = 0) -> None:
        speak_text = text if lang == "en" else _english_fallback(text)
        safe = speak_text.replace("'", "''")
        sapi_rate = max(-10, min(10, 1 + int(rate_delta)))
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Volume = {self._volume}; $s.Rate = {sapi_rate}; $s.Speak('{safe}');"
        )
        logger.info("[TTS] SAPI: %s", speak_text[:60])
        proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive",
             "-WindowStyle", "Hidden", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with self._proc_lock:
            self._proc = proc

        while proc.poll() is None:
            if self._stop_event.is_set():
                proc.terminate()
                break
            time.sleep(0.05)

        logger.info("[TTS] Done (SAPI)")


# ---------------------------------------------------------------------------
# Load modes
# ---------------------------------------------------------------------------

def load_modes():
    sys.path.insert(0, BASE_DIR)
    from modes.ocr_mode      import OCRMode
    from modes.object_mode   import ObjectMode
    from modes.currency_mode import CurrencyMode
    from modes.claude_mode   import ClaudeMode
    return [OCRMode(), ObjectMode(), CurrencyMode(), ClaudeMode()]


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

class SmartGlassWin:

    def __init__(self, root: tk.Tk):
        self._root       = root
        self._tts        = WindowsTTS(volume=90)
        self._modes      = load_modes()
        self._mode       = MODE_OCR
        self._volume     = 90
        self._lock       = threading.Lock()
        self._running    = True
        self._detecting  = False
        self._frame      = None
        self._frame_lock = threading.Lock()
        self._emotion_label = "neutral"
        self._emotion_face = None
        self._adaptive_on = True
        self._pose = None
        self._currency_hits = []
        self._currency_spoken = None
        self._straighten_spoken = False
        self._emo_lock = threading.Lock()
        try:
            from assistive import EmotionDetector
            self._emotion = EmotionDetector(download=True)
        except Exception as exc:
            self._emotion = None
            logger.warning("Emotion detector unavailable: %s", exc)

        self._build_ui()
        self._start_camera()
        self._start_scan_thread()
        self._start_emotion_thread()

        self._modes[self._mode].activate()
        if self._emotion is not None:
            self._log("Emotion detector ready (FER+ / HGB). E toggles adaptive speech.")
        else:
            self._log("Emotion detector unavailable.")
        self._log("Features: 1) OCR EasyOCR bn+en  2) Object YOLOv8  3) Currency YOLO+pose")
        self._log("          4) Claude online  5) Emotion HUD  6) Adaptive TTS  7) Volume")
        self._tts.speak("Smart Glass ready. Text Reading Mode.")
        self._log("Ready | M=mode A=capture C=Claude E=emotion adaptive R=read +/-=vol Q=quit")

        root.bind("<Key-m>",     lambda e: self._switch_mode())
        root.bind("<Key-M>",     lambda e: self._switch_mode())
        root.bind("<Key-a>",     lambda e: self._action())
        root.bind("<Key-A>",     lambda e: self._action())
        root.bind("<Key-c>",     lambda e: self._claude_action())
        root.bind("<Key-C>",     lambda e: self._claude_action())
        root.bind("<Key-e>",     lambda e: self._toggle_adaptive())
        root.bind("<Key-E>",     lambda e: self._toggle_adaptive())
        root.bind("<Key-r>",     lambda e: self._read_stored())
        root.bind("<Key-R>",     lambda e: self._read_stored())
        root.bind("<Key-plus>",  lambda e: self._vol_up())
        root.bind("<Key-equal>", lambda e: self._vol_up())
        root.bind("<Key-minus>", lambda e: self._vol_down())
        root.bind("<Key-q>",     lambda e: self._quit())
        root.bind("<Key-Q>",     lambda e: self._quit())
        root.protocol("WM_DELETE_WINDOW", self._quit)

        self._refresh_frame()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._root.title("Smart Glass — Windows Test")
        self._root.configure(bg="#1a1a2e")
        self._root.resizable(False, False)

        # Camera feed
        self._canvas = tk.Canvas(
            self._root, width=DISPLAY_W, height=DISPLAY_H,
            bg="black", highlightthickness=0,
        )
        self._canvas.pack()

        # Top info row
        info_frame = tk.Frame(self._root, bg="#1a1a2e")
        info_frame.pack(fill=tk.X, padx=10, pady=(6, 0))

        self._mode_var = tk.StringVar(value=f"● {MODE_NAMES_EN[self._mode]}")
        tk.Label(info_frame, textvariable=self._mode_var,
                 font=("Segoe UI", 12, "bold"), fg="#00ff88", bg="#1a1a2e"
                 ).pack(side=tk.LEFT)

        self._hud_var = tk.StringVar(value="emotion: —  adapt: ON  pose: —")
        tk.Label(info_frame, textvariable=self._hud_var,
                 font=("Segoe UI", 9), fg="#7ec8e3", bg="#1a1a2e"
                 ).pack(side=tk.LEFT, padx=14)

        self._tts_var = tk.StringVar(value="ready")
        tk.Label(info_frame, textvariable=self._tts_var,
                 font=("Segoe UI", 10), fg="#aaaaaa", bg="#1a1a2e"
                 ).pack(side=tk.RIGHT)

        # Result area — Nirmala UI renders Bangla
        self._result_var = tk.StringVar(value="Press A for offline OCR, or C for online Claude")
        tk.Label(
            self._root, textvariable=self._result_var,
            font=("Nirmala UI", 13), fg="#ffffff", bg="#1a1a2e",
            wraplength=630, justify="left",
        ).pack(padx=10, pady=(4, 2))

        # Clickable demo buttons — every feature on screen
        btn_row = tk.Frame(self._root, bg="#1a1a2e")
        btn_row.pack(fill=tk.X, padx=8, pady=(4, 2))
        buttons = [
            ("Mode (M)", self._switch_mode),
            ("Capture (A)", self._action),
            ("Claude (C)", self._claude_action),
            ("Emotion (E)", self._toggle_adaptive),
            ("Read (R)", self._read_stored),
            ("Vol +", self._vol_up),
            ("Vol -", self._vol_down),
            ("Quit (Q)", self._quit),
        ]
        for label, cmd in buttons:
            def _click(fn=cmd):
                fn()
                self._root.focus_set()
            tk.Button(
                btn_row, text=label, command=_click,
                font=("Segoe UI", 8, "bold"), fg="#111", bg="#00ff88",
                activebackground="#66ffaa", relief=tk.FLAT, padx=7, pady=3,
                takefocus=0,
            ).pack(side=tk.LEFT, padx=2)

        # Volume bar
        vol_row = tk.Frame(self._root, bg="#1a1a2e")
        vol_row.pack(pady=(0, 2))
        tk.Label(vol_row, text="Vol:", fg="#777", bg="#1a1a2e",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self._vol_bar = ttk.Progressbar(vol_row, length=130, maximum=100, mode="determinate")
        self._vol_bar["value"] = self._volume
        self._vol_bar.pack(side=tk.LEFT, padx=4)
        self._vol_lbl = tk.Label(vol_row, text=f"{self._volume}%",
                                  fg="#777", bg="#1a1a2e", font=("Segoe UI", 9))
        self._vol_lbl.pack(side=tk.LEFT)

        # Hint
        tk.Label(
            self._root,
            text="  Features: OCR | Object YOLO | Currency YOLO+pose | Claude | Emotion HUD | Adaptive TTS  ",
            font=("Segoe UI", 9), fg="#888", bg="#0d0d1e",
        ).pack(fill=tk.X)

        # Log box
        self._log_box = tk.Text(
            self._root, height=6,
            font=("Nirmala UI", 9),          # Bangla-capable font
            bg="#0a0a18", fg="#88ff88",
            state=tk.DISABLED, relief=tk.FLAT, wrap=tk.WORD,
        )
        self._log_box.pack(fill=tk.X, padx=8, pady=(2, 8))

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log_box.config(state=tk.NORMAL)
        self._log_box.insert(tk.END, f"[{ts}] {msg}\n")
        self._log_box.see(tk.END)
        self._log_box.config(state=tk.DISABLED)
        logger.info(msg)

    def _set_tts_status(self, msg: str):
        self._root.after(0, lambda: self._tts_var.set(f"🔊 {msg}"))

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _start_camera(self):
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            self._log("ERROR: Cannot open webcam")
            return
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        def _grab():
            while self._running:
                ret, frame = self._cap.read()
                if ret:
                    with self._frame_lock:
                        self._frame = frame
                else:
                    time.sleep(0.01)

        threading.Thread(target=_grab, daemon=True, name="camera").start()
        self._log("Camera started")

    def _refresh_frame(self):
        with self._frame_lock:
            frame = None if self._frame is None else self._frame.copy()
        if frame is not None:
            vis = frame.copy()
            with self._emo_lock:
                face = self._emotion_face
                label = self._emotion_label
                adaptive = self._adaptive_on
                pose = self._pose
            if face:
                x, y, w, h = face
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 2)
                cv2.putText(
                    vis, f"{label}  adapt={'ON' if adaptive else 'OFF'}",
                    (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2,
                )
            else:
                cv2.putText(
                    vis, f"emotion:{label} adapt={'ON' if adaptive else 'OFF'}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2,
                )
            with self._emo_lock:
                hits = list(self._currency_hits)
            for hit in hits:
                x, y, w, h = hit["bbox"]
                auth = hit.get("auth")
                if auth == "counterfeit":
                    color = (0, 0, 255)
                elif auth == "genuine":
                    color = (0, 220, 0)
                else:
                    color = (0, 220, 255)
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                tag = hit.get("auth_en") or ""
                cv2.putText(
                    vis, f"{hit['name']} {tag} {hit['conf']*100:.0f}%",
                    (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )
            if pose and pose.get("ok") and pose.get("quad") is not None:
                pts = pose["quad"].astype(np.int32).reshape((-1, 1, 2))
                color = (0, 0, 255) if pose.get("needs_straighten") else (0, 255, 255)
                cv2.polylines(vis, [pts], True, color, 2)
                msg = f"R{pose['roll']:+.0f} P{pose['pitch']:+.0f} Y{pose['yaw']:+.0f}"
                if pose.get("needs_straighten"):
                    msg += "  STRAIGHTEN"
                cv2.putText(vis, msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                pose_txt = msg
            else:
                pose_txt = "—"
            self._hud_var.set(
                f"emotion: {label}  adapt: {'ON' if adaptive else 'OFF'}  pose: {pose_txt}"
            )
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            self._tk_img = ImageTk.PhotoImage(image=img)
            self._canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)
        if self._running:
            self._root.after(33, self._refresh_frame)

    def _start_emotion_thread(self):
        def _loop():
            while self._running:
                if self._emotion is None:
                    time.sleep(0.5)
                    continue
                with self._frame_lock:
                    frame = self._frame.copy() if self._frame is not None else None
                if frame is not None:
                    try:
                        result = self._emotion.predict(frame)
                        with self._emo_lock:
                            self._emotion_label = result.get("label") or "neutral"
                            self._emotion_face = result.get("face")
                    except Exception as exc:
                        logger.warning("Emotion error: %s", exc)
                time.sleep(0.45)
        threading.Thread(target=_loop, daemon=True, name="emotion").start()

    def _toggle_adaptive(self):
        with self._emo_lock:
            self._adaptive_on = not self._adaptive_on
            on = self._adaptive_on
        msg = "Emotion adaptive feedback on." if on else "Emotion adaptive feedback off."
        self._log(msg)
        self._tts.speak(msg, emotion="neutral", adaptive=False)

    def _speak(self, text: str) -> None:
        with self._emo_lock:
            emo = self._emotion_label
            on = self._adaptive_on
        self._tts.speak(text, emotion=emo, adaptive=on)

    def _update_pose_from_currency(self, frame):
        mode = self._modes[MODE_CURRENCY]
        bbox = getattr(mode, "last_bbox", None)
        name = getattr(mode, "last_class", None)
        if not bbox or not name:
            with self._emo_lock:
                self._pose = None
            return
        x, y, w, h = bbox
        try:
            from assistive import estimate_pose
            pose = estimate_pose(frame, (x, y, x + w, y + h), name)
        except Exception as exc:
            logger.warning("Pose error: %s", exc)
            pose = None
        with self._emo_lock:
            self._pose = pose
        needs = bool(pose and pose.get("needs_straighten"))
        if needs and not self._straighten_spoken:
            self._straighten_spoken = True
            self._speak("নোট সোজা করে ধরুন")
        elif not needs:
            self._straighten_spoken = False

    # ------------------------------------------------------------------
    # Object mode auto-scan
    # ------------------------------------------------------------------

    def _start_scan_thread(self):
        def _loop():
            while self._running:
                with self._lock:
                    mode = self._mode
                if mode == MODE_OBJECT:
                    with self._frame_lock:
                        frame = self._frame.copy() if self._frame is not None else None
                    if frame is not None and not self._detecting:
                        try:
                            result = self._modes[MODE_OBJECT].process_frame(frame)
                            if result:
                                self._root.after(0, self._show_result, result)
                                self._set_tts_status("speaking...")
                                self._speak(result)
                                self._set_tts_status("ready")
                        except Exception as exc:
                            logger.warning("Object scan error: %s", exc)
                    time.sleep(OBJECT_SCAN_INTERVAL)
                elif mode == MODE_CURRENCY:
                    with self._frame_lock:
                        frame = self._frame.copy() if self._frame is not None else None
                    if frame is not None and not self._detecting:
                        try:
                            hits = self._modes[MODE_CURRENCY].detect_live(frame)
                            with self._emo_lock:
                                self._currency_hits = hits
                            if hits:
                                top = hits[0]
                                self._update_pose_from_currency(frame)
                                spoken_key = (top["name"], top.get("auth"))
                                if spoken_key != self._currency_spoken and top["conf"] >= 0.45:
                                    self._currency_spoken = spoken_key
                                    self._root.after(0, self._show_result, top["text"])
                                    self._speak(top["text"])
                            else:
                                with self._emo_lock:
                                    self._pose = None
                                self._currency_spoken = None
                                self._straighten_spoken = False
                        except Exception as exc:
                            logger.warning("Currency scan error: %s", exc)
                    time.sleep(0.35)
                else:
                    with self._emo_lock:
                        self._currency_hits = []
                    time.sleep(0.2)

        threading.Thread(target=_loop, daemon=True, name="scan").start()

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _switch_mode(self):
        with self._lock:
            old = self._mode
            new = (self._mode + 1) % len(self._modes)
            self._mode = new
        self._tts.stop()
        self._modes[old].deactivate()
        self._modes[new].activate()
        self._mode_var.set(f"● {MODE_NAMES_EN[new]}")
        if new == MODE_CLAUDE:
            self._result_var.set("Press C or A — online Claude (needs internet + API key)")
        elif new == MODE_OCR:
            self._result_var.set("Press A for offline EasyOCR, or C for online Claude")
        elif new == MODE_CURRENCY:
            self._result_var.set("Hold the note — it says the value and whether it is real or jaal")
        else:
            self._result_var.set("Press A to detect")
        self._speak(MODE_NAMES_EN[new])
        self._log(f"Mode: {MODE_NAMES_EN[new]}")

    def _action(self):
        with self._lock:
            mode = self._mode
        with self._frame_lock:
            frame = self._frame.copy() if self._frame is not None else None
        if frame is None:
            self._speak("Camera not ready.")
            return
        if mode == MODE_OBJECT:
            self._modes[MODE_OBJECT].force_announce_now()
            return
        if self._detecting:
            return

        self._detecting = True
        self._result_var.set("⏳ Detecting…")
        self._set_tts_status("processing…")
        self._log(f"Running {MODE_NAMES_EN[mode]}…")

        def _infer():
            try:
                result = self._modes[mode].process_frame(frame)
                msg    = result if result else "Nothing detected."
                if mode == MODE_CURRENCY:
                    self._update_pose_from_currency(frame)
            except Exception as exc:
                msg = f"Error: {exc}"
                logger.error("Inference error: %s", exc)
            finally:
                self._detecting = False
            self._root.after(0, self._show_result, msg)
            self._set_tts_status("speaking…")
            self._speak(msg)
            self._set_tts_status("ready")

        threading.Thread(target=_infer, daemon=True, name="action").start()

    def _claude_action(self):
        """C key — always send the current frame to Claude (online).

        Works from any mode so EasyOCR (A in Text Reading) stays offline.
        """
        with self._frame_lock:
            frame = self._frame.copy() if self._frame is not None else None
        if frame is None:
            self._speak("Camera not ready.")
            return
        if self._detecting:
            return

        with self._lock:
            old = self._mode
            self._mode = MODE_CLAUDE
        if old != MODE_CLAUDE:
            self._tts.stop()
            self._modes[old].deactivate()
            self._modes[MODE_CLAUDE].activate()
            self._mode_var.set(f"● {MODE_NAMES_EN[MODE_CLAUDE]}")
            self._log(f"Mode: {MODE_NAMES_EN[MODE_CLAUDE]}")

        self._detecting = True
        self._result_var.set("⏳ Claude (online)…")
        self._set_tts_status("calling Claude…")
        self._log("Running Online Claude Mode…")

        def _infer():
            try:
                result = self._modes[MODE_CLAUDE].process_frame(frame)
                msg = result if result else "Nothing detected."
            except Exception as exc:
                msg = f"Error: {exc}"
                logger.error("Claude inference error: %s", exc)
            finally:
                self._detecting = False
            self._root.after(0, self._show_result, msg)
            self._set_tts_status("speaking…")
            self._speak(msg)
            self._set_tts_status("ready")

        threading.Thread(target=_infer, daemon=True, name="claude").start()

    def _read_stored(self):
        """R key — speak back the last EasyOCR or Claude result."""
        with self._lock:
            mode = self._mode
        if mode == MODE_OCR:
            result = self._modes[MODE_OCR].read_stored()
        elif mode == MODE_CLAUDE:
            result = self._modes[MODE_CLAUDE].read_stored()
        else:
            return

        if not result:
            return

        self._show_result(result)
        self._log("Reading back captured text…")
        self._set_tts_status("speaking…")
        self._speak(result)
        self._set_tts_status("ready")

    def _show_result(self, text: str):
        self._result_var.set(text)
        self._log(f"↳ {text[:120]}")

    def _vol_up(self):
        self._volume = min(100, self._volume + VOLUME_STEP)
        self._tts.set_volume(self._volume)
        self._vol_bar["value"] = self._volume
        self._vol_lbl["text"]  = f"{self._volume}%"

    def _vol_down(self):
        self._volume = max(0, self._volume - VOLUME_STEP)
        self._tts.set_volume(self._volume)
        self._vol_bar["value"] = self._volume
        self._vol_lbl["text"]  = f"{self._volume}%"

    def _quit(self):
        self._running = False
        self._tts.stop()
        self._modes[self._mode].deactivate()
        for m in self._modes:
            m.cleanup()
        if hasattr(self, "_cap"):
            self._cap.release()
        self._root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  Smart Glass — Windows Test (Piper / SAPI TTS)")
    print("=" * 55)
    print("  No internet required — Piper if installed, else Windows SAPI")
    print()

    root = tk.Tk()
    SmartGlassWin(root)
    root.mainloop()


if __name__ == "__main__":
    main()
