"""
Windows Development / Testing Script
=====================================
Tests all three Smart Glass modes on Windows using:
  - Webcam via OpenCV
  - tkinter window for live camera display
  - Piper TTS for natural offline speech (falls back to Windows SAPI if not installed)
  - Keyboard keys instead of GPIO buttons

Keyboard controls (click the window first):
  M  ->  Cycle mode    (OCR -> Object -> Currency)
  A  ->  CAPTURE       (OCR: snap + OCR + store text in session, doesn't speak it yet
                        | Object/Currency: detect and announce immediately)
  R  ->  READ          (OCR mode only: speak back the text captured by the last A press)
  +  ->  Volume up
  -  ->  Volume down
  Q  ->  Quit

Install:
  pip install easyocr ultralytics torch torchvision numpy Pillow
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

MODE_OCR, MODE_OBJECT, MODE_CURRENCY = 0, 1, 2
MODE_NAMES_EN = ["Text Reading Mode", "Object Detection Mode", "Currency Detection Mode"]
MODE_NAMES_BN = [
    "টেক্সট রিডিং মোড",
    "অবজেক্ট ডিটেকশন মোড",
    "কারেন্সি ডিটেকশন মোড",
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


# ---------------------------------------------------------------------------
# TTS Engine — Windows SAPI (fully offline, no internet required)
# ---------------------------------------------------------------------------

class WindowsTTS:
    """
    Offline TTS via Windows SAPI (System.Speech.Synthesis.SpeechSynthesizer).
    Runs in a background worker thread. English text is spoken; Bangla text
    is shown on screen (SAPI has no built-in Bangla voice).
    """

    def __init__(self, volume: int = 85):
        self._volume     = max(0, min(100, volume))
        self._q: queue.Queue = queue.Queue(maxsize=5)
        self._stop_event = threading.Event()
        self._proc       = None
        self._proc_lock  = threading.Lock()

        self._worker = threading.Thread(
            target=self._run, daemon=True, name="tts-worker"
        )
        self._worker.start()
        logger.info("[TTS] Offline SAPI worker started")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        if not text or not text.strip():
            return

        stripped     = text.replace(" ", "")
        bangla_ratio = len(_BANGLA_RE.findall(stripped)) / max(len(stripped), 1)
        lang         = "bn" if bangla_ratio > 0.25 else "en"

        logger.info("[TTS] Queuing (%s): %s", lang, text[:70])

        if self._q.full():
            try:
                self._q.get_nowait()
                self._q.task_done()
            except queue.Empty:
                pass
        self._q.put_nowait((text.strip(), lang))

    def stop(self) -> None:
        self._stop_event.set()
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
        while True:
            text, lang = self._q.get()
            try:
                if os.path.isfile(PIPER_EXE):
                    self._piper_speak(text, lang)
                else:
                    self._sapi_speak(text, lang)
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
            logger.warning("[TTS] Piper model not found for lang=%s, falling back to SAPI", lang)
            self._sapi_speak(text, lang)
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
            self._sapi_speak(text, lang)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Windows SAPI backend (offline fallback)
    # ------------------------------------------------------------------

    def _sapi_speak(self, text: str, lang: str) -> None:
        # SAPI has no Bangla voice — speak a short English cue instead so
        # the developer knows text was captured; Bangla content is visible
        # in the result label and log box.
        speak_text = text if lang == "en" else "Bangla text captured. See screen."
        safe = speak_text.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Volume = {self._volume}; $s.Rate = 1; $s.Speak('{safe}');"
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
    return [OCRMode(), ObjectMode(), CurrencyMode()]


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

        self._build_ui()
        self._start_camera()
        self._start_scan_thread()

        self._modes[self._mode].activate()
        self._tts.speak("Smart Glass ready. Text Reading Mode.")
        self._log("Ready — Text Reading Mode | M=switch  A=capture  R=read  +/-=vol  Q=quit")

        root.bind("<Key-m>",     lambda e: self._switch_mode())
        root.bind("<Key-M>",     lambda e: self._switch_mode())
        root.bind("<Key-a>",     lambda e: self._action())
        root.bind("<Key-A>",     lambda e: self._action())
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

        self._tts_var = tk.StringVar(value="🔊 ready")
        tk.Label(info_frame, textvariable=self._tts_var,
                 font=("Segoe UI", 10), fg="#aaaaaa", bg="#1a1a2e"
                 ).pack(side=tk.RIGHT)

        # Result area — Nirmala UI renders Bangla
        self._result_var = tk.StringVar(value="Press A to read text or detect objects")
        tk.Label(
            self._root, textvariable=self._result_var,
            font=("Nirmala UI", 13), fg="#ffffff", bg="#1a1a2e",
            wraplength=630, justify="left",
        ).pack(padx=10, pady=(4, 2))

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
            text="  M = Mode    A = Capture / Detect    R = Read captured text    + / - = Volume    Q = Quit  ",
            font=("Segoe UI", 9), fg="#555", bg="#0d0d1e",
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
            frame = self._frame
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            self._tk_img = ImageTk.PhotoImage(image=img)
            self._canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)
        if self._running:
            self._root.after(33, self._refresh_frame)

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
                                self._tts.speak(result)
                                self._set_tts_status("ready")
                        except Exception as exc:
                            logger.warning("Object scan error: %s", exc)
                    time.sleep(OBJECT_SCAN_INTERVAL)
                else:
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
        self._result_var.set("Press A to detect")
        self._tts.speak(MODE_NAMES_EN[new])
        self._log(f"Mode: {MODE_NAMES_EN[new]}")

    def _action(self):
        with self._lock:
            mode = self._mode
        with self._frame_lock:
            frame = self._frame.copy() if self._frame is not None else None
        if frame is None:
            self._tts.speak("Camera not ready.")
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
            except Exception as exc:
                msg = f"Error: {exc}"
                logger.error("Inference error: %s", exc)
            finally:
                self._detecting = False
            self._root.after(0, self._show_result, msg)
            self._set_tts_status("speaking…")
            self._tts.speak(msg)
            self._set_tts_status("ready")

        threading.Thread(target=_infer, daemon=True, name="action").start()

    def _read_stored(self):
        """R key — speak back the text captured by the last A press in OCR
        mode. Capture and playback are separate steps so you only have to
        hold the camera steady for the quick capture, not the whole read-out."""
        with self._lock:
            mode = self._mode
        if mode != MODE_OCR:
            return

        ocr_mode = self._modes[MODE_OCR]
        result = ocr_mode.read_stored()
        if not result:
            return

        self._show_result(result)
        self._log("Reading back captured text…")
        self._set_tts_status("speaking…")
        self._tts.speak(result)
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
