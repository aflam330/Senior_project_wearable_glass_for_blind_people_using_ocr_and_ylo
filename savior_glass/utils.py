"""
Shared utilities: TTS engine, camera manager, and logging setup.
"""
import collections
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from logging.handlers import RotatingFileHandler

import cv2

import config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    logger = logging.getLogger("smart_glass")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = logging.getLogger("smart_glass.utils")


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

# Bangla Unicode block: U+0980–U+09FF
_BANGLA_RE = re.compile(r"[ঀ-৿]")

# Split on sentence-ending punctuation (Bangla দাঁড়ি '।' included) so long
# OCR results are spoken sentence-by-sentence with natural pauses instead
# of one rushed, hard-to-follow block.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[।.!?])\s+")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return parts or ([text] if text.strip() else [])


def detect_language(text: str) -> str:
    """Return 'bn' if >25% of non-space characters are Bangla, else 'en'."""
    stripped = text.replace(" ", "")
    if not stripped:
        return "en"
    bangla_count = len(_BANGLA_RE.findall(stripped))
    return "bn" if (bangla_count / len(stripped)) > 0.25 else "en"


def split_language_segments(text: str) -> list[tuple[str, str]]:
    """
    Split mixed Bangla/English text into ordered (segment, lang) runs.

    Speaking a mixed-script sentence as a single block forces the wrong
    voice onto half of it (e.g. English words read with the Bangla
    espeak-ng voice), which is what made OCR results sound garbled.
    Splitting on script boundaries lets each run be queued and spoken
    with its own voice while preserving the original reading order.
    """
    segments: list[tuple[str, str]] = []
    current_lang: str | None = None
    current_chars: list[str] = []

    for ch in text:
        if _BANGLA_RE.match(ch):
            ch_lang = "bn"
        elif ch.isspace():
            ch_lang = current_lang or "en"
        else:
            ch_lang = "en"

        if current_lang is None:
            current_lang = ch_lang

        if ch_lang != current_lang:
            run = "".join(current_chars).strip()
            if run:
                segments.append((run, current_lang))
            current_chars = [ch]
            current_lang = ch_lang
        else:
            current_chars.append(ch)

    if current_chars:
        run = "".join(current_chars).strip()
        if run:
            segments.append((run, current_lang or "en"))

    return segments


# ---------------------------------------------------------------------------
# TTSEngine
# ---------------------------------------------------------------------------

class TTSEngine:
    """
    Non-blocking TTS via a background worker thread.
    Priority order: Piper TTS (if model present) → espeak-ng fallback.
    """

    def __init__(self, volume: int = config.DEFAULT_VOLUME):
        self._volume = volume
        self._queue: queue.Queue = queue.Queue(maxsize=config.TTS_QUEUE_MAXSIZE)
        self._current_proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True, name="tts-worker")
        self._worker.start()
        self.set_volume(volume)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str, lang: str = "auto") -> None:
        """
        Enqueue text for speech. Non-blocking; drops oldest if full.

        When lang="auto", mixed Bangla/English text is split into
        script-homogeneous runs (and each run further into sentences) so
        every chunk is read by the correct voice, in order, with natural
        pauses — rather than one block read entirely in the wrong voice.
        """
        text = (text or "").strip()
        if not text:
            return

        if lang == "auto":
            for seg_text, seg_lang in split_language_segments(text):
                for sentence in split_sentences(seg_text):
                    self._enqueue(sentence, seg_lang)
        else:
            for sentence in split_sentences(text):
                self._enqueue(sentence, lang)

    def _enqueue(self, text: str, lang: str) -> None:
        try:
            self._queue.put_nowait((text, lang))
        except queue.Full:
            try:
                self._queue.get_nowait()   # drop oldest
            except queue.Empty:
                pass
            self._queue.put_nowait((text, lang))

    def stop_current(self) -> None:
        """Interrupt currently playing audio immediately."""
        with self._lock:
            if self._current_proc and self._current_proc.poll() is None:
                self._current_proc.terminate()
                self._current_proc = None
        # Drain any queued items so we don't replay stale speech
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def set_volume(self, pct: int) -> None:
        pct = max(0, min(100, pct))
        self._volume = pct
        try:
            subprocess.run(
                ["amixer", "sset", "Master", f"{pct}%"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError:
            pass  # amixer not available (dev machine)

    def drain(self, timeout: float = 5.0) -> None:
        """Block until the audio queue is empty (max timeout seconds)."""
        deadline = time.time() + timeout
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            text, lang = self._queue.get()
            try:
                self._synthesize(text, lang)
                # Brief pause between consecutive utterances — without this,
                # back-to-back queue items run together and sound like one
                # rushed, unintelligible block.
                time.sleep(config.TTS_INTER_UTTERANCE_PAUSE)
            except Exception as exc:
                logger.warning("TTS error: %s", exc)
            finally:
                self._queue.task_done()

    def _synthesize(self, text: str, lang: str) -> None:
        """Choose Piper or espeak-ng and play audio.

        Both languages try Piper first (natural-sounding neural voice) and
        fall back to espeak-ng only if the matching Piper model isn't
        installed. Previously Bangla always used espeak-ng even when a
        Piper Bangla model was configured, which is why Bangla sounded
        noticeably more robotic/unclear than English.
        """
        if lang == "bn":
            model_path, fallback_voice = config.PIPER_BN_MODEL, config.ESPEAK_VOICE_BN
        else:
            model_path, fallback_voice = config.PIPER_EN_MODEL, config.ESPEAK_VOICE_EN

        if os.path.isfile(model_path) and os.path.isfile(config.PIPER_BINARY):
            self._piper(text, model_path, fallback_voice)
        else:
            self._espeak(text, fallback_voice)

    _player_cmd: list[str] | None = None  # cached across instances once probed

    @classmethod
    def _audio_player_cmd(cls) -> list[str]:
        """Pick a working raw-PCM player once per process.

        ``aplay`` against the ALSA "default" device is the common case, but on a
        headless Pi with no display attached, PipeWire has nothing to route the
        HDMI ALSA device to and ``aplay`` fails to open it — a failure that would
        otherwise silently repeat on every single utterance. ``pw-cat`` talks to
        PipeWire directly and still works (its dummy/virtual sink paces playback
        in real time). Probe both once with a near-silent buffer and cache the
        first one that actually opens the device.
        """
        if cls._player_cmd is not None:
            return cls._player_cmd
        candidates = [
            ["aplay", "-q", "-r", "22050", "-f", "S16_LE", "-c", "1", "-"],
            ["pw-cat", "--raw", "--playback", "--format", "s16", "--rate", "22050", "--channels", "1", "-"],
        ]
        for cmd in candidates:
            try:
                probe = subprocess.run(
                    cmd, input=b"\x00\x00" * 50,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3,
                )
                if probe.returncode == 0:
                    cls._player_cmd = cmd
                    return cmd
            except Exception:
                continue
        cls._player_cmd = candidates[0]
        return candidates[0]

    def _piper(self, text: str, model_path: str, fallback_voice: str) -> None:
        """Render via Piper and stream raw PCM to whichever player actually works."""
        try:
            piper_proc = subprocess.Popen(
                [config.PIPER_BINARY, "--model", model_path, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            aplay_proc = subprocess.Popen(
                self._audio_player_cmd(),
                stdin=piper_proc.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            piper_proc.stdin.write(text.encode("utf-8"))
            piper_proc.stdin.close()

            with self._lock:
                self._current_proc = aplay_proc

            aplay_proc.wait()
            piper_proc.wait()
            # A crashed Piper (e.g. a voice with phonemes this binary's id map lacks)
            # leaves aplay/pw-cat with nothing to play; both exit, but silently —
            # neither raises on its own. Without this check the utterance is just
            # dropped instead of falling back to espeak-ng.
            if piper_proc.returncode != 0:
                raise RuntimeError(f"piper exited {piper_proc.returncode}")
        except Exception as exc:
            logger.debug("Piper failed (%s), falling back to espeak-ng", exc)
            self._espeak(text, fallback_voice)
        finally:
            with self._lock:
                self._current_proc = None

    def _espeak(self, text: str, voice: str) -> None:
        try:
            proc = subprocess.Popen(
                [
                    "espeak-ng",
                    "-v", voice,
                    "-s", str(config.ESPEAK_SPEED),
                    "-p", str(config.ESPEAK_PITCH),
                    "-g", str(config.ESPEAK_WORD_GAP),
                    "-a", str(int(self._volume * 2)),  # espeak amplitude 0-200
                    text,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self._lock:
                self._current_proc = proc
            proc.wait()
        except FileNotFoundError:
            logger.error("espeak-ng not found. Install: sudo apt install espeak-ng")
        finally:
            with self._lock:
                self._current_proc = None


# ---------------------------------------------------------------------------
# CameraManager
# ---------------------------------------------------------------------------

class CameraManager:
    """
    Grabs frames in a background thread into a small ring buffer.
    Callers get the freshest available frame without blocking the main loop.
    """

    def __init__(
        self,
        index: int = config.CAMERA_INDEX,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
        fps: int = config.CAMERA_FPS,
    ):
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: cv2.VideoCapture | None = None
        self._picam = None   # Picamera2 object when the ribbon camera is used
        self._buffer: collections.deque = collections.deque(
            maxlen=config.FRAME_BUFFER_SIZE
        )
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Open the camera. Order (config.CAMERA_BACKEND = auto): the Pi Camera on the ribbon cable through
        picamera2, then OpenCV on CAMERA_INDEX, then the other video devices until one delivers frames.
        On a Pi 5, OpenCV can open a ribbon camera's /dev/video0 but never receives a frame from it."""
        backend = str(getattr(config, "CAMERA_BACKEND", "auto")).lower()
        errors = []
        if backend in ("auto", "picamera2"):
            try:
                self._start_picamera2()
            except Exception as exc:  # noqa: BLE001  (not installed, no ribbon camera, or a library mismatch)
                errors.append(f"picamera2: {exc}")
                self._picam = None
        if self._picam is None and backend in ("auto", "opencv"):
            for idx in self._candidate_indices():
                cap = self._open_cv(idx)
                if cap is not None:
                    self._cap, self._index = cap, idx
                    break
                errors.append(f"opencv index {idx}: no frames")
        if self._picam is None and self._cap is None:
            raise RuntimeError("No working camera. Tried " + "; ".join(errors) + ". Run: python scripts/check_camera.py")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="camera")
        self._thread.start()
        timeout = time.time() + 5.0
        while not self._buffer and time.time() < timeout:
            time.sleep(0.05)
        if not self._buffer:
            raise RuntimeError("Camera opened but no frames received within 5 s. Run: python scripts/check_camera.py")
        logger.info("Camera started (%s): %dx%d @ %d fps", "picamera2" if self._picam is not None else f"opencv index {self._index}",
                    self._width, self._height, self._fps)

    def _start_picamera2(self) -> None:
        from picamera2 import Picamera2
        if not Picamera2.global_camera_info():
            raise RuntimeError("no ribbon (CSI) camera detected")
        cam = Picamera2()
        # "RGB888" gives arrays in BGR order, which is what OpenCV and the models expect
        cam.configure(cam.create_video_configuration(main={"size": (self._width, self._height), "format": "RGB888"}))
        cam.start()
        self._picam = cam

    def _candidate_indices(self) -> list:
        first = [self._index]
        try:
            import glob
            nodes = sorted(int(p.rsplit("video", 1)[1]) for p in glob.glob("/dev/video*") if p.rsplit("video", 1)[1].isdigit())
        except Exception:  # noqa: BLE001
            nodes = []
        return first + [i for i in (nodes or list(range(1, 4))) if i != self._index][:12]

    def _open_cv(self, idx: int):
        """An opened capture that has already delivered a frame, or None."""
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # USB cameras: full frame rate at 640x480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimise latency
        deadline = time.time() + 2.5
        while time.time() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                return cap
            time.sleep(0.05)
        cap.release()
        return None

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:  # noqa: BLE001
                pass

    def get_frame(self):
        """Return the latest frame (numpy array) or None if not ready."""
        with self._lock:
            return self._buffer[-1].copy() if self._buffer else None

    def _capture_loop(self) -> None:
        while self._running:
            if self._picam is not None:
                try:
                    frame = self._picam.capture_array()
                    ret = frame is not None
                except Exception:  # noqa: BLE001
                    ret, frame = False, None
            else:
                ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._buffer.append(frame)
            else:
                time.sleep(0.01)
