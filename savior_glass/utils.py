"""
Shared utilities: TTS engine, camera manager, and logging setup.
"""
import collections
import logging
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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

    # Only letters decide the language. Punctuation (also the Bangla full stop "।", which is outside the Bangla
    # block), digits and symbols belong to the run they stand in. Before this, every comma and full stop in a
    # Bangla sentence was cut out as a one-character "English" piece: a new voice process and a pause each time,
    # which is what made Bangla reading break up.
    first = next((("bn" if _BANGLA_RE.match(c) else "en") for c in text if _BANGLA_RE.match(c) or (c.isascii() and c.isalpha())), "en")
    for ch in text:
        if _BANGLA_RE.match(ch):
            ch_lang = "bn"
        elif ch.isascii() and ch.isalpha():
            ch_lang = "en"
        else:
            ch_lang = current_lang or first

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
    Non-blocking speech through a background worker thread.

    Every piece is rendered to audio first and played afterwards. Before 14 October 2026 espeak-ng synthesised and
    played at the same time, and Piper was piped straight into the player: when text reading held all four cores of
    the Pi, the player ran out of samples and the voice broke up. A finished buffer needs almost no processor time
    to play, so it cannot be starved that way.

    Voice, per language (config.TTS_VOICE): Piper through its Python package (the model stays loaded), then the
    Piper binary, then espeak-ng. Whatever fails is remembered for the run and not tried again.
    """

    _CACHE_MAX_CHARS = 60       # fixed prompts (mode names, "please wait") are short; their audio is kept
    _CACHE_MAX_ITEMS = 64

    def __init__(self, volume: int = config.DEFAULT_VOLUME):
        self._volume = volume
        # One entry per speak() call: (pieces, keep). A long text is one entry, so it can never push its own first
        # sentences out of the queue (before, the queue held 5 pieces and a page lost its beginning).
        self._msgs: collections.deque = collections.deque()
        self._cv = threading.Condition()
        self._busy = False
        self._talking = False
        self._gen = 0                       # raised by stop_current(): the message being spoken is abandoned
        self._piper_failed: set = set()     # (engine, model) pairs that could not speak in this run (skipped afterwards)
        self._voices: dict = {}             # Piper voices loaded in this process
        self._voice_lock = threading.Lock()
        self._cache: collections.OrderedDict = collections.OrderedDict()   # (text, lang) -> (pcm, rate)
        self._ahead: dict = {}              # (text, lang) -> Future: the next piece, rendered while this one plays
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-render")
        self._espeak_missing = False
        self._current_proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True, name="tts-worker")
        self._worker.start()
        self.set_volume(volume)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str, lang: str = "auto", keep: bool = False) -> None:
        """
        Enqueue text for speech. Non-blocking. When more than TTS_QUEUE_MAXSIZE messages wait, the oldest one
        that was not marked keep=True is dropped (object announcements go stale; the lines of a page do not).

        When lang="auto", mixed Bangla/English text is split into
        script-homogeneous runs (and each run further into sentences) so
        every chunk is read by the correct voice, in order, with natural
        pauses — rather than one block read entirely in the wrong voice.
        """
        text = (text or "").strip()
        if not text:
            return

        if lang == "auto":
            pieces = [(s, seg_lang) for seg_text, seg_lang in split_language_segments(text)
                      for s in split_sentences(seg_text)]
        else:
            pieces = [(s, lang) for s in split_sentences(text)]
        # a piece without a letter or digit (a stray symbol) is not worth a voice process and a pause
        pieces = [(s, l) for s, l in pieces if any(c.isalnum() for c in s)]
        if not pieces:
            return
        with self._cv:
            self._msgs.append((pieces, keep))
            if sum(1 for _, k in self._msgs if not k) > config.TTS_QUEUE_MAXSIZE:
                for i, (_, k) in enumerate(self._msgs):
                    if not k:
                        del self._msgs[i]          # drop the oldest message that may go stale
                        break
            self._cv.notify()

    def stop_current(self) -> None:
        """Interrupt currently playing audio immediately."""
        with self._cv:
            self._gen += 1                 # the rest of the message being spoken is abandoned
            self._msgs.clear()             # and nothing stale is replayed
        self._ahead.clear()
        with self._lock:
            if self._current_proc and self._current_proc.poll() is None:
                self._current_proc.terminate()
                self._current_proc = None
        if sys.platform == "win32":
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:  # noqa: BLE001
                pass

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
        while (self._msgs or self._busy) and time.time() < deadline:
            time.sleep(0.05)

    @property
    def speaking(self) -> bool:
        """True while something is being said or waits to be said (the voice-command listener stays deaf then)."""
        return bool(self._talking or any(p[0][1] != "tone" for p, _ in list(self._msgs)))

    def tick(self) -> None:
        """A soft short tone: "still working". Goes through the same queue as speech, so it never plays over a
        sentence, and is skipped when anything is being said or waits to be said."""
        with self._cv:
            if self._busy or self._msgs:
                return
            self._msgs.append(([("tick", "tone")], False))
            self._cv.notify()

    def prewarm(self, texts) -> None:
        """Load the voices and render fixed prompts in the background, so the first words are not late."""
        def work():
            for t in texts:
                for s, l in split_language_segments(t):
                    try:
                        self._render(s, l)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("prewarm failed for %r: %s", s, exc)
        threading.Thread(target=work, daemon=True, name="tts-prewarm").start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._msgs:
                    self._cv.wait()
                pieces, _ = self._msgs.popleft()
                gen, self._busy = self._gen, True
                self._talking = pieces[0][1] != "tone"      # the "still working" tick is not speech
            try:
                for i, (text, lang) in enumerate(pieces):
                    if gen != self._gen:           # stop_current() was called: drop the rest of this message
                        break
                    if i + 1 < len(pieces):        # the next sentence is rendered while this one is heard
                        self._prefetch(*pieces[i + 1])
                    self._synthesize(text, lang)
                    # Brief pause between consecutive utterances — without this,
                    # back-to-back pieces run together and sound like one
                    # rushed, unintelligible block.
                    time.sleep(config.TTS_INTER_UTTERANCE_PAUSE)
            except Exception as exc:
                logger.warning("TTS error: %s", exc)
            finally:
                self._busy = self._talking = False

    def _prefetch(self, text: str, lang: str) -> None:
        if (text, lang) not in self._cache and (text, lang) not in self._ahead:
            self._ahead[(text, lang)] = self._pool.submit(self._render, text, lang)

    def _synthesize(self, text: str, lang: str) -> None:
        """Render one piece (or take it from the piece rendered ahead) and play the finished audio."""
        fut = self._ahead.pop((text, lang), None)
        pcm, rate = fut.result() if fut is not None else self._render(text, lang)
        if pcm:
            self._play(pcm, rate)

    # -- rendering -----------------------------------------------------

    def _render(self, text: str, lang: str) -> tuple[bytes, int]:
        """Text -> (16-bit mono PCM, sample rate), with the best voice that works for this language."""
        key = (text, lang)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if lang == "tone":
            import math
            import struct
            rate, n = 22050, int(22050 * 0.07)
            out = (b"".join(struct.pack("<h", int(5000 * math.sin(2 * math.pi * 660 * i / rate) * min(1.0, i / 200, (n - i) / 200)))
                            for i in range(n)), rate)
            self._cache[key] = out
            return out
        if lang == "bn":
            model_path, fallback_voice = config.PIPER_BN_MODEL, config.ESPEAK_VOICE_BN
        else:
            model_path, fallback_voice = config.PIPER_EN_MODEL, config.ESPEAK_VOICE_EN
        out = None
        if config.TTS_VOICE != "espeak":
            for name, fn in (("piper", self._render_piper), ("piper-binary", self._render_piper_binary)):
                if (name, model_path) in self._piper_failed:
                    continue
                try:
                    out = fn(text, model_path)
                except Exception as exc:  # noqa: BLE001
                    # The prebuilt Pi Piper binary crashes on the Bangla voice. Trying it again before every sentence
                    # cost a model load and a crash each time, so each engine is tried once per run.
                    self._piper_failed.add((name, model_path))
                    logger.warning("%s cannot speak with %s (%s); the next voice is used from now on",
                                   name, os.path.basename(model_path), exc)
                    out = None
                if out and out[0]:
                    break
        if not (out and out[0]):
            out = self._render_espeak(text, fallback_voice)
        if out[0] and len(text) <= self._CACHE_MAX_CHARS:
            self._cache[key] = out
            while len(self._cache) > self._CACHE_MAX_ITEMS:
                self._cache.popitem(last=False)
        return out

    def _render_piper(self, text: str, model_path: str):
        """Piper through its Python package (pip install piper-tts). None when the package or the model is absent."""
        if not os.path.isfile(model_path) or not os.path.isfile(model_path + ".json"):
            return None
        with self._voice_lock:
            voice = self._voices.get(model_path)
            if voice is None:
                try:
                    from piper import PiperVoice
                except ImportError:
                    self._piper_failed.add(("piper", model_path))
                    return None
                voice = self._voices[model_path] = PiperVoice.load(model_path)
                logger.info("Piper voice loaded: %s", os.path.basename(model_path))
            if hasattr(voice, "synthesize_stream_raw"):           # piper-tts before 1.3
                pcm = b"".join(voice.synthesize_stream_raw(text, length_scale=config.PIPER_LENGTH_SCALE))
            else:
                from piper import SynthesisConfig
                cfg = SynthesisConfig(length_scale=config.PIPER_LENGTH_SCALE)
                pcm = b"".join(c.audio_int16_bytes for c in voice.synthesize(text, syn_config=cfg))
            return pcm, int(voice.config.sample_rate)

    def _render_piper_binary(self, text: str, model_path: str):
        """The stand-alone Piper program (models/piper/engine/piper). None when it or the model is absent."""
        if not (os.path.isfile(model_path) and os.path.isfile(config.PIPER_BINARY)):
            return None
        rate = 22050
        try:
            import json
            with open(model_path + ".json", encoding="utf-8") as f:
                rate = int(json.load(f)["audio"]["sample_rate"])     # "low" voices are 16 kHz, not 22.05 kHz
        except Exception:  # noqa: BLE001
            pass
        proc = subprocess.run([config.PIPER_BINARY, "--model", model_path, "--output-raw"],
                              input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60)
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError(f"piper exited {proc.returncode}")
        return proc.stdout, rate

    def _render_espeak(self, text: str, voice: str) -> tuple[bytes, int]:
        if self._espeak_missing:
            return b"", 0
        try:
            proc = subprocess.run(
                [
                    "espeak-ng",
                    "-v", voice,
                    "-s", str(config.ESPEAK_SPEED),
                    "-p", str(config.ESPEAK_PITCH),
                    "-g", str(config.ESPEAK_WORD_GAP),
                    "-a", str(int(self._volume * 2)),  # espeak amplitude 0-200
                    "--stdout",
                    text,
                ],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60,
            )
        except FileNotFoundError:
            self._espeak_missing = True
            logger.error("espeak-ng not found. Install: sudo apt install espeak-ng")
            return b"", 0
        return wav_pcm(proc.stdout)

    # -- playing -------------------------------------------------------

    _player_cmd: list[str] | None = None  # cached across instances once probed

    @classmethod
    def _audio_player_cmd(cls) -> list[str]:
        """Pick a working raw-PCM player once per process. The sample rate is appended by the caller.

        ``aplay`` against the ALSA "default" device is the common case, but on a
        headless Pi with no display attached, PipeWire has nothing to route the
        HDMI ALSA device to and ``aplay`` fails to open it — a failure that would
        otherwise silently repeat on every single utterance. ``pw-cat`` talks to
        PipeWire directly and still works (its dummy/virtual sink paces playback
        in real time). Probe once with a near-silent buffer and cache the
        first one that actually opens the device. The half-second device buffer
        lets playback ride over moments when the processor is busy.
        """
        if cls._player_cmd is not None:
            return cls._player_cmd
        candidates = [
            ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-c", "1", "--buffer-time=500000", "-r"],
            ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-c", "1", "-r"],
            ["pw-cat", "--raw", "--playback", "--format", "s16", "--channels", "1", "--rate"],
        ]
        for cmd in candidates:
            try:
                probe = subprocess.run(
                    cmd + ["22050", "-"], input=b"\x00\x00" * 50,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3,
                )
                if probe.returncode == 0:
                    cls._player_cmd = cmd
                    return cmd
            except Exception:
                continue
        cls._player_cmd = candidates[1]
        return candidates[1]

    def _play(self, pcm: bytes, rate: int) -> None:
        """Play finished audio. A short silence goes first: many sound devices (HDMI, USB, Bluetooth) need a moment
        to wake up and would swallow the first syllable; a little silence at the end keeps the last one whole."""
        pcm = (b"\x00\x00" * int(rate * config.TTS_LEAD_IN_MS / 1000) + pcm
               + b"\x00\x00" * int(rate * config.TTS_TAIL_MS / 1000))
        if sys.platform == "win32":
            import io
            import wave
            import winsound
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(pcm)
            winsound.PlaySound(buf.getvalue(), winsound.SND_MEMORY)
            return
        try:
            proc = subprocess.Popen(self._audio_player_cmd() + [str(rate), "-"], stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            logger.error("No audio player found. Install: sudo apt install alsa-utils")
            return
        with self._lock:
            self._current_proc = proc
        try:
            proc.stdin.write(pcm)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass                                   # stopped by stop_current()
        finally:
            proc.wait()
            with self._lock:
                self._current_proc = None


def wav_pcm(data: bytes) -> tuple[bytes, int]:
    """(samples, sample rate) from a WAV held in memory. espeak-ng writes a header without the true length when it
    writes to a pipe, so the header is read by hand instead of with the wave module."""
    i = data.find(b"data")
    if len(data) < 44 or data[:4] != b"RIFF" or i < 0:
        return b"", 0
    return data[i + 8:], int.from_bytes(data[24:28], "little")


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
                if self._ribbon_camera_present():
                    # the camera exists but could not be started: almost always another process has it open
                    raise RuntimeError(
                        f"The Pi Camera is present but could not be started ({exc}). Another program is probably using it. "
                        "Stop it with:  pkill -f main.py ; pkill -f rpicam   then start again."
                    ) from exc
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
        ribbon = self._ribbon_cameras()
        if not ribbon:
            raise RuntimeError("no ribbon (CSI) camera detected")
        cam = Picamera2(ribbon[0]["Num"])
        # "RGB888" gives arrays in BGR order, which is what OpenCV and the models expect
        cam.configure(cam.create_video_configuration(main={"size": (self._width, self._height), "format": "RGB888"}))
        cam.start()
        self._picam = cam

    @staticmethod
    def _ribbon_cameras() -> list:
        """Ribbon (CSI) cameras only. libcamera also lists USB webcams (UVC), but picamera2 hands their
        frames over with red and blue swapped, so webcams are left to OpenCV, which gives correct BGR."""
        try:
            from picamera2 import Picamera2
            return [c for c in Picamera2.global_camera_info() if "usb" not in str(c.get("Id", "")).lower()]
        except Exception:  # noqa: BLE001
            return []

    @classmethod
    def _ribbon_camera_present(cls) -> bool:
        return bool(cls._ribbon_cameras())

    @staticmethod
    def _is_real_capture_node(idx: int) -> bool:
        """False for the Pi's internal image-pipeline nodes (pispbe, rp1-cfe, codecs): they open but are not cameras."""
        try:
            with open(f"/sys/class/video4linux/video{idx}/name", encoding="utf-8") as f:
                name = f.read().strip().lower()
        except OSError:
            return True   # not Linux, or no sysfs entry: let OpenCV decide
        return not any(k in name for k in ("pisp", "rp1-cfe", "bcm2835-codec", "bcm2835-isp", "hevc", "rpivid"))

    def _candidate_indices(self) -> list:
        first = [self._index] if self._is_real_capture_node(self._index) else []
        try:
            import glob
            nodes = sorted(int(p.rsplit("video", 1)[1]) for p in glob.glob("/dev/video*") if p.rsplit("video", 1)[1].isdigit())
        except Exception:  # noqa: BLE001
            nodes = []
        rest = [i for i in (nodes or list(range(1, 4))) if i != self._index and self._is_real_capture_node(i)]
        return first + rest[:8]

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
