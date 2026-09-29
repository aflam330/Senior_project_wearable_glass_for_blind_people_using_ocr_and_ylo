"""
Smart Glass for Blind People — Main Entry Point
Raspberry Pi 5 | Fully Offline

Thread layout:
  main          — startup, shutdown, signal handling
  camera        — continuous frame capture (in CameraManager)
  tts-worker    — audio queue consumer (in TTSEngine)
  detection     — object mode auto-scanning loop
  [GPIO ISR]    — button callbacks (rpi-lgpio interrupt threads)

Usage:
  python3 main.py
"""
import logging
import os
import signal
import sys
import threading
import time
import warnings

# EasyOCR sets pin_memory=True internally; suppress the GPU-not-found noise
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)

import config
import utils
from button_handler import ButtonHandler
from modes import CurrencyMode, ObjectMode, OCRMode

logger = logging.getLogger("smart_glass.main")


class SmartGlass:

    def __init__(self) -> None:
        self._running  = threading.Event()
        self._running.set()

        # Shared resources
        self._tts    = utils.TTSEngine(volume=config.DEFAULT_VOLUME)
        self._camera = utils.CameraManager()

        # Modes — OCR loads lazily; YOLO loads on first activate
        self._modes = [OCRMode(), ObjectMode(), CurrencyMode()]
        self._current_mode: int = config.MODE_OCR
        self._mode_lock = threading.Lock()

        # Volume state
        self._volume = config.DEFAULT_VOLUME

        # Buttons
        self._buttons = ButtonHandler()

        # Guards a single OCR/currency inference at a time — EasyOCR/PyTorch
        # calls take seconds on RPi 5, and the shared Reader/classifier
        # isn't guaranteed safe to call concurrently from overlapping
        # button-press threads (which also produced overlapping/garbled TTS).
        self._inferring = threading.Event()

        # Detection thread
        self._detection_thread = threading.Thread(
            target=self._detection_loop, daemon=True, name="detection"
        )

    # ------------------------------------------------------------------
    # Application lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        logger.info("=== Smart Glass starting ===")
        self._tts.speak("স্মার্ট গ্লাস চালু হচ্ছে")   # "Smart glass starting"

        # Start camera
        try:
            self._camera.start()
        except RuntimeError as exc:
            logger.critical("Camera init failed: %s", exc)
            self._tts.speak("ক্যামেরা সংযুক্ত করুন এবং আবার চালু করুন")
            sys.exit(1)

        # Setup GPIO buttons
        self._buttons.register_callbacks(
            mode_cb     = self._on_mode_press,
            action_cb   = self._on_action_press,
            read_cb     = self._on_read_press,
            vol_up_cb   = self._on_vol_up,
            vol_down_cb = self._on_vol_down,
        )
        self._buttons.setup()

        # Activate initial mode (OCR)
        self._modes[self._current_mode].activate()

        # Start detection worker
        self._detection_thread.start()

        # Announce initial mode
        time.sleep(0.5)   # give TTS time to finish boot message
        self._tts.speak(config.MODE_NAMES_BN[self._current_mode])

        logger.info("Startup complete. Current mode: %s",
                    config.MODE_NAMES_BN[self._current_mode])

    def run(self) -> None:
        """Block until SIGINT / SIGTERM."""
        self.start()
        try:
            while self._running.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        self.shutdown()

    def shutdown(self) -> None:
        logger.info("Shutting down…")
        self._running.clear()
        self._tts.speak("বন্ধ হচ্ছে")   # "Shutting down"
        self._tts.drain(timeout=3.0)
        self._modes[self._current_mode].deactivate()
        for mode in self._modes:
            mode.cleanup()
        self._camera.stop()
        self._buttons.cleanup()
        logger.info("Shutdown complete")

    # ------------------------------------------------------------------
    # Button callbacks (called from GPIO interrupt threads)
    # ------------------------------------------------------------------

    def _on_mode_press(self) -> None:
        with self._mode_lock:
            old_mode = self._current_mode
            new_mode = (self._current_mode + 1) % len(self._modes)
            self._current_mode = new_mode

        self._tts.stop_current()
        self._modes[old_mode].deactivate()
        self._modes[new_mode].activate()
        self._tts.speak(config.MODE_NAMES_BN[new_mode])
        logger.info("Mode → %s", config.MODE_NAMES_BN[new_mode])

    def _on_action_press(self) -> None:
        with self._mode_lock:
            mode_idx = self._current_mode

        frame = self._camera.get_frame()
        if frame is None:
            self._tts.speak("ক্যামেরা প্রস্তুত নয়")
            return

        # In object mode, ACTION forces an immediate announce with no cooldown
        if mode_idx == config.MODE_OBJECT:
            obj_mode: ObjectMode = self._modes[config.MODE_OBJECT]  # type: ignore[assignment]
            obj_mode.force_announce_now()
            return

        # OCR and currency: run inference in a short-lived thread so we don't
        # block the GPIO ISR thread. Ignore the press if one is already
        # running — overlapping calls would race on the shared EasyOCR
        # Reader / classifier and queue overlapping, garbled TTS output.
        if self._inferring.is_set():
            logger.debug("Ignoring ACTION press — inference already in progress")
            return
        self._inferring.set()

        def _infer():
            try:
                result = self._modes[mode_idx].process_frame(frame)
                if result:
                    self._tts.speak(result)
            finally:
                self._inferring.clear()

        threading.Thread(target=_infer, daemon=True, name="action-infer").start()

    def _on_read_press(self) -> None:
        """READ button — speaks back the text most recently captured by
        ACTION in OCR mode. Capture and playback are separate steps so the
        user only has to hold the camera steady for the quick capture, not
        through a long read-out. No-op outside OCR mode."""
        with self._mode_lock:
            mode_idx = self._current_mode

        if mode_idx != config.MODE_OCR:
            return

        ocr_mode: OCRMode = self._modes[config.MODE_OCR]  # type: ignore[assignment]
        result = ocr_mode.read_stored()
        if result:
            self._tts.speak(result)

    def _on_vol_up(self) -> None:
        self._volume = min(100, self._volume + config.VOLUME_STEP)
        self._tts.set_volume(self._volume)
        logger.info("Volume → %d%%", self._volume)

    def _on_vol_down(self) -> None:
        self._volume = max(0, self._volume - config.VOLUME_STEP)
        self._tts.set_volume(self._volume)
        logger.info("Volume → %d%%", self._volume)

    # ------------------------------------------------------------------
    # Detection loop (object mode auto-scan)
    # ------------------------------------------------------------------

    def _detection_loop(self) -> None:
        """Runs continuous object detection when in object mode."""
        while self._running.is_set():
            with self._mode_lock:
                mode_idx = self._current_mode

            if mode_idx == config.MODE_OBJECT:
                frame = self._camera.get_frame()
                if frame is not None:
                    try:
                        result = self._modes[config.MODE_OBJECT].process_frame(frame)
                        if result:
                            self._tts.speak(result)
                    except Exception as exc:
                        logger.warning("Object detection error: %s", exc)
                time.sleep(config.OBJECT_SCAN_INTERVAL)
            else:
                time.sleep(config.DETECTION_THREAD_SLEEP)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    utils.setup_logging()
    app = SmartGlass()

    # Graceful shutdown on SIGTERM (used by systemd)
    def _sigterm(_sig, _frame):
        logger.info("SIGTERM received")
        app._running.clear()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT,  _sigterm)

    app.run()


if __name__ == "__main__":
    main()
