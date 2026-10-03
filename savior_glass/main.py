"""
Smart Glass for Blind People — Main Entry Point
Raspberry Pi 5 | offline except the optional Claude mode (internet + API key)

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
from modes import ClaudeMode, CurrencyMode, EmotionMode, ObjectMode, OCRMode

logger = logging.getLogger("smart_glass.main")

MODE_KEYS = {config.MODE_OCR: "ocr", config.MODE_OBJECT: "object", config.MODE_CURRENCY: "currency",
             config.MODE_CLAUDE: "claude", config.MODE_EMOTION: "emotion"}


def _flog(event, mode_idx=None, **fields):
    """Field-test log (field_log.py). Cheap: puts one row on a queue; never raises into the app."""
    try:
        import field_log
        return field_log.get().log(event, MODE_KEYS.get(mode_idx, "") if mode_idx is not None else "", **fields)
    except Exception:  # noqa: BLE001
        return 0


def _result_fields(mode, mode_idx):
    """What each mode decided, for the field log."""
    if mode_idx == config.MODE_CURRENCY:
        hits = getattr(mode, "last_hits", None) or []
        if hits:
            h = hits[0]
            return {"denomination": h.get("name", ""), "confidence": h.get("conf"), "verdict": h.get("auth", ""),
                    "score": h.get("genuine_prob"), "boxes": len(hits)}
        return {"denomination": "none"}
    if mode_idx == config.MODE_OCR:
        return {"lang": getattr(mode, "_stored_lang", ""), "ocr_text": getattr(mode, "_stored_text", "") or ""}
    if mode_idx == config.MODE_EMOTION:
        last = getattr(mode, "last", None) or {}
        return {"verdict": last.get("label", ""), "confidence": last.get("prob"), "face_found": last.get("face") is not None,
                "backend": last.get("backend", "")}
    return {}


class SmartGlass:

    def __init__(self) -> None:
        self._running  = threading.Event()
        self._running.set()

        # Shared resources
        self._tts    = utils.TTSEngine(volume=config.DEFAULT_VOLUME)
        self._camera = utils.CameraManager()

        # Modes — OCR loads lazily; YOLO loads on first activate
        self._modes = [OCRMode(), ObjectMode(), CurrencyMode(), ClaudeMode(), EmotionMode()]
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

        # Model loading runs off the button thread. A mode's event is set once its models are loaded
        # and warmed up; ACTION waits for it instead of racing a half-loaded model.
        self._ready = [threading.Event() for _ in self._modes]
        self._prep_locks = [threading.Lock() for _ in self._modes]

        # Detection thread
        self._detection_thread = threading.Thread(
            target=self._detection_loop, daemon=True, name="detection"
        )

    # ------------------------------------------------------------------
    # Application lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        logger.info("=== Smart Glass starting ===")
        _flog("session_start", self._current_mode)
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
        try:
            self._buttons.setup()
        except RuntimeError as exc:
            logger.critical("%s", exc)
            self._camera.stop()
            self._buttons.cleanup()
            sys.exit(1)

        # Activate initial mode (OCR)
        self._prepare(self._current_mode)
        # Preload currency mode in the background so the first switch to it is quick
        if self._current_mode != config.MODE_CURRENCY:
            threading.Thread(target=self._prepare, args=(config.MODE_CURRENCY, False), daemon=True,
                             name="preload-currency").start()

        # Start detection worker
        self._detection_thread.start()

        # Announce initial mode
        time.sleep(0.5)   # give TTS time to finish boot message
        self._tts.speak(config.MODE_NAMES_BN[self._current_mode])

        logger.info("Startup complete. Current mode: %s",
                    config.MODE_NAMES_BN[self._current_mode])

    def run(self) -> None:
        """Block until SIGINT / SIGTERM (or Q in the test preview window)."""
        import preview
        view = None
        if preview.enabled():
            try:
                view = preview.Preview(self)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Preview window unavailable: %s", exc)
        self.start()
        try:
            while self._running.is_set():
                if view is None:
                    time.sleep(0.2)
                    continue
                try:
                    if not view.tick():
                        break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Preview window failed, continuing without it: %s", exc)
                    view = None
        except KeyboardInterrupt:
            pass
        if view is not None:
            view.close()
        self.shutdown()

    def shutdown(self) -> None:
        try:
            import field_log
            field_log.get().close()
        except Exception:  # noqa: BLE001
            pass
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

    def _prepare(self, idx: int, activate: bool = True) -> None:
        """Load (activate) and warm up one mode, then mark it ready. Safe to call twice."""
        mode = self._modes[idx]
        with self._prep_locks[idx]:
            try:
                if activate or not self._ready[idx].is_set():
                    mode.activate()
                if not self._ready[idx].is_set() and hasattr(mode, "warm_up"):
                    mode.warm_up()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Preparing mode %d failed: %s", idx, exc)
            finally:
                self._ready[idx].set()

    def _on_mode_press(self) -> None:
        with self._mode_lock:
            old_mode = self._current_mode
            new_mode = (self._current_mode + 1) % len(self._modes)
            self._current_mode = new_mode

        self._tts.stop_current()
        self._modes[old_mode].deactivate()
        # say the new mode at once; loading happens in the background
        self._tts.speak(config.MODE_NAMES_BN[new_mode])
        _flog("mode_switch", new_mode, result=MODE_KEYS.get(new_mode, ""), from_mode=MODE_KEYS.get(old_mode, ""))
        threading.Thread(target=self._prepare, args=(new_mode,), daemon=True, name=f"prepare-{new_mode}").start()
        logger.info("Mode → %s", config.MODE_NAMES_BN[new_mode])

    def _on_action_press(self) -> None:
        with self._mode_lock:
            mode_idx = self._current_mode

        frame = self._camera.get_frame()
        if frame is None:
            self._tts.speak("ক্যামেরা প্রস্তুত নয়")
            _flog("camera_not_ready", mode_idx)
            return

        # In object mode, ACTION forces an immediate announce with no cooldown
        if mode_idx == config.MODE_OBJECT:
            obj_mode: ObjectMode = self._modes[config.MODE_OBJECT]  # type: ignore[assignment]
            obj_mode.force_announce_now()
            _flog("action_press", mode_idx, frame=frame)
            return

        # OCR and currency: run inference in a short-lived thread so we don't
        # block the GPIO ISR thread. Ignore the press if one is already
        # running — overlapping calls would race on the shared EasyOCR
        # Reader / classifier and queue overlapping, garbled TTS output.
        if self._inferring.is_set():
            logger.debug("Ignoring ACTION press — inference already in progress")
            _flog("action_ignored_busy", mode_idx)
            return
        self._inferring.set()

        t_press = time.perf_counter()
        eid = _flog("action_press", mode_idx, frame=frame)

        def _infer():
            try:
                waited = 0.0
                if not self._ready[mode_idx].is_set():
                    self._tts.speak("একটু অপেক্ষা করুন")  # please wait a moment (models still loading)
                    t_wait = time.perf_counter()
                    self._ready[mode_idx].wait(timeout=120)
                    waited = (time.perf_counter() - t_wait) * 1000
                mode = self._modes[mode_idx]
                t0 = time.perf_counter()
                result = mode.process_frame(frame)
                infer_ms = (time.perf_counter() - t0) * 1000
                if result:
                    self._tts.speak(result)
                _flog("result", mode_idx, latency_ms=infer_ms, result=result or "", press_event=eid,
                      press_to_result_ms=round((time.perf_counter() - t_press) * 1000, 1),
                      waited_for_load_ms=round(waited, 1), **_result_fields(mode, mode_idx))
                # 500 / 1,000 Taka: guided back-lit watermark check (config.CAPTURE_GUIDE_ENABLED)
                if mode_idx == config.MODE_CURRENCY and mode.wants_guided_watermark():
                    ev = mode.guided_watermark(self._camera.get_frame, self._tts.speak) or {}
                    _flog("watermark_check", mode_idx, latency_ms=1000 * float(ev.get("seconds") or 0),
                          result=ev.get("outcome", "failed"), denomination=ev.get("denomination", ""),
                          verdict=ev.get("outcome", ""), score=ev.get("watermark_prob"), press_event=eid,
                          frames=ev.get("frames"), rejected=ev.get("rejected"), condition=ev.get("condition"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Inference failed: %s", exc)
                _flog("error", mode_idx, result=f"{type(exc).__name__}: {exc}", press_event=eid)
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

        if mode_idx == config.MODE_OCR:
            ocr_mode: OCRMode = self._modes[config.MODE_OCR]  # type: ignore[assignment]
            result = ocr_mode.read_stored()
        elif mode_idx == config.MODE_CLAUDE:
            claude_mode: ClaudeMode = self._modes[config.MODE_CLAUDE]  # type: ignore[assignment]
            result = claude_mode.read_stored()
        else:
            return
        if result:
            self._tts.speak(result)
        _flog("read_press", mode_idx, result=result or "")

    def _on_vol_up(self) -> None:
        self._volume = min(100, self._volume + config.VOLUME_STEP)
        self._tts.set_volume(self._volume)
        logger.info("Volume → %d%%", self._volume)
        _flog("volume", result=str(self._volume))

    def _on_vol_down(self) -> None:
        self._volume = max(0, self._volume - config.VOLUME_STEP)
        self._tts.set_volume(self._volume)
        logger.info("Volume → %d%%", self._volume)
        _flog("volume", result=str(self._volume))

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
                        t0 = time.perf_counter()
                        result = self._modes[config.MODE_OBJECT].process_frame(frame)
                        if result:
                            self._tts.speak(result)
                            _flog("object_announce", config.MODE_OBJECT, latency_ms=(time.perf_counter() - t0) * 1000,
                                  result=result)
                    except Exception as exc:
                        logger.warning("Object detection error: %s", exc)
                        _flog("error", config.MODE_OBJECT, result=f"{type(exc).__name__}: {exc}")
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
