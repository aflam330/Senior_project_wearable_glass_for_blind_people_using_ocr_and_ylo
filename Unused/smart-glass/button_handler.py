"""
GPIO Button Handler for Raspberry Pi 5.

Uses rpi-lgpio — a drop-in RPi.GPIO replacement that supports the RP1
GPIO chip found on RPi 5.  Install: pip install rpi-lgpio

Wiring (all buttons connect pin → GND; internal pull-ups enabled):
  GPIO 17 → MODE button   (cycle modes)
  GPIO 27 → ACTION button (capture frame / trigger detection)
  GPIO 24 → READ button   (speak back the last captured OCR text)
  GPIO 22 → VOL UP button
  GPIO 23 → VOL DOWN button
"""
import logging
import threading
from typing import Callable, Optional

try:
    import RPi.GPIO as GPIO          # provided by rpi-lgpio on RPi 5
    _GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    _GPIO_AVAILABLE = False

import config

logger = logging.getLogger("smart_glass.button_handler")


class ButtonHandler:
    """
    Manages four physical push-buttons via GPIO interrupts.
    All callbacks are invoked from the GPIO interrupt thread; keep them fast.
    """

    def __init__(self) -> None:
        self._mode_cb:     Optional[Callable] = None
        self._action_cb:   Optional[Callable] = None
        self._read_cb:     Optional[Callable] = None
        self._vol_up_cb:   Optional[Callable] = None
        self._vol_down_cb: Optional[Callable] = None

        # Per-button software debounce tracking (belt-and-suspenders on top of
        # the hardware bouncetime= parameter, which can be unreliable on lgpio)
        self._last_press: dict[int, float] = {
            config.BUTTON_MODE:     0.0,
            config.BUTTON_ACTION:   0.0,
            config.BUTTON_READ:     0.0,
            config.BUTTON_VOL_UP:   0.0,
            config.BUTTON_VOL_DOWN: 0.0,
        }
        self._debounce_s = config.BUTTON_DEBOUNCE_MS / 1000.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def register_callbacks(
        self,
        mode_cb:     Callable,
        action_cb:   Callable,
        read_cb:     Callable,
        vol_up_cb:   Callable,
        vol_down_cb: Callable,
    ) -> None:
        """Register application callbacks before calling setup()."""
        self._mode_cb     = mode_cb
        self._action_cb   = action_cb
        self._read_cb     = read_cb
        self._vol_up_cb   = vol_up_cb
        self._vol_down_cb = vol_down_cb

    def setup(self) -> None:
        if not _GPIO_AVAILABLE:
            logger.warning(
                "RPi.GPIO (rpi-lgpio) not available — buttons disabled. "
                "Install: pip install rpi-lgpio"
            )
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        pins = [
            config.BUTTON_MODE,
            config.BUTTON_ACTION,
            config.BUTTON_READ,
            config.BUTTON_VOL_UP,
            config.BUTTON_VOL_DOWN,
        ]
        for pin in pins:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(
                pin,
                GPIO.FALLING,
                callback=self._dispatch,
                bouncetime=config.BUTTON_DEBOUNCE_MS,
            )

        logger.info("GPIO buttons initialised on pins %s", pins)

    def cleanup(self) -> None:
        if _GPIO_AVAILABLE:
            try:
                GPIO.cleanup()
            except Exception:
                pass
        logger.info("GPIO cleaned up")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, *_):
        self.cleanup()

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, pin: int) -> None:
        import time
        now = time.time()
        with self._lock:
            if now - self._last_press.get(pin, 0.0) < self._debounce_s:
                return
            self._last_press[pin] = now

        cb_map = {
            config.BUTTON_MODE:     self._mode_cb,
            config.BUTTON_ACTION:   self._action_cb,
            config.BUTTON_READ:     self._read_cb,
            config.BUTTON_VOL_UP:   self._vol_up_cb,
            config.BUTTON_VOL_DOWN: self._vol_down_cb,
        }
        cb = cb_map.get(pin)
        if cb:
            try:
                cb()
            except Exception as exc:
                logger.error("Button callback error on pin %d: %s", pin, exc)
        else:
            logger.warning("Unregistered pin triggered: %d", pin)
