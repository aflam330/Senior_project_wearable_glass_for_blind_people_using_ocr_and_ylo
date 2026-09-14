"""Haptic vibration patterns for genuine / counterfeit / pose prompts.

On Windows: XInput gamepad rumble if a controller is present, plus an
on-screen HUD pulse (and optional winsound click).
On Raspberry Pi: GPIO PWM on HAPTIC_GPIO_PIN (vibration motor).
"""

from __future__ import annotations

import ctypes
import threading
import time

from .config import HAPTIC_GPIO_PIN, HAPTIC_PATTERNS

try:
    import winsound
except Exception:
    winsound = None


def _try_gpio_pwm():
    try:
        import RPi.GPIO as GPIO

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(HAPTIC_GPIO_PIN, GPIO.OUT)
        pwm = GPIO.PWM(HAPTIC_GPIO_PIN, 120)
        pwm.start(0)
        return GPIO, pwm
    except Exception:
        return None, None


class _XInputVibration(ctypes.Structure):
    _fields_ = [("wLeftMotorSpeed", ctypes.c_ushort), ("wRightMotorSpeed", ctypes.c_ushort)]


def _xinput_set(left: int, right: int) -> bool:
    for dll_name in ("xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"):
        try:
            dll = ctypes.windll.LoadLibrary(dll_name)
            vib = _XInputVibration(left, right)
            # DWORD XInputSetState(DWORD dwUserIndex, XINPUT_VIBRATION*)
            err = dll.XInputSetState(0, ctypes.byref(vib))
            return err == 0
        except Exception:
            continue
    return False


class HapticEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._busy = False
        self.last_pattern = "idle"
        self._gpio, self._pwm = _try_gpio_pwm()
        self.backend = "gpio" if self._pwm is not None else "software"

    def play(self, name: str, also_beep: bool = True) -> None:
        pattern = HAPTIC_PATTERNS.get(name) or HAPTIC_PATTERNS["unknown"]
        self.last_pattern = name
        threading.Thread(target=self._run, args=(pattern, also_beep), daemon=True).start()

    def _run(self, pattern: list[tuple[int, int]], also_beep: bool) -> None:
        with self._lock:
            self._busy = True
            try:
                for on_ms, off_ms in pattern:
                    self._pulse(True)
                    if also_beep and winsound is not None:
                        try:
                            winsound.Beep(880 if on_ms < 120 else 440, max(30, min(on_ms, 200)))
                        except Exception:
                            pass
                    time.sleep(on_ms / 1000.0)
                    self._pulse(False)
                    time.sleep(off_ms / 1000.0)
            finally:
                self._pulse(False)
                self._busy = False

    def _pulse(self, on: bool) -> None:
        if self._pwm is not None:
            try:
                self._pwm.ChangeDutyCycle(80 if on else 0)
            except Exception:
                pass
        _xinput_set(40000 if on else 0, 40000 if on else 0)

    def for_auth(self, auth_label: str, conf: float) -> str:
        if conf < 0.45:
            name = "low_conf"
        elif auth_label == "counterfeit":
            name = "counterfeit"
        elif auth_label == "genuine":
            name = "genuine"
        else:
            name = "unknown"
        self.play(name)
        return name

    def close(self) -> None:
        self._pulse(False)
        if self._gpio is not None:
            try:
                self._gpio.cleanup(HAPTIC_GPIO_PIN)
            except Exception:
                pass
