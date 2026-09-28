"""Button and haptic logic test with a simulated RPi.GPIO (no Raspberry Pi needed).

A fake GPIO module records pin setup, lets the test fire button edges, and timestamps
every output write. The real SmartGlass app (main.py) runs with its real modes; only the
camera (serves a real 100-taka photo) and the speech engine (records text) are stubbed.
This tests the software path from button press to spoken result and vibration pattern.
Real switches, wiring, contact bounce and the motor itself need the Pi: NOT_MEASURED here.

Writes results/buttons_haptics_test.json.
"""
from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PHOTO = ROOT.parent / "data set" / "Bangladeshi_Paper_Currency_Raw" / "Bangladeshi_Paper_Currency_Raw" / "100"


class FakeGPIO(types.ModuleType):
    BCM, IN, OUT, PUD_UP, FALLING, HIGH, LOW = "BCM", "IN", "OUT", "PUD_UP", "FALLING", 1, 0

    def __init__(self):
        super().__init__("RPi.GPIO")
        self.setups, self.events, self.writes = {}, {}, []

    def setmode(self, mode): self.mode = mode
    def setwarnings(self, flag): pass
    def setup(self, pin, direction, pull_up_down=None): self.setups[pin] = (direction, pull_up_down)
    def add_event_detect(self, pin, edge, callback=None, bouncetime=None): self.events[pin] = (edge, callback, bouncetime)
    def output(self, pin, value): self.writes.append((time.perf_counter(), pin, value))
    def cleanup(self, *a): pass

    def press(self, pin):
        edge, cb, _ = self.events[pin]
        cb(pin)


class FakeTTS:
    def __init__(self, volume=80):
        self.spoken, self.volume = [], volume
    def speak(self, text, *a, **k): self.spoken.append((time.perf_counter(), text))
    def set_volume(self, pct): self.volume = pct
    def stop_current(self): pass
    def stop(self): pass
    def shutdown(self): pass


class FakeCamera:
    def __init__(self):
        self.frame = cv2.imread(str(sorted(PHOTO.glob("*.jpg"))[30]))
    def start(self): pass
    def stop(self): pass
    def get_frame(self): return self.frame.copy()


def pulses(writes, pin, t0):
    """(start_ms_after_t0, on_ms) for each HIGH..LOW on pin."""
    out, on = [], None
    for t, p, v in writes:
        if p != pin:
            continue
        if v == 1:
            on = t
        elif on is not None:
            out.append((round((on - t0) * 1000), round((t - on) * 1000)))
            on = None
    return out


def main() -> None:
    gpio = FakeGPIO()
    rpi = types.ModuleType("RPi")
    rpi.GPIO = gpio
    sys.modules["RPi"], sys.modules["RPi.GPIO"] = rpi, gpio

    import config
    import main as app_main
    app_main.utils.TTSEngine = FakeTTS
    app_main.utils.CameraManager = FakeCamera
    app = app_main.SmartGlass()
    app.start()
    tts = app._tts
    res = {"hardware": "NOT_MEASURED (simulated RPi.GPIO on Windows)", "checks": {}}
    chk = res["checks"]

    pins = [config.BUTTON_MODE, config.BUTTON_ACTION, config.BUTTON_READ, config.BUTTON_VOL_UP, config.BUTTON_VOL_DOWN]
    chk["pins_configured"] = {
        "pass": all(gpio.setups.get(p) == ("IN", "PUD_UP") and gpio.events[p][0] == "FALLING"
                    and gpio.events[p][2] == config.BUTTON_DEBOUNCE_MS for p in pins),
        "pins": pins, "debounce_ms": config.BUTTON_DEBOUNCE_MS}

    # mode cycling: OCR -> Object -> Currency -> Claude -> OCR, each announced
    seen = []
    for _ in range(4):
        time.sleep(0.3)  # past the 250 ms debounce
        gpio.press(config.BUTTON_MODE)
        seen.append((app._current_mode, tts.spoken[-1][1]))
    chk["mode_cycle"] = {"pass": [m for m, _ in seen] == [1, 2, 3, 0]
                         and all(s == config.MODE_NAMES_BN[m] for m, s in seen), "sequence": seen}

    # debounce: a second press 50 ms later is ignored
    time.sleep(0.3)
    before = app._current_mode
    gpio.press(config.BUTTON_MODE)
    time.sleep(0.05)
    gpio.press(config.BUTTON_MODE)
    chk["debounce"] = {"pass": app._current_mode == (before + 1) % 4, "mode_changes_from_two_presses_50ms_apart":
                       (app._current_mode - before) % 4}

    # volume: steps of 10, clamped to 0..100
    vols = []
    for pin, n in ((config.BUTTON_VOL_UP, 3), (config.BUTTON_VOL_DOWN, 12)):
        for _ in range(n):
            time.sleep(0.27)
            gpio.press(pin)
            vols.append(app._volume)
    chk["volume"] = {"pass": vols[:3] == [90, 100, 100] and vols[-1] == 0 and tts.volume == 0, "sequence": vols}

    # action press in currency mode: camera frame -> YOLO + jaal -> spoken text + vibration
    while app._current_mode != config.MODE_CURRENCY:
        time.sleep(0.3)
        gpio.press(config.BUTTON_MODE)
    time.sleep(0.3)
    n_spoken, n_writes = len(tts.spoken), len(gpio.writes)
    t0 = time.perf_counter()
    gpio.press(config.BUTTON_ACTION)
    while len(tts.spoken) == n_spoken and time.perf_counter() - t0 < 60:
        time.sleep(0.01)
    time.sleep(1.0)  # let the vibration pattern finish
    spoken_t, text = tts.spoken[-1]
    buzz = pulses(gpio.writes[n_writes:], config.HAPTIC_PIN, t0)
    verdict = "counterfeit" if "জাল" in text else ("genuine" if "আসল" in text else "none")
    expected = {"genuine": 2, "counterfeit": 3, "none": 1}[verdict]
    chk["action_currency"] = {
        "pass": text.startswith("একশত টাকার নোট") and len(buzz) == expected,
        "spoken": text, "verdict": verdict, "press_to_speech_ms": round((spoken_t - t0) * 1000),
        "vibration_pulses_ms[start,on]": buzz,
        "vibration_start_relative_to_speech_ms": (buzz[0][0] - round((spoken_t - t0) * 1000)) if buzz else None}

    # vibration patterns themselves
    cm = app._modes[config.MODE_CURRENCY]
    patterns = {}
    for name, n_exp, on_exp in (("genuine", 2, 80), ("counterfeit", 3, 220), ("detect", 1, 50)):
        k = len(gpio.writes)
        t1 = time.perf_counter()
        cm._buzz(name).join()
        p = pulses(gpio.writes[k:], config.HAPTIC_PIN, t1)
        patterns[name] = {"pass": len(p) == n_exp and all(abs(on - on_exp) <= 25 for _, on in p), "pulses": p}
    chk["haptic_patterns"] = patterns
    chk["intensity_control"] = {"pass": None, "note": "the glass drives the motor on/off (no PWM intensity); "
                                "roboeye.haptics uses PWM 80% duty. Not adjustable by the user."}
    chk["emergency_stop"] = {"pass": None, "note": "no emergency-stop button exists in the 5-button design"}

    app._running.clear()
    res["summary"] = {k: v["pass"] for k, v in chk.items() if isinstance(v, dict) and "pass" in v and k != "haptic_patterns"}
    res["summary"]["haptic_patterns"] = all(v["pass"] for v in patterns.values())
    (ROOT / "results" / "buttons_haptics_test.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
