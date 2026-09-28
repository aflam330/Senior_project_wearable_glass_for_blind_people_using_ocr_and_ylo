# GPIO buttons test

**Hardware: NOT_MEASURED** (no Raspberry Pi here). The button *software* was tested on the
laptop with a simulated `RPi.GPIO` module that records pin setup and fires button edges.

Script: `savior_glass/scripts/test_buttons_haptics.py`; raw:
`savior_glass/results/buttons_haptics_test.json`. The real `SmartGlass` app (`main.py`) ran
with its real modes; only the camera (a real 100-taka photo) and the speech engine (records
text) were stubbed.

| Check | Result |
|---|---|
| Pins 17, 27, 24, 22, 23 set as inputs with pull-up, falling-edge interrupts, 250 ms bounce time | PASS |
| MODE cycles OCR → Object → Currency → Claude → OCR and announces each in Bangla | PASS |
| Debounce: two MODE presses 50 ms apart change the mode once | PASS |
| VOL UP / VOL DOWN step by 10 and clamp at 100 and 0 (80 → 90 → 100 → 100 … → 0) | PASS |
| ACTION in currency mode: camera frame → Taka YOLO + jaal → spoken "একশত টাকার নোট। আসল" | PASS, 989 ms press-to-speech |
| Emergency stop | not applicable: the 5-button design has no emergency-stop button |

Not covered: real switches, wiring, contact bounce and interrupt latency on the Pi 5 (rpi-lgpio).
Run `main.py` on the Pi with the buttons wired as in `button_handler.py` to measure those.
