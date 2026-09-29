# Savior Glass

Offline assistive smart glass for visually impaired users, targeting **Raspberry Pi 5**.

Three modes, physical GPIO buttons, and Bangla-first speech:

1. **OCR** - capture Bangla/English text, then speak it
2. **Object** - YOLOv8s everyday-object announcements (YOLOv8n if yolov8s.pt is absent)
3. **Currency** - Bangladeshi Taka denomination with the trained YOLOv8s (../realtime_bangla_taka_detection/models/best.pt); HSV + optional CNN only if those weights are missing. The genuine/jaal verdict is off (`JAAL_VERDICT_ENABLED` in config.py; see ../paper_evidence/JAAL_VERDICT_FIXED.md).
4. **Claude (online, optional)** - scene description through the Anthropic API; needs internet and `ANTHROPIC_API_KEY`

## Raspberry Pi

`ash
chmod +x install.sh
./install.sh
sudo systemctl enable --now smart_glass.service
`

Run without systemd: python3 main.py

### Measuring speed on the Pi 5

Raspberry Pi 5 latency has not been measured yet. With the glass venv active, on the Pi:

```bash
python3 scripts/benchmark_pi5.py                 # every mode, ~5 min
python3 scripts/benchmark_pi5.py --sustained 30  # also a 30-minute mixed run: temperature, throttling
```

It times each mode's real `process_frame()` (currency, jaal check, object, OCR, emotion)
and the Taka detector as PT, ONNX and INT8 ONNX, and records CPU temperature,
`vcgencmd get_throttled` and memory. Results go to `results/pi5_benchmark_<host>_<time>.json`;
a run on any other machine is labelled `raspberry_pi_5: false`.

## Windows (no GPIO)

`powershell
python test_windows.py
`

or run_windows_test.bat

## Layout

`
savior_glass/
  main.py              # Pi entry point
  config.py
  button_handler.py    # GPIO (rpi-lgpio)
  modes/               # ocr_mode, object_mode, currency_mode
  install.sh
  smart_glass.service
  test_windows.py
  assets/labels_bn.json
`

Full system write-up: ../docs/PAPER2_Smart_Glass_System.md
