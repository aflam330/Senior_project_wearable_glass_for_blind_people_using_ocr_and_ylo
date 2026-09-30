# Savior Glass

Offline assistive smart glass for visually impaired users, targeting **Raspberry Pi 5**.

Three modes, physical GPIO buttons, and Bangla-first speech:

1. **OCR** - capture Bangla/English text, then speak it
2. **Object** - YOLOv8s everyday-object announcements (YOLOv8n if yolov8s.pt is absent)
3. **Currency** - Bangladeshi Taka denomination. On a laptop this uses `../realtime_bangla_taka_detection/models/best.pt`. On a Raspberry Pi 5 it uses `best_int8.onnx`, and the back-lit watermark check is on. The old "jaal" verdict stays off (`JAAL_VERDICT_ENABLED`); the safe check asks for a human look instead of saying counterfeit.
4. **Claude (online, optional)** - scene description through the Anthropic API; needs internet and `ANTHROPIC_API_KEY`

## Raspberry Pi

From `savior_glass/` on the Pi 5:

```bash
bash scripts/deploy_pi5.sh
python scripts/benchmark_pi5.py --iters 100 --sustained 30
python main.py
```

On that board, currency mode loads `best_int8.onnx` and the back-lit watermark check is on. Both stay off the laptop path. `scripts/pi5_preflight.py` confirms the model files before the first timing run. Raspberry Pi 5 latency is still unmeasured until that benchmark writes `results/pi5_benchmark_<host>_<time>.json`.

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
