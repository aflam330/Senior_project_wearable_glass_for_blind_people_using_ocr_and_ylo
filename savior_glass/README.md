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

## Voice and voice commands (14 October 2026)

```bash
pip install -r requirements-speech.txt
python scripts/get_speech_models.py     # voices and the command recogniser, about 210 MB, once
python scripts/speech_check.py          # speaks six sentences; prints which voice spoke and how long rendering took
```

**Voice.** Every sentence is rendered to audio first and then played, so heavy work on the processor cannot break the
sound up. Bangla and English use Piper voices when they are installed and espeak-ng otherwise. `TTS_VOICE=espeak`
forces espeak-ng; `PIPER_LENGTH_SCALE=1.15` slows the Piper voice down.

**Voice commands.** With a microphone plugged in, the glass listens for these English words (offline):

| Say | What happens |
|---|---|
| "read text", "read", "text" | text mode, one capture is read |
| "money", "check money", "taka" | currency mode, one note is checked |
| "objects", "find objects", "around me" | object mode |
| "describe", "describe scene" | online description (needs internet and a key) |
| "emotion", "face" | emotion mode, one face is read |
| "next mode" | MODE button |
| "capture", "scan", "check" | ACTION button |
| "repeat", "say again" | READ button |
| "louder", "quieter", "volume up", "volume down" | volume |
| "stop", "quiet" | stops the voice |

`VOICE_COMMANDS=0` turns listening off. `VOICE_WAKE_WORD=glass` makes every command start with "glass".
`VOICE_MIC=<part of the name>` picks a microphone. The buttons keep working.
Measured with synthetic voices on a laptop only (`results/voice_commands_synthetic.json`); not yet with people or on the Pi.

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
