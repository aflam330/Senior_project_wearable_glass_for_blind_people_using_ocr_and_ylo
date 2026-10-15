# Test results (15 October 2026)

## Automated checks of the glass software (all run on the laptop, all passing)

| Script | What it checks | Result |
|---|---|---|
| `scripts/test_tts_queue.py` | speech queue, language splitting, render-then-play, prefetch, cache, voice fallback chain, WAV parsing, the "still working" tick, the real Bangla Piper voice | all checks pass |
| `scripts/test_voice_commands.py` | command parsing, dispatch to the button handlers, spoken command recognised, ordinary sentence ignored, wake word required | all checks pass |
| `scripts/test_thermal.py` | governor states, hysteresis, spike smoothing, firmware flag override, off without a sensor | all checks pass |
| `scripts/test_ocr_single_pass.py` | one detector pass per capture; identical text to EasyOCR on 16 images; empty picture answered at once | all checks pass |
| `scripts/test_multi_note.py` | counting several notes, overlap rule, threshold | all checks pass |
| `scripts/test_uv_check.py` | UV fibre check on real genuine photographs and edge cases | all checks pass |
| `scripts/test_capture_guide.py` | spoken guidance for the watermark check | all checks pass |
| `scripts/test_field_log.py` | field log under load (10,008 rows, none dropped; 1.9 µs per call) | all checks pass |
| `scripts/test_buttons_haptics.py` | every button path, vibration | 0 failures |
| `scripts/pi5_preflight.py` | every model file present and each ONNX graph runs | pass |
| fresh clone of the public repository | the app's tests run from a clean checkout | pass |

## Measured evaluations of this round (each chosen on validation, test read once)

| Evaluation | Note | Headline |
|---|---|---|
| Hand-held Taka detector | `HANDHELD_FINETUNE.md` | NSTU hand-held test 11.8 → 47.5 % right (Pi 8.7 → 33.5 %); Bangla Money 89.6 → 94.1 % |
| Voice commands in noise and whispering | `VOICE_COMMANDS_FAST.md` | 12 unseen voices: clean 100 %, whisper 98.9 %, street 0 dB 82.5 %, babble 10 dB 54.3 %; 13 false in 5,040 |
| Offline Bangla commands | `BANGLA_VOICE_COMMANDS.md` | 45 % / 62 % on unseen voices; not deployed |
| Real-photo text reading | `OCR_REAL_IMPROVED.md` | Bangla CER 39.2 → 28.7 %, Latin 62.5 → 50.6 % |
| Multi-note with the new detector | `HANDHELD_FINETUNE.md` | total right 97.0 % (was 99.2 %) |

## Not tested (needs the Pi, people or material not on this laptop)

- Every timing, temperature and power figure of the 14-15 October changes on the Raspberry Pi 5 (`scripts/pi_check_all.sh`, `scripts/power_log.py`).
- The voice heard by people; voice commands spoken by people into the glass's microphone.
- Real counterfeit notes under the UV lamp; a user study.
