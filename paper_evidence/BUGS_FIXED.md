# Bugs found and fixed (13-15 October 2026)

Each row is a defect in code that ran (or would run) on the glass, with how it was found and how the fix was checked. Static check: `pyflakes` over every app module and script found no undefined names or syntax errors; unused imports and variables were removed where they were leftovers (`queue` in utils.py, `time` in thermal.py, `Optional` in ekush_letters.py, `io`/`PIL.Image` in make_closeups.py). Three imports are kept on purpose: they test whether an optional package is installed (`pytesseract`, `vosk`, `onnxruntime`).

| # | Bug | Effect for the user | Found by | Fix | Checked by |
|---|---|---|---|---|---|
| 1 | Speech synthesised and played at the same time (espeak-ng live; Piper piped into aplay) | Voice broke up on the Pi while text reading used all cores | User report + code reading | Each sentence rendered fully, then played; next sentence rendered during playback | `scripts/test_tts_queue.py` (render before play, prefetch, cache) |
| 2 | English Piper voice (16 kHz) played at 22,050 Hz | English 38 % too fast and too high | Reading the voice's .json against the player command | Player started at the voice's own sample rate | same test (player command carries the rate) |
| 3 | First syllable lost on HDMI/USB/Bluetooth sound devices | Words clipped | Known device wake-up behaviour | 150 ms lead-in, 80 ms tail of silence | same test (lengths) |
| 4 | Bangla Piper voice could not be used (prebuilt binary crashes) | Robotic espeak-ng Bangla | 1 Oct Pi run | Piper's Python package, binary and espeak-ng as fallbacks; failures remembered per run | test 7 of the speech test; Bangla voice rendered on the laptop |
| 5 | Text detector ran twice for short signs | 34 s instead of 18 s on the Pi | 13 Oct Pi report | One detector pass per capture | `scripts/test_ocr_single_pass.py` (identical text on 16 images) |
| 6 | Empty picture: 27 s and 48 invented letters | False text spoken | 13 Oct Pi report | No text boxes -> "no text found" at once, handwriting reader skipped | same test |
| 7 | Silence during a 15-35 s capture | User cannot tell the press was taken | Code reading | "পড়ছি" at once, soft tick every 4 s, through the speech queue | speech test 11 |
| 8 | Installer used the wrong folder and a Camera Module 3 overlay; boot service ran as user "pi" in a folder that does not exist | Fresh install would not start at boot | Code reading against the Pi layout | `install.sh`, `scripts/install_service.sh` (user service, real paths) | shell syntax check; fresh clone |
| 9 | Hand-held fine-tune crashed (out of memory) and lost progress | — (research) | Training log | workers=2, one heavy job at a time, resume from the last epoch | `CRASH_FIXED.md`; the run completed 12 epochs |
| 10 | Evaluation looked for checkpoints in the wrong folder | Selection would have seen only the deployed model | Directory listing | Path to where Ultralytics writes | selection run found all 12 epochs |
| 11 | PDF and slide builders kept only in a temporary folder that was emptied | Deliverables could not be rebuilt | Missing files | Recovered from the session record into `Thesis Report and paper/tools/`; rebuilt PDFs identical in text | text comparison of old and rebuilt PDFs |
| 12 | Voice-command listener would have heard the "still working" tick as speech | Commands cut off during text reading | Code reading | The tick is not counted as speech | speech test 11 |
| 13 | A LaTeX edit made through a shell here-document turned "\\r" into a carriage return | Thesis build errors | Build log | Edits written as files, control characters scanned before every build | clean build, 119 pages |

## Not bugs, but found and recorded

- The 6.7 % watermark false-counterfeit figure in the request has no source in the result files (`WEAKNESS_FIXES_2026-10-14.md`, row 18).
- With other people talking nearby (babble at 10 dB), the English command grammar turns their speech into commands; see `VOICE_COMMANDS_FAST.md`.
