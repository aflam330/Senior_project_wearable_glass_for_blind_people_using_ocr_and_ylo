# Final verdict on the two requests of 14-15 October 2026

## What is fixed, with the measured result

| Area | Before | Now | Evidence |
|---|---|---|---|
| Hand-held notes (laptop / Pi) | 11.8 % / 8.7 % right | **47.5 % / 33.5 %** | `HANDHELD_FINETUNE.md` |
| Other people's whole-note photos (laptop) | 89.6 %, 50 wrong | **94.1 %, 17 wrong** | same |
| Text on real photos (Bangla / Latin CER) | 39.2 % / 62.5 % | **28.7 % / 50.6 %** | `OCR_REAL_IMPROVED.md` |
| Voice commands, clean / whispered (unseen voices) | 99.2 % / 92.9 % | **100 % / 98.9 %** | `VOICE_COMMANDS_FAST.md` |
| Voice commands with people talking (10 dB) | 3.2 % | **54.3 %** | same |
| False voice commands (5,040 sentences) | 212 | **13** | same |
| Broken voice on the Pi | render and play at once; English 38 % fast | render first; right sample rate; Bangla Piper voice | `VOICE_2026-10-14.md`, `BUGS_FIXED.md` |
| Fine-tune crash | out of memory, progress lost | completed; resume and memory rules | `CRASH_FIXED.md` |
| Pi heat | no control | governor for background work; cause (cooling at idle) identified | `THERMAL.md` |

## What got worse, and is kept anyway

- One-taka notes (not a class) are announced as a value more often: 22.8 → 36.6 % (laptop), 5.0 → 8.9 % (Pi).
- Three notes in one picture: spoken total right 99.2 → 97.0 % (never too many).
- Hand-held answers on the Pi: one in seven is a wrong value.

## Targets not reached, and why

| Target | Reached | Why not |
|---|---|---|
| Voice response under 200 ms | **No**: 0.86 s median | A command can only be decided after the speaker stops; the recogniser waits ~0.8 s. Faster end-of-speech detection (0.37-0.48 s) lost more commands than it saved |
| Voice commands at −10 dB noise | **No**: 39.5 % | Noise ten times the voice's power; 82.5 % at 0 dB |
| 95 % voice accuracy "in any condition" | **Clean, soft and whispered: yes. Loud noise or nearby talkers: no** | See the table above |
| Bangla voice commands | **No**: 45-62 % | No offline Bangla recogniser of Pi size; Whisper transcribes Bangla poorly |
| Pi below 80 °C | **Not measured** | Needs a Pi run; the board idled at 71 °C, which points at cooling first |
| Wake-word engine (Porcupine) | Not used | Needs a personal Picovoice key; Vosk with "glass" does the job |
| Real counterfeits, user study, online mode, Objects365 | No | Need material, people, credit or storage this laptop does not have |

## Venue probabilities

Not given. They cannot be measured; any number would be invented (`VENUE_READINESS.md`). What can be said: the hand-held, real-word OCR and voice results strengthen the application paper (IEEE Access) and the journal paper; all three rest on synthetic voices or other people's photos, and the Pi timings, the glass camera and a user study are still the missing evidence for every venue.

## To do on the Pi next

```
cd ~/"SARA UPDATE"/SARA && git pull
cd savior_glass && . ~/.venvs/savior_glass/bin/activate
pip install -r requirements-speech.txt && python scripts/get_speech_models.py
sh scripts/pi_check_all.sh                       # tests, voice timing, text-reading timing
python scripts/power_log.py --minutes 30 --label session &   python main.py
```
Then say "glass, check money" with a note held close to the camera.
