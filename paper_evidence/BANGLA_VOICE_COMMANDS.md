# Offline Bangla voice commands: measured, not deployed (14 October 2026)

Scripts: `savior_glass/scripts/bangla_commands_study.py`, `scripts/bangla_commands_heldout.py`.
Results: `savior_glass/results/bangla_voice_commands.json`, `bangla_voice_commands_heldout.json`.

**Why Whisper.** Vosk (used for the English commands) has no Bangla model. Whisper's multilingual models include Bengali and run offline through faster-whisper (CTranslate2, 8-bit). Fifteen Bangla command phrases for the same eleven commands as the English list (e.g. "টাকা দেখো", "লেখা পড়ো", "থামো"); ten ordinary Bangla sentences count false commands. A command is accepted when the transcript is at least 0.6 similar (1 − edit distance / length) to a command phrase. Synthetic voices on a laptop CPU; not people, not the Pi.

## Step 1: Whisper alone, matched in Bangla script (Piper and Google Bangla voices, 15 phrases each)

| Model | Commands carried out | Laptop time per command |
|---|---:|---:|
| tiny | 0 / 30 | 0.9 s |
| base | 0 / 30 | 2.1 s |
| small | 0 / 30 | 9.5 s |
| small, command list as prompt | 1 / 30 (and 1 false command) | 32.7 s |

Tiny produced mixed-script nonsense. **Base heard many commands but wrote them in Latin letters** ("Taka Dekku", "Lekha Bodha"), which the Bangla-script matcher could not accept.

## Step 2: matching Latin spellings too, tested on voices never used

A fixed Latin spelling was added for every phrase ("taka dekho", "lekha poro", "thamo", ...). Because this was designed after seeing the step-1 outputs, the Piper and Google voices were only used as design voices (base: 18 / 30). "Small" was dropped beforehand for speed. The measurement was made once on four Microsoft neural voices never used before, two from Bangladesh (Nabanita, Pradeep) and two from India (Tanishaa, Bashkar):

| Held-out voices (Whisper base) | Carried out | Wrong command | Not heard | False commands |
|---|---:|---:|---:|---:|
| Clean | 27 / 60 (45 %) | 0 | 33 | 0 / 40 sentences |
| White noise, 10 dB SNR | 37 / 60 (62 %) | 0 | 23 | 0 / 40 sentences |

Median 1.7 s per command on the laptop CPU; on the Pi it will be several times that. Whisper still writes some Bangla commands in Urdu or Arabic script ("مک دکو" for "মুখ দেখো"), and hears "টাকা" once as "Thanks for watching.", so about half of the commands are not recognised.

## Decision

Not put on the glass. It never carried out a wrong command and never reacted to ordinary speech, which is the safe side, but a blind user would have to repeat every second command and wait several seconds each time. The English command words (Vosk, 99 % on synthetic voices) remain the voice interface. A dedicated Bangla keyword-spotting model trained on recordings of Bangladeshi speakers is what would close this gap; no such model of a size the Pi can run was found.
