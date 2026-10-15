# Voice commands: accuracy, noise, whispering and speed (15 October 2026)

Code: `savior_glass/voice_commands.py`, `voice_frontend.py`; evaluation `scripts/voice_corpus.py`, `scripts/voice_eval.py`; results `savior_glass/results/voice_eval.json` (decodes in `voice_eval_r2_*.pkl`, `voice_eval_test_*.pkl`).

## Test material (synthetic; not people)

- **Voices:** 20 Microsoft neural voices. **8 design voices** (US, GB, AU, IN, NG, SG, KE, PH) are the only ones any setting was chosen on. **12 held-out voices** (IN, US, GB, IE, ZA, PH, NZ, CA, and four **Bangla voices reading the English commands**: Nabanita and Pradeep from Bangladesh, Tanishaa and Bashkar from India) were used once, at the end.
- **Per voice:** the 39 command phrases (with "glass, " in front for the wake-word mode) and 30 ordinary sentences, several containing command words ("we should stop at the next station").
- **Noise:** babble (three other voices talking at once; separate design and test babble) and real environmental sound from ESC-50 (all non-animal classes: traffic, engines, rain, wind, crowds, household sounds; folds 1-4 for design, fold 5 for test), each at 10, 5, 0, −5 and −10 dB SNR.
- **Quiet speech:** soft voice (−20 dB); whispering, simulated by exciting each frame's spectral envelope (LPC order 16) with noise instead of the vocal folds, at −25 dB; whispering with babble at 10 dB.
- **Response time:** from the end of the speech to the end of the 0.1 s audio block in which the command was decided, plus the computing time spent on that block. Laptop CPU.

## What was compared (design voices only)

| Dimension | Options |
|---|---|
| Front end | none; noise filter (Wiener gain, minimum-statistics noise estimate); level control (automatic gain, up to +30 dB); both |
| Decision | at the end of speech; early, on a settled partial result that cannot grow into a longer phrase |
| End of speech | Vosk's own; Silero voice-activity detection after 8 or 12 silent 32 ms chunks |
| Word confidence | 0.7 / 0.6 / 0.5 |
| Wake word | none; "glass" before every command |
| Unknown words at the edges (wake mode) | reject; ignore |

Score: right − 3 × wrong − 3 × false commands, over all 14 conditions. Two rounds: round 1 (66 settings) showed that other people's speech is the hard case and that the wake word removes most false commands; round 2 (72 settings, wake word throughout) added the edge rule and a slower Silero setting.

**Chosen:** wake word "glass", unknown words at the edges ignored, confidence 0.5, Vosk's own end of speech, no front end.

## Held-out result (12 voices, measured once)

Each condition: 468 spoken commands, 360 ordinary sentences.

| Condition | Before (no wake word) | **Chosen** | Bangla voices only, chosen |
|---|---:|---:|---:|
| Clean | 99.2 % | **100 %** | 100 % |
| Soft voice (−20 dB) | 99.4 % | **99.8 %** | 99.4 % |
| Whisper (simulated) | 92.9 % | **98.9 %** | 99.4 % |
| Street and home sounds, 10 dB | 97.9 % | **98.5 %** | 98.7 % |
| — 5 dB | 92.5 % | **93.8 %** | 91.7 % |
| — 0 dB | 80.1 % | **82.5 %** | 80.8 % |
| — −5 dB | 59.8 % | 57.7 % | 50.6 % |
| — −10 dB | 41.7 % | 39.5 % | 30.1 % |
| Other people talking, 10 dB | 3.2 % | **54.3 %** | 54.5 % |
| — 5 dB | 0.4 % | 26.1 % | 28.2 % |
| — 0 dB and below | ≤ 0.2 % | ≤ 5.6 % | ≤ 5.1 % |
| Whisper with people talking, 10 dB | 33.6 % | **54.3 %** | 44.2 % |
| **False commands** (all conditions, of 5,040 sentences) | 212 | **13** | 7 of 1,680 |
| Wrong commands (all conditions) | 79 | 84 | 39 |
| Median response after the speech ends | 0.86 s | 0.86 s | 0.86 s |

## Reading

- **Quiet and whispered commands work** (98.9-100 %). A blind user can give commands softly in public.
- **Background sound** is handled down to 0 dB (82.5 %), where the noise is as loud as the voice. Below that, accuracy falls; −10 dB (noise ten times the voice's power) is not reached: 39.5 %.
- **Other people talking** was the real weakness: without the wake word the command grammar turned their speech into commands (212 false commands) and the user's own command was lost when their speech touched it. With "glass" and the edge rule, false commands fall to 13 in 5,040 sentences, and half the commands get through at 10 dB babble. Louder babble still defeats it.
- **Speed: the 200 ms target is not met, and cannot be met this way.** A command can only be decided once the speaker has stopped. Vosk waits about 0.75-0.85 s of silence. Silero end-of-speech detection answers in 0.37-0.48 s, but cut commands short at the pause after "glass" and lost more commands than it saved time (round 2, design voices, otherwise the chosen setting: 2,301 right with Silero after 12 silent chunks, 1,929 after 8, against 2,388 with Vosk's own), so it is not the default. Early decisions on partial results saved about 0.05 s and caused false commands. Most of the response time is this wait for the end of speech. `VOICE_CMD_ENDPOINT=vad` gives the faster, less accurate option.
- **The noise filter and level control did not help overall** on these voices (round 1 best with both: 1,958 against 2,040 without), except for whispering without a wake word. They stay available (`VOICE_CMD_FRONTEND`).

## Not done, and why

- **Porcupine** wake-word engine: needs a personal Picovoice access key; Vosk with "glass" in its grammar does the same job here.
- **Recordings of people:** none were made; every voice is synthetic. Bangladeshi speakers, the glass's microphone and a real street are not measured.
- **A trained command classifier, Whisper fine-tuning, RNNoise, echo cancellation, beamforming:** a classifier or fine-tune needs recordings of real speakers; the glass has one microphone (no beamforming); its own speech is not heard because the microphone is ignored while it speaks (no echo to cancel). The Wiener filter stands in for the classical denoisers and did not help.
- **Bangla command words:** see `BANGLA_VOICE_COMMANDS.md` (45-62 % on unseen voices; not deployed).
