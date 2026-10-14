# Weaknesses: status at the end of 14 October 2026

Follows `WEAKNESS_STATUS_2026-10-13.md`. A number appears only if it is in a result file. "Pi" means measured on the Raspberry Pi 5; everything else is the laptop.

## Fixed or improved today

| Weakness | What was done | Evidence | On the Pi? |
|---|---|---|---|
| Voice broken on the Pi | Each sentence rendered first, then played; player at the voice's own sample rate (English was 38 % too fast); silence before each sentence; next sentence prepared during playback | `VOICE_2026-10-14.md`; `scripts/test_tts_queue.py` | **Not yet heard** |
| Robotic Bangla voice (espeak-ng) | Piper Bangla voice through its Python package; espeak-ng fallback | 2 of 6 sentences recognised exactly, 2 nearly, by a machine listener (`speech_piper_bn_intelligibility.json`) | **Render time unknown** |
| Buttons only | Offline voice commands (Vosk, 39 phrases, 11 commands) | 99.1 % / 99.1 % / 96.6 % carried out (clean / 20 dB / 10 dB), no wrong command (`voice_commands_synthetic.json`, synthetic voices) | No; no people tested |
| Real-photo text reading weak | Recogniser fine-tuned on real words and blended with the released weights | Test, read once: Bangla CER 39.2 → 28.7 %, Latin 62.5 → 50.6 %, rendered 2.46 → 2.27 % (`real_ft_eval.json`) | Same network: same speed |
| Silence during a 15–35 s capture | Spoken "reading" at once, soft tick every 4 s | `test_tts_queue.py` | No |
| Installer and boot service out of date | Rewritten for the real layout; user service | Fresh clone checked | No |
| Pi timings missing for the latest settings | `scripts/pi_check_all.sh`: one command for all tests and timings | — | Waits for a run |

## Tried and not adopted

- **Smaller picture before reading (880 px):** chosen by the validation rule, contradicted by the single test read (rendered test 3.15 % against 2.46 %). Kept at 1,200 (`OCR_REAL_IMPROVED.md`).
- **Rotation retry:** fewer unread words, same error. Off.
- **Fully fine-tuned recognisers:** better on street words, worse on printed English, over the rule.

## Still open, and what each needs

1. Hearing the new voice and timing it on the Pi: `sh scripts/pi_check_all.sh`.
2. Voice commands with people and the glass's microphone; Bangla command words need an offline Bangla recogniser.
3. Text reading on the Pi: one detector pass is 11–18 s on a hot board; the 8-bit detector and the 880-px option are timed by the Pi script.
4. Heat: the board throttled in the 13 October run; needs a cooler run and a hood or fan check.
5. Counterfeit notes (UV check, more prints), the online mode (API credit), user study: need material or people from outside.
