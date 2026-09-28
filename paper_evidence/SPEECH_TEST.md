# Speech output test

Run 2026-09-28 on Windows with the glass's own routing and the SAPI settings of the Windows
harness. Script: `savior_glass/scripts/test_speech.py`; samples in
`savior_glass/results/speech_samples/`; raw: `savior_glass/results/speech_test.json`.
Voices on this machine: Microsoft David and Zira (English only). The Raspberry Pi speaks
Bangla offline through espeak-ng or Piper, which are not installed here.

## Bugs found and fixed

1. **A stressed user did not hear the verdict.** For sad, fearful, angry or disgusted faces,
   `adapt_feedback` kept only the first clause, so "100 taka note. Genuine. …" became "It's
   okay. Take your time. 100 taka note", and the counterfeit warning was dropped the same way.
   Sentences containing a verdict word (আসল, জাল, genuine, counterfeit, jaal, fake) are now
   always kept.
2. **Bangla results collapsed to a generic sentence on the English fallback.** "একশত টাকার নোট।
   আসল" and object announcements became "Detection complete. Bangla is on the screen."
   because the fallback table lacked the verdict words and object names. The verdict,
   the emotion prefixes and all 80 object names (from `assets/labels_bn.json`) are now mapped.

## Results after the fixes

| Check | Result |
|---|---|
| Currency message on the English fallback | "একশত টাকার নোট। আসল" → "100 taka. Genuine" |
| Object message on the English fallback | "সামনে আছে: চেয়ার, মানুষ" → "In front: chair, person" |
| Free Bangla OCR text, offline on Windows | still "Detection complete. Bangla is on the screen." (cannot be glossed; the Pi uses a Bangla voice) |
| Emotion-adaptive rate | neutral rate 1 (6.30 s), happy rate 2 (6.61 s incl. "Good."), sad/fear rate −2 (8.36 s) |
| Verdict kept for stressed users | "It's okay. Take your time. 100 taka note. Genuine." |
| Volume setting | volume 40 has 0.27 × the RMS amplitude of volume 85 |
| Bangla through gTTS (online, used by the Windows harness) | synthesised (24 KB mp3) |
| English intelligibility, speech-recognition round trip (Google, online) | 58 % of reference words recognised; "100 taka note" was missed |

**NOT_MEASURED:**
- Subjective speech quality, which needs human listeners.
- The offline Bangla voice (espeak-ng/Piper) on the Raspberry Pi.
