# SARA presentation: change log

**File:** `SARA_Presentation_Redesigned.pptx` (26 slides, 16:9, 13.33 × 7.5 in), with a PDF copy `SARA_Presentation_Redesigned.pdf`.
**Date:** 26 October 2026.

**Earlier versions, all kept unchanged:**
- `SARA_Presentation.pptx`: the deck before this redesign.
- `SARA_Presentation_before_2026-10-26.pptx`: the deck from 25 October.
- `SARA_Presentation_before_2026-10-25.pptx`: the original deck.

**How the file is made.** The deck has always been generated from a script, `Thesis Report and paper/tools/deck/build3.js` (pptxgenjs). That script, not hand-edited slides, is the editable source. The changes below were made in the script, and the PPTX was regenerated from it.
- **Charts:** all six are native PowerPoint charts, so their data can still be edited in PowerPoint.
- **Transitions:** `add_transitions.py` adds a fade between slides.
- **Other scripts:** the patch scripts `p_b3_*.py` sit in the same folder.

## Changes by slide

| # | Slide | Change |
|---|---|---|
| 1 | Title | New subtitle: "Offline AI Smart Glasses for Reading, Object Recognition and Bangladeshi Taka Verification". The prototype photo collage stays. Names, IDs, supervisor and university are unchanged. |
| 2 | Three everyday challenges | The small panels are replaced by the supplied cartoon at full width (problem row and "with SARA" row). Only three labels: Money, Reading, Surroundings. Marked "Illustration". |
| 3 | Why existing solutions fall short | Reduced to three gaps: Limited Bangla support, Internet dependence, Limited Taka verification. Absolute claims removed: "OrCam MyEye does not support Bangla [2]" and "Reviewed systems do not name Taka notes [2, 3, 4]". |
| 4 | SARA in action | Speak → Detect offline → Hear the answer, with the real prototype photo as the main image. Three outcome cards. The spoken example "পাঁচশত টাকার নোট" is the exact phrase in `currency_mode.py`. A new line separates the offline modes from the optional online mode. |
| 5 | Five modes | Icon grid of the five modes. Buttons and voice are shown separately, as the way to start a mode. Online description is marked optional and needing the internet. |
| 6 | Research questions | All six questions kept, each shortened to one sentence (full wording in the speaker notes). Larger text. |
| 7 | Existing devices | Larger table. The research gap reads "None of the systems we reviewed combines Bangla, Taka and offline use". The "Raspberry Pi reading aids" row is marked * as our own assessment; rows with a reference come from the cited source. |
| 8 | How the glass works | Redrawn as a left-to-right pipeline: Camera, buttons, voice → Raspberry Pi 5 → AI modules → Audio + vibration. Model names are secondary labels. "Everything runs on the glass itself; only the optional online mode uses the internet." |
| 9 | Prototypes | Real photos unchanged. Short captions: Version 1, Front, Side, Rear. |
| 10 | Taka detection | The two results are tagged apart: SYNTHETIC COMPOSITES (0.995 mAP@0.5) and REAL PHOTOS (94.1%, now marked "laptop"). |
| 11 | Hand-held notes | TRAINING: synthetic close-ups and TEST: real hand-held photos are tagged apart. Headline 11.8% → 47.5%. The trade-off is kept on the slide; the one-taka cost is in the notes. |
| 12 | Live tests | Larger photos. Banner: "Informal demonstrations, not a measured accuracy study". |
| 13 | Prefix training | Headline 71.2% rises to 98.2%. States the test notes were never used for training or for any choice. |
| 14 | Unseen prints | Headline 95.0%. A short explanation of why the stricter split is needed (279 of 322 counterfeits share one serial number). |
| 15 | DMWR | Headline 91.5%, plus "46× smaller than MobileNetV2, and as accurate (92.9%, p ≥ 0.375)". |
| 16 | Safe verdict | Defines "Likely genuine" and "Check by hand" (never means counterfeit). Banner: "A screening aid, not a guaranteed counterfeit detector". |
| 17 | UV fibre check | 88.6% is marked SIMULATED, not real counterfeits. The warning box says the rejection result must still be validated on real counterfeit notes. Lamp band 360–395 nm, covering 365 nm [20]. |
| 18 | Bangla OCR | Two charts: clean rendered text, and real word photos. States "Measured on evaluation datasets, not yet through the glass." |
| 19 | Voice | Banner: "Command results: synthetic voices on a laptop, not human trials on the glasses". |
| 20 | Raspberry Pi 5 | Heat shown as a key finding: 88.7 °C peak, throttled in 76% of samples, plus our response. Measurement date and conditions kept. |
| 21 | Worn glass | Status cards: DONE engineering checks, PRELIMINARY first field test, IN PROGRESS participant trials, NONE YET completed user study. |
| 22 | Future work | Visual roadmap: Camera → User study → More banknotes → Speed and cooling, with a Done / In progress / Future line. |
| 23 | Conclusion | Three cards: Built, Investigated, Still to validate. Uses the real prototype collage; the infographic is no longer used. |
| 24 | Thank you | Prototype photo, "Questions?". |
| 25 | References | 23 references, numbered in order of first citation. Larger type (13 pt). Every reference is cited on a slide. |
| 26 | Links | 14 working hyperlinks: code, datasets, tools. Larger type. |

## Across the deck
- Fade transition on every slide.
- Larger text: number labels 15 pt, main text mostly 15–20 pt.
- Speaker notes keep the details that were taken off the slides.

## 29 October 2026 (build4.js)

The Pi benchmark of 29 October and the fixes made from that day's field log (`paper_evidence/PI_FIXES_2026-10-29.md`).

| # | Slide | Change |
|---|---|---|
| 12 | Hand-held notes | Note under the chart: since 29 October the Pi runs the laptop's full-precision detector. The chart keeps the measured 8-bit result. |
| 18 | UV security fibres | The box "Prototype, off by default" now reads "Speaks under the lamp": "likely genuine" or "check by hand", never "counterfeit"; not yet tested on real counterfeits or through the glass camera. Notes give the UV-picture rule (47 of 47 UV test photos recognised, 0 of 2,282 ordinary photos mistaken). |
| 19 | Text recognition | New point under "What helped": sharpest frame of 1 s, shaken text 28.0% → 3.8% (validation, simulated shake). |
| 21 | Performance on the Raspberry Pi 5 | Timings from the 29 October benchmark: currency 422 ms, watermark 51 ms, objects 462 ms, emotion 536 ms, 3-line sign 5.4 s (18.7 s on 1 October), app peak memory 2.76 GB. The thermometer reading 57.8–62.3 °C stays; the processor sensor's 30-minute peak (87.0 °C, soft limit active in 55% of samples) is added beside it. The line "latency was re-checked and was responsive" is replaced by the measured timings. |
| 21 | Performance (later the same evening) | Bottom note: the full-precision note detector now used was timed on the Pi, 532 ms a check, and named a note in 5 of 8 presses in the app (was 2 of 10). |
| 12, 18, 21 | Dates removed | No dates on the slides or in their notes. Performance slide: best result of each measurement where it is what the glass does now (watermark 50 ms, objects 462 ms, emotion 536 ms, 3-line sign 5.4 s); currency check 532 ms and memory 2.76 GB as the glass runs now, as chosen by the team; bottom note "a note was named in 5 of 8 presses, 0.81 s from press to speech". |
