# SARA presentation: quality-assurance report

**File checked:** `SARA_Presentation_Redesigned.pptx`. **Date:** 26 October 2026.

## Checks that were run

| Check | How | Result |
|---|---|---|
| The file opens | Opened in Microsoft PowerPoint (Windows) | Opens; 26 slides, 960 × 540 pt (16:9) |
| Every slide renders | PowerPoint exported all 26 slides to PNG, plus a PDF | All rendered |
| Visual inspection | All 26 images looked at for clipping, overlap, missing pictures, tiny text and unreadable charts | Issues found and fixed (see below); final render has no clipping, overlap or missing images |
| Charts | Counted in the file | 6 native, editable charts |
| Hyperlinks | Counted in the file | 14 external links on the Links slide |
| Citations | Every reference key searched in the slide source | All 23 are cited on at least one slide, numbered by first use |
| Slide numbers | Read from the render | 1–26, in order |
| Numbers on slides | Compared with the files in `paper_evidence/` | Agree, except the points flagged below |
| Risky wording | Searched the slide text | "guarantee" appears only in "not a guaranteed counterfeit detector"; no "No wearable…" claim remains |

**Issues found during rendering and fixed:**
- On slide 13, "71.2% ↓ 98.2%" read like a fall; it now says "rises to".
- A heading wrapped on slide 4.
- Unused space on slides 8 and 22.
- Small text on slides 5, 8, 25 and 26.
- On slide 10, the 94.1% did not say it was measured on the laptop.

## Checks that were not done
- **Projector:** not tested on a projector or the room's display.
- **Other software:** not opened in Google Slides, Keynote or LibreOffice.
- **Fonts:** not embedded. The deck uses Cambria, Calibri and Nirmala UI (Bangla). Check that the defence computer has them, especially Nirmala UI.
- **Animations:** only slide transitions (fade). Shape-by-shape animations, such as revealing the pipeline stage by stage, are not supported by the generator. Add them in PowerPoint if wanted.
- **Live links:** the links were not opened to confirm each page still loads.

## Values that disagree between sources (flagged, not silently reconciled)

| Item | Deck | Other source | Note |
|---|---|---|---|
| UV check, genuine notes accepted | 85.6% (95 / 111) | Public SARA README: 92.8% (103 / 111) | `UV_CHECK_IMPROVED.md` says 92.8% was replaced by 85.6% once the colour condition was added. **The README is out of date**; the 92.8% line was added on 26 October. Fix the README. |
| Taka, other people's photos | 94.1% (hand-held detector, laptop) | README: 91.5% | `HANDHELD_FINETUNE.md` gives 89.6% → 94.1% on the laptop, and "level" on the Pi. The README's 91.5% comes from an earlier stage. |
| OCR, rendered text | 9.3% → 2.3% | README: 9.3% → 2.4% | 2.3% matches `OCR_FINETUNE.md` / `OCR_TEXT_REGION.md`; 2.4% is an earlier figure. |
| Pi timings | 399 ms, 64 ms, 562 ms, 18.3 s (13 October run, hot board) | README: 327 ms, 50 ms, 207 ms (YOLOv8n), 18.7 s (1 October run) | Two different runs. The deck uses the later one and says the board was throttled. |
| Pi temperature | "idle 71 °C" | `PI5_SECOND_RUN_2026-10-13.md`: 73 °C before the first call | 71 °C comes from `THERMAL.md` (idle, fan at 7,800 rpm). Both may be right for different moments. Confirm which to quote. |

## Claims that need your confirmation
1. **"OrCam MyEye does not support Bangla"** (slides 3, 7): taken from the thesis literature review. Confirm against OrCam's current language list before the defence.
2. **"Research glasses send pictures to a server [3, 4]"** and the table rows for Google Glass and AIris: taken from the thesis literature review. Worth a final check against the two papers.
3. **"Raspberry Pi reading aids" row** (slide 7): our own assessment, marked *. It has no citation.
4. **Cartoon on slide 2:** a supplied illustration, labelled as such. Its speech bubble says "500 Taka", but the note drawn in it shows 1000/5000-style numbers and is not a real Taka design. Some viewers may notice.
5. **Bibliographic details missing in the source** (not invented):
   - [2] OrCam, [6] EasyOCR, [7] YOLOv8, [10] Piper and [11] eSpeak NG have no year (software and product pages).
   - [12] BanglaTaka has no authors or year; complete it from its Mendeley Data page.
6. **[14] and [15]** have the same title on purpose: one is the JaalTaka dataset (Mendeley, 2025), the other its Data in Brief article (2026).
7. **Text size:** titles are 30 pt. Main text is mostly 15–20 pt; labels and source lines are 10–14 pt. That is below the 20–24 pt target in places, because the evidence lines and citations were kept rather than cut.

## Content kept, as required
- **Measured results:** sample counts, metric names and chart axes are unchanged.
- **Negative results and trade-offs:**
  - one-taka notes named more often (in the notes)
  - several-note total accuracy 97.0% (was 99.2%)
  - weak real-photo OCR
  - noisy-voice results
  - heat and throttling
  - first field test with the glass camera
- **Safety policy:** the glass never says "counterfeit".
- **Clear status labels:**
  - UV check: off by default and simulated
  - voice: synthetic voices on a laptop
  - live tests: informal
  - participant trials: in progress, with no completed study claimed
