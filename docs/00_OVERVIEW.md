# Publication Documentation Overview

This folder documents the three project repositories for **two planned papers**.

| Repo folder | GitHub | Role in publications |
|-------------|--------|----------------------|
| `realtime_bangla_taka_detection/` | [Mukut313/realtime_bangla_taka_detection](https://github.com/Mukut313/realtime_bangla_taka_detection) | **Paper 1** — currency detection research |
| `savior_glass/` | [Mukut313/savior_glass](https://github.com/Mukut313/savior_glass) | **Paper 2** — full smart-glass system (canonical copy) |
| `Unused/smart-glass/` | [Mukut313/smart-glass](https://github.com/Mukut313/smart-glass) | Outdated copy, moved to `Unused/smart-glass/` on 2026-09-28: HSV-only currency mode (7% on raw note photos), no YOLO, no 2/5 Taka |

---

## Two-paper plan

### Paper 1 — Currency (computer vision focus)
**Use:** `docs/PAPER1_Currency_Detection.md`  
**Source of truth:** `realtime_bangla_taka_detection/` (+ existing `Bangla_Currency_Detection_Report.pdf`)

Focus: synthetic compositing, YOLOv8s training, metrics, real-time webcam + TTS demo.

### Paper 2 — Smart Glass (assistive systems / edge AI focus)
**Use:** `docs/PAPER2_Smart_Glass_System.md`  
**Source of truth:** `savior_glass/`

Focus: offline Bangla assistive wearable on Raspberry Pi 5 — OCR, object detection, currency mode, GPIO UX, Piper/espeak TTS.

---

## Important relationship note

- `smart-glass` was an early copy of `savior_glass` and fell behind it (no YOLO currency detector, no jaal check, no emotion/assistive modules). It was moved to `Unused/smart-glass/` on 2026-09-28. Cite only `savior_glass` in the paper.
- The glass project’s currency mode currently uses **HSV + optional MobileNetV3**, not the YOLOv8s detector from Paper 1. For a stronger Paper 2, plan to **integrate** `models/best.pt` from the currency repo into `CurrencyMode`.
- Paper 1 test metrics are on **held-out synthetic composites**. State this honestly; strengthen with in-the-wild evaluation if possible before submission.

---

## Document index

| File | Contents |
|------|----------|
| [PAPER1_Currency_Detection.md](PAPER1_Currency_Detection.md) | Full technical documentation for the currency paper |
| [PAPER2_Smart_Glass_System.md](PAPER2_Smart_Glass_System.md) | Full technical documentation for the system paper |
| [PAPER_OUTLINES.md](PAPER_OUTLINES.md) | Suggested IMRAD / conference outlines + abstract drafts |
| [RELATED_WORK_BIBLIOGRAPHY.md](RELATED_WORK_BIBLIOGRAPHY.md) | All papers from `Review and paper links.xlsx` (links, abstracts, tags) |
| Source spreadsheet | `Review and paper links.xlsx` (sheets: Review, Downloaded Paper) |
| Existing PDF | `realtime_bangla_taka_detection/Bangla_Currency_Detection_Report.pdf` (16-page technical report) |

---

## What you may still need from you (for stronger papers)

1. Author names, affiliations, acknowledgements.
2. Hardware BOM photos / wiring diagram for the glass.
3. User-study protocol (if any) or planned evaluation on RPi latency.
4. Whether Paper 2 will claim the YOLOv8 currency model after integration.
5. Target venues (IEEE Access, Sensors, ICIP workshop, local journal, etc.) so tone/length can be tuned.
