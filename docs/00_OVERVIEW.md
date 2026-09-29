# Publication Documentation Overview

This folder documents the project for **two planned papers**. All code lives in one repository:
[aflam330/Senior_project_wearable_glass_for_blind_people_using_ocr_and_ylo](https://github.com/aflam330/Senior_project_wearable_glass_for_blind_people_using_ocr_and_ylo).

| Folder in the repository | Role in publications |
|-------------|----------------------|
| `realtime_bangla_taka_detection/` | **Paper 1** — currency detection research |
| `savior_glass/` | **Paper 2** — full smart-glass system (canonical copy) |
| `Unused/smart-glass/` | Outdated copy, moved to `Unused/smart-glass/` on 2026-09-28: HSV-only currency mode (7% on raw note photos), no YOLO, no 2/5 Taka |

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
- The glass project’s currency mode now uses the YOLOv8s detector from Paper 1 (`realtime_bangla_taka_detection/models/best.pt`); HSV + MobileNetV3 is only a fallback when those weights are missing. For 500 / 1,000 Taka the glass says "likely genuine" or "check by hand" and never "counterfeit". Other denominations say the check was not done (`paper_evidence/JAAL_VERDICT_FIXED.md`).
- Paper 1 test metrics are on **held-out synthetic composites**. State this honestly; strengthen with in-the-wild evaluation if possible before submission.

---

## Document index

| File | Contents |
|------|----------|
| [PAPER1_Currency_Detection.md](PAPER1_Currency_Detection.md) | Full technical documentation for the currency paper |
| [PAPER2_Smart_Glass_System.md](PAPER2_Smart_Glass_System.md) | Full technical documentation for the system paper |
| [PROJECT_INVENTORY.md](PROJECT_INVENTORY.md) | Folders, datasets, numbers to cite, and what is still open |
| [PAPER_OUTLINES.md](PAPER_OUTLINES.md) | Suggested IMRAD / conference outlines + abstract drafts |
| [RELATED_WORK_BIBLIOGRAPHY.md](RELATED_WORK_BIBLIOGRAPHY.md) | All papers from `literature/Review and paper links.xlsx` (links, abstracts, tags) |
| Source spreadsheet | `literature/Review and paper links.xlsx` (sheets: Review, Downloaded Paper) |
| Existing PDF | `realtime_bangla_taka_detection/Bangla_Currency_Detection_Report.pdf` (16-page technical report) |

---

## What is still needed from you

The organized list, with the files behind each item, is `docs/PROJECT_INVENTORY.md`.

1. Grouped whole-note photos of known counterfeit notes, taken with the glass camera. Since 2026-09-29 the glass runs a safe policy for 500 / 1,000 Taka ("likely genuine" or "check by hand", never "counterfeit"; `paper_evidence/JAAL_VERDICT_FIXED.md`). It is safe on all measured data but rarely confirms a whole note. Those photos are what would let it confirm more notes, and give its whole-note safety more than about four notes of evidence.
2. One run of `savior_glass/scripts/benchmark_pi5.py` on the Raspberry Pi 5.
3. The user study, when the participants are available. The protocol is already written.
4. Author names, affiliation, and supervisor.
5. An API key only if the Claude mode will be demonstrated.
