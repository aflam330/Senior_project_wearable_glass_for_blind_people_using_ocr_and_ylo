"""Write a sendable status PDF + Markdown for Savior Glass / RoboEye."""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs"
OUT_PDF = OUT_DIR / "Savior_Glass_Progress_Report_2026-09-21.pdf"
OUT_MD = OUT_DIR / "Savior_Glass_Progress_Report_2026-09-21.md"

MD = """# Savior Glass / RoboEye — Progress and Publication Status

**Date:** 21 September 2026  
**Project:** Wearable assistive smart glass (OCR, objects, Bangladeshi Taka, emotion-adaptive speech)  
**Repos:** `savior_glass`, `realtime_bangla_taka_detection`

This note is for sharing with a supervisor or collaborator. Every accuracy figure below was measured in this workspace. Do not replace them with draft paper numbers.

---

## Executive summary

The system is a **working senior-project prototype**, not an A* conference paper.

- Emotion recognition on the **official RAF-DB test set** (3,068 images) is **86.5%** accuracy (macro-F1 0.80). It was **81.0%** before this week’s GPU training. The requested **93–96%** was **not** reached.
- Published 7-class RAF-DB SOTA is about **92.2%** (POSTER V2). 93–96% is above typical published numbers on this split.
- Taka detection on 450 wild notes: **98.4%** detection, **96.9%** top-1 denomination.
- **Not eligible for A* venues** (CVPR, ICCV, NeurIPS, ICLR, CHI) as the work stands: existing models were fine-tuned; there is no new algorithm and no controlled user study.

**Eligible today:** thesis, demo, IEEE application / regional conference.  
**Not eligible today:** CVPR / NeurIPS / CHI.

---

## What is implemented

| Area | Implementation | Status |
|------|----------------|--------|
| Emotion | EfficientNet-V2-S on RAF-DB; live path in `roboeye/fer_emotion.py` (Haar face, FER+ ONNX fallback, emotion-adaptive TTS) | Working, **86.5%** RAF-DB |
| Currency | YOLOv8s, 9 Bangladeshi Taka classes; ONNX/INT8 export | Working |
| Authenticity | JaalTaka CNN+ViT, dual-head, FedAvg, EWC scripts | Code works; **do not publish split acc 1.00** |
| OCR | EasyOCR Bangla + English | Working |
| Objects | YOLOv8 COCO | Working |
| Assistive I/O | Offline TTS, ASR intents, haptics, solvePnP straighten, Grad-CAM overlay | PC/Windows working; not bone-conduction or VR headset |
| Glass app | `savior_glass` Pi + `test_windows.py` | Working |

Live weights for emotion: `savior_glass/models/emotion_faces.pt` (arch `efficientnet_v2_s`, 224 px).

---

## Emotion experiments (this week)

GPU: NVIDIA GeForce RTX 3050 Laptop (CUDA PyTorch installed; it had been CPU-only). Dataset already on disk: official RAF-DB train 12,271 / test 3,068.

| Run | Test set | Accuracy | Kept as live model? |
|-----|----------|----------|---------------------|
| FER+ ONNX (older eval) | FER-2013 holdout, n=5,741 | 57.9% | Fallback only |
| HGB + FER+ blend | FER-2013 holdout, n=5,741 | 67.6% | No |
| MobileNetV3-Large (previous) | RAF-DB official test, n=3,068 | 81.0% | Replaced |
| **EfficientNet-V2-S + flip TTA** | **RAF-DB official test, n=3,068** | **86.5%** | **Yes** |
| ViT (`trpakov/vit-face-expression`) fine-tune | RAF-DB official test | 84.3% peak | Stopped |
| AffectNet EfficientNet-B2 fine-tune | RAF-DB official test | 85.2% then overfit | Stopped |

### Best model — per class (RAF-DB official test)

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| Happy | 1185 | 0.946 | 0.925 | 0.936 |
| Sad | 478 | 0.839 | 0.839 | 0.839 |
| Angry | 162 | 0.789 | 0.809 | 0.799 |
| Surprise | 329 | 0.867 | 0.872 | 0.870 |
| Fear | 74 | 0.774 | 0.554 | 0.646 |
| Disgust | 160 | 0.671 | 0.588 | 0.627 |
| Neutral | 680 | 0.815 | 0.890 | 0.851 |
| **Overall** | **3068** | macro-P **0.815** | macro-R **0.782** | **acc 86.5% / macro-F1 0.795** |

Bottleneck: Fear (74 test images) and Disgust (160). Happy is already strong (F1 0.936).

---

## Other measured results (already in the repo)

| Task | Protocol | Result |
|------|----------|--------|
| Taka wild notes | 450 images, existing YOLOv8s `best.pt` | Detection 98.4%, top-1 96.9% |
| Taka YOLO val | Saved val metrics | mAP@0.5 0.994, mAP@0.5:0.95 0.844 (synthetic composites) |
| OCR | EasyOCR, n=80 (40 BN + 40 EN) | CER 15.5% overall (BN 27.8%, EN 3.2%) |
| Objects | YOLOv8s, 250-image COCO val subset | mAP@0.5 0.60 |
| Authenticity | Local JaalTaka split | Acc 1.00 — **same-capture leak; not a paper number** |

---

## A* conference eligibility

A* here means CORE A* venues such as **CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR**, and CHI-style HCI A* tracks.

| Expectation | This project |
|-------------|--------------|
| New method others will cite | Fine-tunes YOLO, EfficientNet, FER+, EasyOCR |
| SOTA or a large clean gain on a standard benchmark | Emotion 86.5% vs ~92% SOTA; Taka numbers are strong but application-specific |
| Ablations, seeds, fair baselines | Mostly single-run figures |
| Or a CHI/ASSETS user study | No controlled study with visually impaired users |

**Verdict**

- **CVPR / ICCV / ECCV — No.**
- **NeurIPS / ICML / ICLR — No.**
- **CHI / ASSETS — Not yet** (need a user study; the prototype is only a prerequisite).

**What you can submit now:** undergraduate/graduate thesis; system demo; IEEE application or regional venues (e.g. TENCON, ICCIT, ICAEE). A later ASSETS-style paper still needs a user study and honest metrics. It would still not be CVPR unless the *method* is new.

---

## Do not claim in any draft or email

- 93–96% emotion accuracy (not measured).
- 96.7% / 98.1% authenticity (never measured by this stack).
- Raspberry Pi 15 FPS (not timed on a Pi).
- Qwen-VL / LLaVA / bone-conduction hardware (not what runs today).
- Full 22,315-image composite set on this machine (pipeline exists; full set not generated here).

---

## Recommended next step

Freeze **86.5%** emotion and present this as a **thesis + live demo**. Record a short video: note detection, speech, emotion-adaptive TTS.

Only restart emotion training if the goal is a higher *honest* RAF-DB number (e.g. POSTER V2), and report that number even if it is 90–92%.

---

## Key files

- `savior_glass/models/emotion_faces.pt` — live EfficientNet-V2-S weights  
- `savior_glass/results/emotion_rafdb.json` — official RAF-DB metrics  
- `savior_glass/scripts/train_emotion_rafdb.py` — training entrypoint  
- `realtime_bangla_taka_detection/roboeye/fer_emotion.py` — runtime detector  
- `realtime_bangla_taka_detection/models/best.pt` — Taka YOLO  

*Prepared from local evaluation JSON in this workspace, 21 September 2026.*
"""


class Report(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, "Savior Glass / RoboEye  |  Progress report  |  21 September 2026", align="L")
        self.ln(12)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}  |  Measured numbers only  |  Not for A* submission", align="C")

    def heading(self, text: str, size: int = 14) -> None:
        self.ln(3)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(20, 20, 20)
        self.multi_cell(self.epw, 8, text)
        self.ln(1)

    def body(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(30, 30, 30)
        self.multi_cell(self.epw, 6, text)
        self.ln(1)

    def bullet(self, text: str) -> None:
        self.set_font("Helvetica", "", 11)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin + 4)
        self.multi_cell(self.epw - 4, 6, "- " + text)
        self.ln(0.4)

    def table(self, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 8.5)
        self.set_fill_color(235, 235, 235)
        self.set_text_color(20, 20, 20)
        for h, w in zip(headers, widths):
            self.cell(w, 7, h, border=1, fill=True)
        self.ln()
        self.set_font("Helvetica", "", 8)
        for row in rows:
            if self.get_y() > 270:
                self.add_page()
            self.set_x(self.l_margin)
            for txt, w in zip(row, widths):
                if self.get_string_width(txt) > w - 2:
                    txt = txt[: max(8, int(len(txt) * (w - 2) / self.get_string_width(txt) - 1))] + ".."
                self.cell(w, 6.5, txt, border=1)
            self.ln()
        self.ln(3)


def build_pdf() -> None:
    pdf = Report(format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(pdf.epw, 9, "Savior Glass / RoboEye")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(pdf.epw, 7, "Progress and publication-status report")
    pdf.ln(2)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(70, 70, 70)
    pdf.multi_cell(
        pdf.epw,
        6,
        "Date: 21 September 2026\n"
        "Project: Wearable assistive smart glass (OCR, objects, Bangladeshi Taka, emotion-adaptive speech)\n"
        "Audience: supervisor / collaborator\n"
        "Rule: every accuracy figure below was measured in this workspace.",
    )
    pdf.ln(3)
    box_h = 32
    y = pdf.get_y()
    pdf.set_fill_color(255, 236, 230)
    pdf.set_draw_color(180, 70, 50)
    pdf.rect(pdf.l_margin, y, pdf.epw, box_h, style="DF")
    pdf.set_xy(pdf.l_margin + 4, y + 3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(120, 30, 20)
    pdf.multi_cell(
        pdf.epw - 8,
        6,
        "VERDICT: Not eligible for an A* conference (CVPR, ICCV, NeurIPS, ICLR, CHI) as the work stands.\n"
        "Eligible: thesis, live demo, IEEE application / regional conference.\n"
        "Emotion (official RAF-DB test): 86.5%  |  requested 93-96%: not reached.",
    )
    pdf.set_xy(pdf.l_margin, y + box_h + 4)

    pdf.heading("1. Executive summary")
    pdf.body(
        "This is a working senior-project prototype that fine-tunes existing models. "
        "It is not a new computer-vision method and it has no controlled user study with visually impaired participants."
    )
    pdf.bullet("Emotion on official RAF-DB test (3,068 images): 86.5% accuracy, macro-F1 0.80. Was 81.0% before this week's GPU training.")
    pdf.bullet("Published 7-class RAF-DB SOTA is about 92.2% (POSTER V2). 93-96% sits above typical published numbers on this split.")
    pdf.bullet("Taka on 450 wild notes: 98.4% detection, 96.9% top-1 denomination (YOLOv8s fine-tune).")
    pdf.bullet("Do not put 93-96% emotion or 96.7% authenticity into any slide, email, or draft.")

    pdf.heading("2. What was implemented")
    pdf.table(
        ["Area", "What landed", "Status"],
        [
            ["Emotion", "EfficientNet-V2-S + FER+ fallback + adaptive TTS", "Working, 86.5% RAF-DB"],
            ["Currency", "YOLOv8s 9-class Taka; ONNX/INT8", "Working"],
            ["Authenticity", "CNN+ViT, dual-head, FedAvg, EWC", "Code yes; acc 1.00 not publishable"],
            ["OCR / objects", "EasyOCR BN+EN; YOLOv8 COCO", "Working"],
            ["Assistive I/O", "TTS, ASR, haptics, pose, Grad-CAM", "PC working; not bone-conduction/VR"],
        ],
        [32, 95, 63],
    )

    pdf.heading("3. Emotion experiments this week")
    pdf.body(
        "GPU: RTX 3050 Laptop after installing CUDA PyTorch (it had been CPU-only). "
        "Dataset already on disk: RAF-DB train 12,271 / official test 3,068. Live weights: savior_glass/models/emotion_faces.pt"
    )
    pdf.table(
        ["Run", "Test", "Acc", "Kept?"],
        [
            ["FER+ ONNX (older)", "FER-2013 n=5741", "57.9%", "Fallback"],
            ["HGB + FER+ blend", "FER-2013 n=5741", "67.6%", "No"],
            ["MobileNetV3-Large (before)", "RAF-DB official", "81.0%", "Replaced"],
            ["EfficientNet-V2-S + TTA", "RAF-DB official", "86.5%", "YES - live"],
            ["ViT fine-tune", "RAF-DB official", "84.3% peak", "Stopped"],
            ["AffectNet ENET-B2 FT", "RAF-DB official", "85.2% then overfit", "Stopped"],
        ],
        [52, 48, 42, 48],
    )

    pdf.heading("Best model per class (RAF-DB official test)")
    pdf.table(
        ["Class", "n", "Prec", "Rec", "F1"],
        [
            ["Happy", "1185", "0.946", "0.925", "0.936"],
            ["Sad", "478", "0.839", "0.839", "0.839"],
            ["Angry", "162", "0.789", "0.809", "0.799"],
            ["Surprise", "329", "0.867", "0.872", "0.870"],
            ["Fear", "74", "0.774", "0.554", "0.646"],
            ["Disgust", "160", "0.671", "0.588", "0.627"],
            ["Neutral", "680", "0.815", "0.890", "0.851"],
            ["OVERALL", "3068", "macro-P 0.815", "macro-R 0.782", "86.5% / F1 0.795"],
        ],
        [38, 28, 42, 42, 40],
    )
    pdf.body("Bottleneck: Fear (74 images) and Disgust (160). Happy is already strong (F1 0.936).")

    pdf.heading("4. Other measured results")
    pdf.table(
        ["Task", "Protocol", "Result"],
        [
            ["Taka wild", "450 images, YOLOv8s best.pt", "Detect 98.4%, top-1 96.9%"],
            ["Taka YOLO val", "Saved val (synthetic composites)", "mAP50 0.994, mAP50-95 0.844"],
            ["OCR", "EasyOCR n=80 (40 BN + 40 EN)", "CER 15.5% (BN 27.8%, EN 3.2%)"],
            ["Objects", "YOLOv8s, 250 COCO val subset", "mAP50 0.60"],
            ["Authenticity", "Local JaalTaka split", "Acc 1.00 - LEAKY, do not publish"],
        ],
        [38, 72, 80],
    )

    pdf.heading("5. Why this is not A* material")
    pdf.body(
        "A* reviewers expect a new method, a SOTA (or large clean) gain on a standard benchmark, "
        "or a CHI/ASSETS-grade user study. This repo fine-tunes public models for a wearable pipeline."
    )
    pdf.bullet("CVPR / ICCV / ECCV: No. Emotion 86.5% is below ~92% SOTA; Taka is a YOLOv8s application.")
    pdf.bullet("NeurIPS / ICML / ICLR: No. There is no new learning algorithm.")
    pdf.bullet("CHI / ASSETS: Not yet. Need a study with blind/low-vision users, not only a PC demo.")
    pdf.body(
        "Reasonable venues now: thesis examination, system demo, IEEE application or regional conferences "
        "(TENCON, ICCIT, ICAEE, similar). A later accessibility paper still needs a user study."
    )

    pdf.heading("6. Do not claim")
    pdf.bullet("93-96% emotion (not measured).")
    pdf.bullet("96.7% / 98.1% authenticity (never measured by this stack).")
    pdf.bullet("Raspberry Pi 15 FPS (not timed on a Pi).")
    pdf.bullet("Qwen-VL / LLaVA / bone-conduction hardware (not what runs today).")
    pdf.bullet("Full 22,315-image composite set on this machine (script exists; set not generated here).")

    pdf.heading("7. What to do now")
    pdf.body(
        "Freeze 86.5% emotion and present a thesis + live demo. Record 1-2 minutes: note detection, speech, "
        "emotion-adaptive TTS. Only restart emotion training if the goal is a higher honest RAF-DB number "
        "(for example POSTER V2), and report that number even if it is 90-92%."
    )

    pdf.heading("8. Key files")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Courier", "", 9)
    pdf.multi_cell(
        pdf.epw,
        5,
        "savior_glass/models/emotion_faces.pt\n"
        "savior_glass/results/emotion_rafdb.json\n"
        "savior_glass/scripts/train_emotion_rafdb.py\n"
        "realtime_bangla_taka_detection/roboeye/fer_emotion.py\n"
        "realtime_bangla_taka_detection/models/best.pt",
    )
    pdf.ln(2)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(pdf.epw, 5, "Prepared from local evaluation JSON in this workspace, 21 September 2026.")

    OUT_DIR.mkdir(exist_ok=True)
    pdf.output(str(OUT_PDF))


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    OUT_MD.write_text(MD, encoding="utf-8")
    build_pdf()
    print("WROTE", OUT_PDF)
    print("WROTE", OUT_MD)


if __name__ == "__main__":
    main()
