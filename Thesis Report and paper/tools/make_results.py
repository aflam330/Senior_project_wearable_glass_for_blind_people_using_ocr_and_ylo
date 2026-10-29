"""Build 'SARA: results, benchmarks and latency' as a PDF in Times New Roman.
Every number is copied from a result file of the project or from the Pi's benchmark report (benchmark_report/2026-10-29/)."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(r"E:\Final SP")
REPORT = ROOT / "Thesis Report and paper" / "thesis report" / "My report"
PI = ROOT / "benchmark_report" / "graphs"
FIGS = ROOT / "paper_evidence" / "figures"
OUT = REPORT / "SARA_Results_Benchmarks.pdf"
F = r"C:\Windows\Fonts"
for n, f in (("TNR", "times.ttf"), ("TNR-B", "timesbd.ttf"), ("TNR-I", "timesi.ttf"), ("TNR-BI", "timesbi.ttf")):
    pdfmetrics.registerFont(TTFont(n, F + "\\" + f))
pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")

body = ParagraphStyle("body", fontName="TNR", fontSize=11.5, leading=16, alignment=TA_JUSTIFY, spaceAfter=6)
h1 = ParagraphStyle("h1", fontName="TNR-B", fontSize=14, leading=18, spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#1F4E79"))
h2 = ParagraphStyle("h2", fontName="TNR-B", fontSize=12, leading=16, spaceBefore=8, spaceAfter=3)
cell = ParagraphStyle("cell", fontName="TNR", fontSize=10, leading=12.5)
cellb = ParagraphStyle("cellb", parent=cell, fontName="TNR-B")
centre = ParagraphStyle("centre", fontName="TNR", fontSize=11.5, leading=16, alignment=TA_CENTER)
title = ParagraphStyle("title", fontName="TNR-B", fontSize=17, leading=22, alignment=TA_CENTER)
cap = ParagraphStyle("cap", fontName="TNR-B", fontSize=10.5, leading=13, alignment=TA_CENTER, spaceBefore=6, spaceAfter=4)
note = ParagraphStyle("note", fontName="TNR-I", fontSize=10, leading=13, alignment=TA_CENTER, spaceAfter=8)
story = []
P = lambda t: story.append(Paragraph(t, body))
H = lambda t: story.append(Paragraph(t, h1))
H2 = lambda t: story.append(Paragraph(t, h2))


def bullets(items):
    story.append(ListFlowable([ListItem(Paragraph(t, body), leftIndent=16) for t in items], bulletType="bullet",
                              bulletFontName="TNR", bulletFontSize=11, leftIndent=16))


def table(caption, rows, widths):
    story.append(Paragraph(caption, cap))
    t = Table([[Paragraph(c, cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)],
              colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.9, 0.93, 0.97)),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story.append(t)
    story.append(Spacer(1, 6))


def figure(path, width_cm, caption):
    if path.exists():
        story.append(Image(str(path), width=width_cm * cm, height=width_cm * cm, kind="proportional"))
        story.append(Paragraph(caption, note))


story.append(Paragraph("SARA (Smart Assistive Reading Aid)", title))
story.append(Paragraph("<b>Results, benchmarks and latency</b> &nbsp;·&nbsp; 14 October 2026", centre))
story.append(Paragraph("MD Abdun Fattah Lam (2220014), Mukut Protim Memo (2020086), Tanjila Hossain Tapti (2130561)<br/>"
                       "Supervisor: Mohammad Shidujaman, PhD &nbsp;·&nbsp; Department of CSE, Independent University, Bangladesh", centre))
story.append(Spacer(1, 6))
P("This document collects every measured result of the project in one place: how accurate each function is, how fast it "
  "runs on the Raspberry Pi 5 and on the laptop, and what is not measured. Each number comes from a saved result file or "
  "from the benchmark report written on the Pi on 13 October 2026. Where a number was measured on a hot, throttled board, "
  "or only on the laptop, it says so.")

# ------------------------------------------------------------------ 1
H("1. Accuracy of each function")
table("Table 1: Accuracy on data the models were not trained on", [
    ["Function", "Test data", "Result"],
    ["Taka recognition", "2,282 held-out pictures made like the training pictures", "Precision 0.995, recall 0.996, mAP@0.5 0.995, mAP@0.5:0.95 0.849"],
    ["Taka recognition", "1,536 photographs by other people (Bangla Money)", "94.1% right, 17 wrong (hand-held detector, 0.75); earlier detector 89.6%, 50 wrong (91.5% at 0.25)"],
    ["Taka recognition, notes held in a hand", "1,127 NSTU-BDTAKA close-ups, one per photo", "47.5% right (earlier detector 11.8%); Pi 8-bit: 33.5% (earlier 8.7%)"],
    ["Unknown notes (one Taka, not a class)", "101 Bangla Money photos", "Announced as a value: 36.6% (earlier 22.8%); Pi 8.9% (earlier 5.0%)"],
    ["Several notes in one picture", "900 pictures of 1, 2 or 3 test-split notes on new backgrounds", "Spoken total right 97.0% (earlier detector 99.2%); never too many counted"],
    ["Counterfeit check, 1 view / 6 views", "208 test notes, usual split, 3 training runs", "98.2 ± 0.7% / 98.7 ± 0.3% (71.2 ± 11.4% at 1 view without prefix training)"],
    ["Counterfeit check on unseen prints", "222 test notes whose counterfeit prints are not in training", "Network alone about 90%; with the watermark check 94.4% (1 view), 95.0% (6 views)"],
    ["Safe spoken verdict", "88 test counterfeits", "0 passed as \"likely genuine\"; the glass never says \"counterfeit\""],
    ["Safe spoken verdict, genuine notes confirmed", "Close-up views / whole-note photos by other people", "57.5% / 0.4–2%"],
    ["Watermark check (MobileNetV2, in the glass)", "197 unseen-print notes", "92.9% (AUC 0.976); 92.6% on all 222 with the learned watermark finder"],
    ["Watermark check, DMWR (47,818 parameters)", "Same 197 notes, 3 training runs", "91.5 ± 0.3%; not significantly different from MobileNetV2"],
    ["UV fibre check, with the colour condition", "111 real UV photos of genuine notes; simulated fibre-free printed sheets", "85.6% of genuine photos accepted; 88.6% of held-out simulated sheets rejected. No real counterfeit tested"],
    ["Text reading", "Rendered benchmark, test set (432 pictures)", "Letter error 2.3% (Bangla 3.0%, English 1.5%) with the new recogniser; 2.5% before it; 9.3% before this work"],
    ["Text reading", "1,000 real word photos (ICDAR 2019 MLT)", "Letter error 28.7% Bangla, 50.6% Latin script with the recogniser trained on real words (39.2% and 62.5% before); nothing read on 12% and 33%"],
    ["Text reading, picture enlarged to 880 instead of 1,200 pixels", "Rendered test set", "3.15% (full-precision detector), 3.12% (8-bit): worse, so not adopted"],
    ["Object detection, YOLOv8s", "All 5,000 COCO val2017 photos", "mAP@0.5 0.551; 0.521 through the glass path (0.493 before a fix)"],
    ["Object detection, YOLOv8n / YOLOv8m", "Same", "0.453 / 0.607"],
    ["Emotion", "RAF-DB official test, 3,068 faces", "86.5% (macro-F1 0.795)"],
    ["Handwritten Bangla letters (fall-back)", "38,608 characters from 305 unseen writers", "90.0%"],
    ["Voice commands (wake word \"glass\")", "12 held-out synthetic voices, 468 commands per condition; not people",
     "Clean 100%; soft 99.8%; whispered 98.9%; street noise 10 / 5 / 0 / -5 / -10 dB: 98.5 / 93.8 / 82.5 / 57.7 / 39.5%; people talking 10 / 5 dB: 54.3 / 26.1%"],
    ["Voice commands, false commands", "360 ordinary sentences per condition, 14 conditions",
     "13 of 5,040 (212 without the wake word); median response 0.86 s after the speech ends"],
    ["Bangla voice commands (Whisper base, offline)", "4 unseen Bangla voices, 15 phrases each", "45% clean, 62% with noise; no wrong command; not deployed"],
    ["Bangla voice (Piper), machine-listener check", "6 sentences given to an online Bangla speech recogniser",
     "2 exact, 2 with small differences, 2 with wrong words"],
], [4.2, 5.4, 6.6])

# ------------------------------------------------------------------ 2
H("2. Latency on the Raspberry Pi 5")
P("Measured on the glass's own Raspberry Pi 5 (8 GB, fan and heat sink, Logitech C270 webcam), 100 calls per row after "
  "3 warm-up calls, from the Pi's benchmark report (benchmark_report/2026-10-29/REPORT.md). The currency check is given "
  "for the full-precision Taka detector the glass uses (benchmark_report/2026-10-29/raw/currency_current_code.json).")
table("Table 2: Latency on the Raspberry Pi 5 (median; 95th percentile in brackets)", [
    ["Part of the glass", "Time"],
    ["Currency check, end to end (full-precision detector, used)", "532 ms (627 ms)"],
    ["Taka detector alone: full precision (used) / 8-bit option", "483 ms (566 ms) / 339 ms (551 ms)"],
    ["Currency answer in use, press to speech (8 presses)", "0.81 s"],
    ["Watermark check, learned finder (used)", "51 ms (63 ms)"],
    ["Watermark check, SIFT alignment", "377 ms (854 ms)"],
    ["Counterfeit safe-check, one crop", "37 ms (51 ms)"],
    ["Counterfeit network, 1 view / 6 views", "35 ms / 83 ms"],
    ["Object scan as the app runs it (YOLOv8s)", "462 ms (635 ms)"],
    ["Object detector YOLOv8n alone", "191 ms (222 ms)"],
    ["Emotion, face in view", "536 ms (844 ms)"],
    ["Emotion, no face in view (face search only)", "47 ms (51 ms)"],
], [10.4, 5.8])
figure(ROOT / "benchmark_report" / "2026-10-29" / "graphs" / "01_latency_all_modes.png", 15.5,
       "Latency of every mode and model on the Pi (from the Pi's benchmark report).")

H2("Text reading on the Pi")
table("Table 3: Text reading on the Raspberry Pi 5 (8-bit text detector, read width 1,200; median of 4 or 5 captures)", [
    ["Capture", "Total", "Finding the text", "Reading the letters", "First line spoken", "Letter errors"],
    ["English sign, 3 lines", "5.43 s", "2.83 s", "2.61 s", "–", "0%"],
    ["Bangla sign, 1 line", "3.61 s", "2.82 s", "0.64 s", "–", "0%"],
    ["Bangla sign, 3 lines", "5.29 s", "2.84 s", "2.44 s", "–", "0%"],
    ["Label, 6 lines, Bangla + English", "9.30 s", "2.87 s", "6.35 s", "4.27 s", "0%"],
    ["Page, 12 lines (640×480)", "15.38 s", "2.96 s", "12.40 s", "4.35 s", "0%"],
    ["Page, 12 lines (1280×720)", "14.79 s", "2.52 s", "12.32 s", "3.87 s", "0%"],
    ["No text in view", "2.90 s", "2.90 s", "0 s", "–", "–"],
], [5.2, 1.8, 2.3, 2.3, 2.3, 2.3])
figure(ROOT / "benchmark_report" / "2026-10-29" / "graphs" / "06_ocr_breakdown.png", 14.5,
       "Text reading on the Pi: where the time of each capture goes (from the Pi's benchmark report).")
bullets([
    "<b>One pass of the 8-bit text detector takes 2.5 to 3.0 s</b> on the Pi, against 9.9 to 11.2 s in full precision, "
    "and it made no letter error on these test pictures. On the rendered validation set it costs 0.57 points of letter "
    "error (3.15% against 2.58%), a stated exception to the project's 0.5-point rule, made for speed.",
    "<b>A picture with no text</b> answers \"no text found\" after one detector pass.",
    "<b>Real pages read with letter errors</b> through the camera, unlike these rendered test pictures; the field log "
    "of the app shows 12.2 to 22.2 s per printed page.",
    "<b>Steadier reading.</b> After a press the glass reads the sharpest frame of the next second: on validation text "
    "with simulated hand shake the letter error fell from 28.0% to 3.8% (savior_glass/results/ocr_bench/best_frame.json). "
    "<i>Real camera shake on the glass not measured.</i>",
])

H2("Other timings on the Pi")
table("Table 4: Loading, speech and camera", [
    ["What", "Time"],
    ["Loading the text reader (EasyOCR + 8-bit detector)", "14.4 s"],
    ["Currency mode, first switch into it / warm-up pass", "2.6 s / 1.9 s"],
    ["Emotion detector / object mode / watermark checker / counterfeit network, loading", "4.7 s / 1.6 s / 0.6 s / 0.3 s"],
    ["Camera (C270), open to first frame", "1.4 s"],
    ["Camera, time between frames (640×480)", "132 ms (7.6 frames a second)"],
    ["Speech: making the audio for a short Bangla / English sentence", "14.0 ms / 13.6 ms"],
    ["Object scan and announcement in use, app log", "0.74 s"],
    ["Memory: whole benchmark process (peak) / running app with all modes", "2.55 GB / 2.76 GB"],
], [11.0, 5.2])
table("Table 5: Board state", [
    ["", "Value"],
    ["Temperature, processor sensor, 30 minutes of non-stop load", "mean 82.8 °C, peak 87.0 °C"],
    ["Soft temperature limit active during that run", "55% of samples (fan at full speed)"],
    ["Temperature of the board surface in normal use (thermometer, team)", "57.8 to 62.3 °C"],
    ["Board power: idle / sustained load / reading text", "3.7 W / 7.8 W / about 9 W"],
    ["Under-voltage", "never"],
], [9.0, 7.2])
figure(ROOT / "benchmark_report" / "2026-10-29" / "graphs" / "03_thermal_power_timeline.png", 14.5,
       "Temperature, clock and power during the benchmark (from the Pi's benchmark report).")

# ------------------------------------------------------------------ 3
H("3. Speed measured on the laptop")
P("These were measured on the development laptop (AMD Ryzen 7 5800H, RTX 3050), with the CPU limited to four threads "
  "where stated, the number of cores of the Pi. <b>They are not Pi numbers.</b> Only ratios can be expected to carry over.")
table("Table 6: Laptop timings", [
    ["What", "Before", "After", "Accuracy"],
    ["Text detector alone, 1,200×900 picture, 4 CPU threads (PyTorch → ONNX Runtime)", "5.55 s", "2.63 s", "Identical output; validation error 2.14% both"],
    ["12-line page, 4 CPU threads (same change)", "9.99 s", "5.59 s", "Same"],
    ["One benchmark picture, 4 CPU threads (same change)", "2.30 s", "1.22 s", "Same"],
    ["Text detector, 8-bit (INT8), 4 CPU threads", "2.63 s", "1.29 s (page 4.28 s)", "Validation error +0.57 to +0.65 points"],
    ["12-line page, beam search → greedy decoder (11 October run)", "8.9 s", "6.5 s", "Validation error 2.18% → 2.14%"],
    ["12-line page spoken line by line: first line ready", "nothing spoken until the page is done", "about 5.5 s", "Unchanged"],
    ["Object scan through the glass path, 4 CPU threads (shrunk frame → frame as it is)", "163 ms", "173 ms", "mAP@0.5 0.493 → 0.521"],
    ["Speech: rendering one Bangla sentence of 1.5 to 4.3 s with the Piper voice (voice loaded)", "", "0.2 to 0.5 s", "Not timed on the Pi"],
    ["Taka detector on the laptop GPU / CPU (PyTorch)", "", "13.9 ms / 109 ms", ""],
    ["Taka detector, INT8 ONNX, laptop CPU", "", "101 ms", ""],
    ["Counterfeit network, 6 views, laptop GPU", "", "42.6 ms", ""],
    ["Watermark check, learned finder / SIFT, laptop CPU", "", "15.9 ms / 97.9 ms", ""],
], [6.8, 2.6, 3.0, 3.8])
figure(FIGS / "fig47_detector_onnx.png", 15.5, "Text detector through ONNX Runtime on four laptop CPU threads.")

# ------------------------------------------------------------------ 4
H("4. Results that corrected an earlier number")
bullets([
    "<b>Objects.</b> 0.604 mAP came from 250 photos; on all 5,000 the same model scores 0.551.",
    "<b>Text on real photos.</b> 2.5% letter error on rendered text; 39% (Bangla) and 62% (Latin script) on real street words.",
    "<b>UV check.</b> The first design accepted 92.8% of genuine photos, and in a simulation also 93% of fibre-free printed "
    "sheets. With a colour condition: 85.6% of genuine photos accepted, about 89% of simulated sheets rejected.",
    "<b>Counterfeit benchmark.</b> About 98% on the usual split; about 90% on prints never seen in training (95.0% with the watermark).",
    "<b>Emotion on the Pi.</b> 48 ms was a frame without a face; with a face it is 536 ms.",
])

H("5. Not measured")
bullets([
    "The rebuilt voice on the Pi: how it sounds, and how long the Piper voices take to render there.",
    "Voice commands spoken by people, with the glass's microphone, in a room or a street.",
    "The UV check and the watermark check through the C270 webcam (their thresholds come from phone and JaalTaka photos).",
    "Any counterfeit note under the UV lamp; any function through the glass camera in daylight.",
    "The online description mode (the account has no credit).",
    "Battery life.",
    "A study with visually impaired users (trial sessions have started; no result is reported).",
])
P("<i>Sources.</i> Pi figures: <font face='TNR'>benchmark_report/2026-10-29/REPORT.md</font> and its raw files."
  " Laptop and accuracy figures: the result files under "
  "<font face='TNR'>savior_glass/results/</font> and <font face='TNR'>paper_evidence/</font>.")


def footer(canvas, doc):
    canvas.setFont("TNR", 9.5)
    canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"SARA: results, benchmarks and latency  ·  page {doc.page}")


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.3 * cm, rightMargin=2.3 * cm, topMargin=2.0 * cm, bottomMargin=2.0 * cm,
                  title="SARA: results, benchmarks and latency", author="MD Abdun Fattah Lam, Mukut Protim Memo, Tanjila Hossain Tapti"
                  ).build(story, onFirstPage=footer, onLaterPages=footer)
print("written", OUT)
