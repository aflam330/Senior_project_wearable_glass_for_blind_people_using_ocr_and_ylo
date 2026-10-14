"""Build 'SARA: results, benchmarks and latency' as a PDF in Times New Roman.
Every number is copied from a result file of the project or from the Pi's benchmark report (13 October 2026)."""
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
    ["Taka recognition", "1,536 photographs by other people (Bangla Money)", "91.5% correct note value"],
    ["Taka recognition", "1,144 close-ups of folded notes held in a hand (NSTU-BDTAKA)", "18.5% correct note value"],
    ["Several notes in one picture", "900 pictures of 1, 2 or 3 test-split notes on new backgrounds", "Count right 99.7%; spoken total right 99.2%"],
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
    ["Voice commands (new, 14 October)", "39 command phrases spoken by 3 synthetic voices (117 commands); not people",
     "Carried out: 99.1% clean, 99.1% at 20 dB noise, 96.6% at 10 dB noise; never a wrong command"],
    ["Voice commands, false commands", "90 ordinary English sentences and 10 Bangla sentences per noise level; 60 s of noise alone",
     "English: 0 / 1 / 1 of 90 (clean / 20 dB / 10 dB); Bangla: 0 / 0 / 2 of 10; noise alone: 0"],
    ["Bangla voice (Piper), machine-listener check", "6 sentences given to an online Bangla speech recogniser",
     "2 exact, 2 with small differences, 2 with wrong words"],
], [4.2, 5.4, 6.6])

# ------------------------------------------------------------------ 2
H("2. Latency on the Raspberry Pi 5")
P("Two runs exist. The first (1 October 2026, 100 calls per row) ended at 77.1 °C with no throttling. The second "
  "(13 October 2026, 30 calls per row, after the code changes of 11–13 October) ran on a board that was already at "
  "73 °C when it started, reached 88.7 °C, and was throttled or clock-capped in 76% of the samples of the text-reading "
  "test, with the CPU down to 1.5 GHz from 2.4 GHz. <b>The second run is therefore the speed of a hot glass.</b> "
  "How much of the difference between the runs is heat cannot be told from these two runs.")
table("Table 2: Latency on the Raspberry Pi 5 (median; 95th percentile in brackets)", [
    ["Part of the glass", "13 October (hot board)", "1 October"],
    ["Currency check, end to end", "399 ms (524 ms)", "327 ms (560 ms)"],
    ["Taka detector, INT8 ONNX (used on the Pi)", "306 ms (446 ms)", "254 ms (339 ms)"],
    ["Taka detector, full-precision ONNX / PyTorch", "583 ms / 534 ms", "485 ms / 507 ms"],
    ["Watermark check, learned finder (used)", "64 ms (109 ms)", "50 ms (62 ms)"],
    ["Watermark check, SIFT alignment", "343 ms (381 ms)", "372 ms (854 ms)"],
    ["Counterfeit safe-check, one crop", "47 ms (78 ms)", "36 ms"],
    ["Counterfeit network, 1 view / 6 views", "43 ms / 98 ms", "37 ms / 90 ms"],
    ["Object scan as the app runs it (YOLOv8s)", "562 ms (610 ms)", "not run"],
    ["Object detector YOLOv8n alone", "227 ms (342 ms)", "207 ms (217 ms)"],
    ["Emotion, face in view", "680 ms (926 ms)", "not run"],
    ["Emotion, no face in view (face search only)", "48 ms (50 ms)", "48 ms (52 ms)"],
], [8.0, 4.4, 3.8])
figure(PI / "01_latency_all_modes.png", 15.5, "Latency of every mode and model on the Pi, 13 October 2026 (from the Pi's benchmark report).")

H2("Text reading on the Pi")
table("Table 3: Text reading on the Raspberry Pi 5, 13 October 2026 (hot board; median of 3 to 5 captures)", [
    ["Capture", "Total", "Finding the text", "Reading the letters", "First line spoken", "Letter errors"],
    ["English sign, 3 lines", "34.0 s", "30.1 s", "4.0 s", "–", "0%"],
    ["Bangla sign, 1 line", "35.4 s", "33.0 s", "1.0 s", "–", "0%"],
    ["Bangla sign, 3 lines", "38.6 s", "35.1 s", "3.5 s", "–", "0%"],
    ["Label, 6 lines, Bangla + English", "21.8 s", "13.3 s", "8.6 s", "15.1 s", "1.0%"],
    ["Page, 12 lines (640×480)", "36.5 s", "14.2 s", "22.3 s", "16.4 s", "0%"],
    ["Page, 12 lines (1280×720)", "28.0 s", "11.3 s", "16.7 s", "13.1 s", "0%"],
    ["No text in view", "27.5 s", "27.3 s", "0 s", "–", "–"],
    ["English sign, 3 lines, text found once (as on 1 October: 18.7 s)", "18.3 s", "", "", "", ""],
], [5.2, 1.8, 2.3, 2.3, 2.3, 2.3])
figure(PI / "03_ocr_response_time.png", 14.5, "Text reading on the Pi: response time of each capture and where it goes (from the Pi's benchmark report).")
P("<b>What this run exposed, and what was changed on 14 October.</b>")
bullets([
    "<b>Short signs ran the text detector twice</b> (34.0 s against 18.3 s with the text found once). It now runs once per "
    "capture. On the laptop the new path reads the same text as before on 16 benchmark pictures. <i>Not yet timed on the Pi.</i>",
    "<b>A blank picture took 27.5 s and returned 48 made-up letters.</b> A picture with no text boxes now answers "
    "\"no text found\" after one detector pass. <i>Not yet timed on the Pi.</i>",
    "<b>One pass of the text detector costs 11 to 18 s on this board.</b> The 8-bit text detector is now the default on the "
    "Pi. It costs 0.57 points of letter error on the rendered validation set (3.15% against 2.58%), which is over the "
    "project's own 0.5-point rule; this is a stated exception made for speed. <i>Its speed on the Pi is not measured yet.</i>",
    "<b>The 1 October emotion figure (48 ms) was a frame without a face.</b> With a face, one answer takes 680 ms.",
])

H2("Other timings on the Pi, 13 October 2026")
table("Table 4: Loading, speech, camera and the app's own log", [
    ["What", "Time"],
    ["Loading the text reader (EasyOCR, Bangla + English)", "8.7 s"],
    ["Currency mode, first switch into it / warm-up pass", "2.1 s / 1.5 s"],
    ["Emotion detector / object mode / watermark checker / counterfeit network, loading", "2.2 s / 0.7 s / 0.6 s / 0.3 s"],
    ["Camera, open to first frame", "1.4 s"],
    ["Camera, time between frames (640×480, indoors at night)", "132 ms (7.6 frames a second)"],
    ["Speech: making the audio for a short Bangla / English sentence", "17.5 ms / 14.3 ms"],
    ["App log, two sessions with the real camera: currency check (8)", "406 ms median"],
    ["App log: object scan and announcement (10)", "777 ms median"],
    ["App log: watermark check, guided, several frames (1)", "3.6 s"],
    ["App log: UV fibre check (3)", "0.8 to 1.4 s"],
    ["App log: one text capture", "39.9 s"],
    ["Memory: mode benchmark / text test / running app with all modes and preview", "1.44 GB / 2.33 GB / 2.99 GB"],
], [11.0, 5.2])
table("Table 5: Board state during the runs", [
    ["", "13 October", "1 October"],
    ["Peak temperature", "88.7 °C", "77.1 °C (30 minutes of mixed use)"],
    ["Throttled or clock-capped", "76% of samples in the text test; 11% in the mode benchmark", "never"],
    ["Lowest CPU clock logged", "1.5 GHz (full speed 2.4 GHz)", "not logged"],
    ["Under-voltage", "never", "never"],
    ["Peak memory in the benchmark", "1.44 GB", "1.48 GB"],
], [5.0, 6.2, 5.0])
figure(PI / "05_thermal_timeline.png", 14.5, "Temperature, throttling and CPU clock during the 13 October run (from the Pi's benchmark report).")

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
    "<b>Emotion on the Pi.</b> 48 ms was a frame without a face; with a face it is 680 ms.",
])

H("5. Not measured")
bullets([
    "The Pi after the 14 October changes: single detector pass, blank-picture answer, 8-bit text detector.",
    "The rebuilt voice on the Pi: how it sounds, and how long the Piper voices take to render there.",
    "Voice commands spoken by people, with the glass's microphone, in a room or a street.",
    "The Pi on a cool board with the current code, and the 30-minute sustained run with it.",
    "Any counterfeit note under the UV lamp; any function through the glass camera in daylight.",
    "The online description mode (the account has no credit).",
    "Battery life.",
    "A study with visually impaired users (trial sessions have started; no result is reported).",
])
P("<i>Sources.</i> Pi figures: <font face='TNR'>benchmark_report/REPORT.md</font> and its raw files (13 October), and the "
  "1 October run in the thesis. Laptop and accuracy figures: the result files under "
  "<font face='TNR'>savior_glass/results/</font> and <font face='TNR'>paper_evidence/</font>.")


def footer(canvas, doc):
    canvas.setFont("TNR", 9.5)
    canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"SARA: results, benchmarks and latency  ·  page {doc.page}")


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.3 * cm, rightMargin=2.3 * cm, topMargin=2.0 * cm, bottomMargin=2.0 * cm,
                  title="SARA: results, benchmarks and latency", author="MD Abdun Fattah Lam, Mukut Protim Memo, Tanjila Hossain Tapti"
                  ).build(story, onFirstPage=footer, onLaterPages=footer)
print("written", OUT)
