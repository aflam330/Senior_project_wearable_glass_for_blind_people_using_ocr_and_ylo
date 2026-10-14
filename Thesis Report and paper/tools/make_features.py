"""Build 'SARA: all features' as a PDF in Times New Roman. Every number is copied from the project's result files."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

REPORT = Path(r"E:\Final SP\Thesis Report and paper\thesis report\My report")
FIG = REPORT / "DIagram fig tab"
OUT = REPORT / "SARA_Features.pdf"
F = r"C:\Windows\Fonts"
for n, f in (("TNR", "times.ttf"), ("TNR-B", "timesbd.ttf"), ("TNR-I", "timesi.ttf"), ("TNR-BI", "timesbi.ttf")):
    pdfmetrics.registerFont(TTFont(n, F + "\\" + f))
pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")

body = ParagraphStyle("body", fontName="TNR", fontSize=11.5, leading=16, alignment=TA_JUSTIFY, spaceAfter=6)
h1 = ParagraphStyle("h1", fontName="TNR-B", fontSize=14, leading=18, spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#1F4E79"))
cell = ParagraphStyle("cell", fontName="TNR", fontSize=10, leading=12.5)
cellb = ParagraphStyle("cellb", parent=cell, fontName="TNR-B")
centre = ParagraphStyle("centre", fontName="TNR", fontSize=11.5, leading=16, alignment=TA_CENTER)
title = ParagraphStyle("title", fontName="TNR-B", fontSize=17, leading=22, alignment=TA_CENTER)
cap = ParagraphStyle("cap", fontName="TNR-B", fontSize=10.5, leading=13, alignment=TA_CENTER, spaceBefore=6, spaceAfter=4)
note = ParagraphStyle("note", fontName="TNR-I", fontSize=10, leading=13, alignment=TA_CENTER, spaceAfter=8)
story = []
P = lambda t: story.append(Paragraph(t, body))
H = lambda t: story.append(Paragraph(t, h1))


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


def figure(name, width_cm, caption):
    p = FIG / name
    if p.exists():
        story.append(Image(str(p), width=width_cm * cm, height=width_cm * cm, kind="proportional"))
        story.append(Paragraph(caption, note))


story.append(Paragraph("SARA (Smart Assistive Reading Aid)", title))
story.append(Paragraph("An Offline Smart Glass for Visually Impaired People that Reads Bangla and English Text, Detects "
                       "Objects, and Recognises and Checks Bangladeshi Taka Notes", centre))
story.append(Spacer(1, 4))
story.append(Paragraph("<b>All features of the project</b> &nbsp;·&nbsp; 14 October 2026", centre))
story.append(Paragraph("MD Abdun Fattah Lam (2220014), Mukut Protim Memo (2020086), Tanjila Hossain Tapti (2130561)<br/>"
                       "Supervisor: Mohammad Shidujaman, PhD &nbsp;·&nbsp; Department of CSE, Independent University, Bangladesh", centre))
story.append(Spacer(1, 6))

H("1. What SARA is")
P("SARA is a smart glass for blind and low-vision people in Bangladesh. A camera on a spectacle frame, a Raspberry Pi 5, "
  "five push-buttons and an earphone let the user read printed Bangla and English, hear what objects are in front, "
  "know which Taka note is in the hand, and get careful advice on whether the note is genuine. It speaks Bangla, and "
  "everything except one optional mode works without the internet.")
figure("worn_200.jpg", 5.2, "The prototype worn during a trial session.")

H("2. Features at a glance")
table("Table 1: The modes of the glass", [
    ["Feature", "What the user gets", "How it works", "Measured result"],
    ["Text reading", "Printed Bangla and English read aloud; READ repeats it; whole pages are spoken line by line",
     "EasyOCR, with only the main block of text kept, tuned settings and clean-up rules",
     "Rendered benchmark: letter error 9.3% → 2.3%. Real street photos: 39% → 29% (Bangla), 62% → 51% (Latin script)"],
    ["Object announcement", "Names of objects in front, in Bangla, every 1.5 s", "YOLOv8 with a Bangla name for each of 80 objects",
     "mAP@0.5 0.551 on 5,000 test photos (0.521 through the glass path); 207 ms on the Pi 5 with the nano model"],
    ["Taka recognition", "Which of the 9 notes (2 to 1,000 Taka) is in view, with a short vibration",
     "YOLOv8s trained on 22,333 automatically labelled pictures", "mAP@0.5 0.995 on held-out pictures; 91.5% on 1,536 photos by other people"],
    ["Safe counterfeit check (500, 1,000 Taka)", "\"Likely genuine\" or \"check by hand\"; never \"counterfeit\"",
     "Multi-view network (PRMVT) with a threshold fixed on validation notes", "0 of 88 test counterfeits passed as genuine"],
    ["Watermark check", "Spoken guidance to hold the note against a light, then \"watermark clear\" or \"check by hand\"",
     "Learned watermark finder + MobileNetV2 classifier", "92.6% on unseen counterfeit prints; 50 ms on the Pi 5"],
    ["UV fibre check (200, 500, 1,000 Taka) — new", "\"UV marks found, likely genuine\" or \"UV marks not found, check by hand\"",
     "Counts the glowing security fibres under a UV lamp (Section 4)", "Prototype: see Section 4"],
    ["Emotion", "The facial expression of the person in front", "YuNet face finder + EfficientNet-V2-S", "86.5% on the RAF-DB test set"],
    ["Online description (optional)", "A spoken description of the scene", "A cloud vision model; needs the internet and a key", "Not part of the offline results"],
    ["Voice commands — new", "Say \"read text\", \"check money\", \"objects\", \"repeat\", \"louder\", \"stop\" instead of pressing buttons",
     "Offline recogniser (Vosk) limited to 39 command phrases; ignores its own voice", "Synthetic speech: 99.1% carried out, no wrong command (Section 5)"],
], [3.0, 4.3, 4.6, 4.1])

H("3. New methods developed in this project")
bullets([
    "<b>Training pictures without hand labelling.</b> 5,073 note photos are pasted onto real photographs with random "
    "angle, size, light and blur, and the program writes each box label itself: 22,333 labelled pictures, no manual work.",
    "<b>Prefix training (PRMVT).</b> A counterfeit model trained on six views fails when the user captures fewer. Training "
    "on a random number of views (1 to 6) keeps it accurate: 98.2% with one view against 71.2% (three training runs).",
    "<b>Print-disjoint test.</b> We found that most counterfeit notes in the public dataset are copies of a few prints "
    "(279 of 322 readable 500 Taka counterfeits carry one serial number), and built a test in which every counterfeit "
    "print is new. On it, adding the watermark check raises accuracy from about 90% to 95.0%.",
    "<b>DMWR watermark method.</b> The lamp is removed from the back-lit photo (Q = I / (G∗I)), the watermark is compared "
    "with a \"genuine portrait\" of the note value built from genuine notes only, and what is left over is classified. "
    "With 47,818 parameters (0.21 MB) it reaches 91.5% on unseen prints, statistically level with a MobileNetV2 that is "
    "46 times larger.",
    "<b>Safe spoken verdict.</b> The threshold is the highest score any validation counterfeit received, so a new "
    "counterfeit passes with probability 1/(n+1) = 1.1%. The glass never accuses.",
    "<b>UV fibre check (ENFR).</b> Described next.",
])
figure("fig_dmwr_steps.png", 12.5, "DMWR on a genuine and a counterfeit 500 Taka test note: photo, lamp removed, genuine portrait, residual.")

H("4. New: UV security-fibre check")
P("<b>What the UV light shows.</b> Our own photographs of genuine 200, 500 and 1,000 Taka notes under a UV lamp show "
  "short fibres that glow, on paper that stays dull, while the plain white surface beside the note glows as a whole. "
  "So a genuine note <i>has</i> UV marks; a note without them, or one whose paper glows, is the suspicious one.")
P("<b>Method (excitation-normalised fluorescence residual, ENFR).</b> The lamp's own light reaches the camera as blue. "
  "Anything that fluoresces gives off another colour. For the paper of one note, green and red are a fixed share of blue "
  "(a = median G/B, b = median R/B). What that share does not explain is fluorescence: E = max(G − a·B, R − b·B). Thin "
  "bright structures in E, measured relative to the local lamp level, are the fibres; the check counts those that are "
  "thin and elongated inside the note box. Because a, b and the lamp level are measured in the same picture, the "
  "brightness and distance of the lamp cancel out. The camera's colour balance does not, so the thresholds belong to one camera.")
figure("fig_uv_fibres.jpg", 16, "Genuine notes under UV (left of each pair) and the fibres found by the check (green, right).")
table("Table 2: UV fibre check on 118 phone photographs of genuine notes (111 with a note box found)", [
    ["What was measured", "Result"],
    ["<b>Blocked 5-fold test</b> (each block of consecutive photos judged with a threshold set on the other blocks): genuine notes accepted", "<b>103 of 111 (92.8%)</b>; 95% range 86–96%"],
    ["&nbsp;&nbsp;200 Taka / 500 Taka / 1,000 Taka", "45 of 45 / 36 of 40 / 22 of 26"],
    ["&nbsp;&nbsp;Plain-surface patches beside the note rejected", "71 of 71"],
    ["First test, one split (threshold set on front-side photos, tested on back-side ones)", "29 of 43 (67%)"],
    ["Same test photos made half as bright / 1.5 times brighter", "67% / 74% accepted in the first test (brightness changes little)"],
    ["Same test photos slightly out of focus / at half resolution", "30% / 44% accepted in the first test (sharpness matters most)"],
], [10.0, 6.0])
P("<b>Reading these numbers.</b> The check recognises about nine in ten photographs of genuine notes as genuine. "
  "The misses are photos of the back of the note or from far away, where only a few faint fibres are visible. "
  "The thresholds are low (a handful of fibres), and the only fibre-free examples are smooth surfaces, so printed "
  "paper without fibres still has to be tested. "
  "On the glass, the lamp and camera should sit in a small hood about 10 to 15 cm from the note. Two things are not "
  "measured yet: real counterfeit notes (none were available to us) and the glass's own camera. For that reason the "
  "check is switched off by default, and it says \"check by hand\" instead of \"counterfeit\".")

H("5. Speech and interaction")
bullets([
    "The language of each sentence is chosen automatically. Bangla and English are spoken with Piper voices when "
    "they are installed and with espeak-ng otherwise.",
    "<b>Rebuilt on 14 October 2026 after the voice came out broken on the Pi.</b> Speech used to be produced and played "
    "at the same time, so the sound ran dry while text reading used all four cores; and the English voice was played "
    "38% too fast. Every sentence is now finished first and then played at its own rate, the next sentence is prepared "
    "while the current one is heard, and Bangla has a natural (Piper) voice. Still to be heard and timed on the Pi.",
    "<b>Voice commands (new).</b> With a microphone, the glass takes spoken commands without the internet: "
    "\"read text\", \"check money\", \"objects\", \"describe\", \"emotion\", \"next mode\", \"capture\", \"repeat\", "
    "\"louder\", \"quieter\", \"stop\". A mode word selects the mode and takes one capture. The glass does not listen "
    "while it speaks. Tested with three synthetic voices: 116 of 117 commands carried out with no noise or light noise, "
    "113 of 117 in heavy noise, never a wrong command; 1 false command in 90 ordinary English sentences in noise. "
    "Not yet tested with people or on the Pi. The command words are English: no small offline Bangla recogniser exists.",
    "<b>Fixed on 12 October 2026:</b> Bangla reading used to break up. Every comma, digit and full stop was cut out as a "
    "separate piece with its own pause, long texts lost their first sentences, and each Bangla piece first tried a voice "
    "that is known to fail on the Pi. Sentences now stay whole, long texts are spoken complete and in order, and the "
    "failing voice is tried once per run.",
    "Five buttons found by touch: MODE, ACTION, READ, VOLUME UP, VOLUME DOWN. Every change is spoken; a vibration confirms a note.",
    "A field log and a spoken questionnaire for trials with visually impaired users.",
])

H("6. Checked on 13 October 2026")
bullets([
    "<b>Faster text detector.</b> The text detector now runs through ONNX Runtime: 2.6 s instead of 5.6 s on four laptop "
    "CPU threads, a 12-line page in 5.6 s instead of 10.0 s, with the same validation error (2.14%). An 8-bit version is "
    "faster again (page 4.3 s) but costs 0.65 points of error, over our 0.5-point rule, so it is optional.",
    "<b>Text on real photographs.</b> On 1,000 real word photos (ICDAR 2019 MLT), read once with no tuning, the letter "
    "error was 41.7% for Bangla and 64.9% for Latin script, against 2.4% on our rendered benchmark. A setting chosen on a "
    "separate sample of real words brought it to 39.2% and 62.5%, and cut the words left unread from 25% to 17% and "
    "from 48% to 38%.",
    "<b>Recogniser trained on real words (14 October).</b> The letter recogniser was retrained on real word photos and "
    "then blended half and half with the original weights, so that printed text is not forgotten. Read once on the "
    "1,000 test words: Bangla letter error 39.2% → 28.7%, Latin script 62.5% → 50.6%; printed text 2.46% → 2.27%. Same "
    "network, same speed; now the default. Reading real signs is still the hardest part of the text mode.",
    "<b>Text reading on the Pi feels faster to use.</b> One detector pass per capture (a short sign was paying for two), "
    "an immediate \"no text found\" on an empty picture, the 8-bit detector on the Pi, a spoken \"reading\" at once "
    "and a soft tick while it works.",
    "<b>Objects, measured on all 5,000 COCO test photos.</b> YOLOv8s scores 0.551 (the earlier 0.604 came from a "
    "250-photo sample). The object mode used to shrink each camera picture before a detector that enlarges it again; "
    "removing that raised the score through the glass path from 0.493 to 0.521 at no cost.",
    "<b>Online description.</b> Not measured: the cloud account has no credit, and a local model could not be loaded "
    "on the laptop (blocked by a Windows security policy).",
])

H("7. Speed on the Raspberry Pi 5")
table("Table 3: Measured on a Raspberry Pi 5 (8 GB), 100 calls per part", [
    ["Part", "Typical time"],
    ["Currency answer (find the note + safe check)", "327 ms"],
    ["Watermark check", "50 ms"],
    ["Object scan", "207 ms"],
    ["Emotion", "48 ms"],
    ["Text reading, one picture", "18.7 s (measured before the speed-ups below)"],
], [10.0, 6.0])
P("Thirty minutes of mixed use stayed within 1.48 GB of memory, peaked at 77.1 °C and never throttled. For pages, a "
  "greedy decoder reads a 12-line page 27% faster with the same accuracy, and lines are spoken as soon as they are "
  "read (measured on a laptop limited to four cores; still to be timed on the Pi). A faster setting (OCR_CANVAS_SIZE=640) "
  "was about three times faster on the Pi at about half a point more error and can be switched on when speed matters more.")

H("8. What is still open")
bullets([
    "UV check: photographs through the glass with the lamp in a hood, and tests on real counterfeit notes with a bank or the police.",
    "Timing the new text detector and speech changes on the Pi, and tuning the glass camera for small text and notes.",
    "Text-finding settings and training data for real signs.",
    "Completing the trials with visually impaired users that have started.",
])
P("<i>On a patent.</i> DMWR and the UV fibre method were designed in this project, but UV fibres and quotient images are "
  "known ideas in banknote checking. Whether the specific methods can be patented needs a prior-art search by a patent "
  "attorney; this document makes no claim about that.")


def footer(canvas, doc):
    canvas.setFont("TNR", 9.5)
    canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"SARA: all features  ·  page {doc.page}")


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.3 * cm, rightMargin=2.3 * cm, topMargin=2.0 * cm, bottomMargin=2.0 * cm,
                  title="SARA: all features", author="MD Abdun Fattah Lam, Mukut Protim Memo, Tanjila Hossain Tapti"
                  ).build(story, onFirstPage=footer, onLaterPages=footer)
print("written", OUT)
