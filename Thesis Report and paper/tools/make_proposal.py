"""Build the SARA project proposal as a PDF in Times New Roman."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

REPORT = Path(r"E:\Final SP\Thesis Report and paper\thesis report\My report")
OUT = REPORT / "proposal" / "SARA_Project_Proposal.pdf"
F = r"C:\Windows\Fonts"
pdfmetrics.registerFont(TTFont("TNR", F + r"\times.ttf"))
pdfmetrics.registerFont(TTFont("TNR-B", F + r"\timesbd.ttf"))
pdfmetrics.registerFont(TTFont("TNR-I", F + r"\timesi.ttf"))
pdfmetrics.registerFont(TTFont("TNR-BI", F + r"\timesbi.ttf"))
pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")

body = ParagraphStyle("body", fontName="TNR", fontSize=12, leading=17, alignment=TA_JUSTIFY, spaceAfter=8)
h1 = ParagraphStyle("h1", fontName="TNR-B", fontSize=14, leading=18, spaceBefore=12, spaceAfter=6)
cell = ParagraphStyle("cell", fontName="TNR", fontSize=10.5, leading=13)
cellb = ParagraphStyle("cellb", parent=cell, fontName="TNR-B")
centre = ParagraphStyle("centre", fontName="TNR", fontSize=12, leading=17, alignment=TA_CENTER)
title = ParagraphStyle("title", fontName="TNR-B", fontSize=18, leading=24, alignment=TA_CENTER)
cap = ParagraphStyle("cap", fontName="TNR-B", fontSize=11, leading=14, alignment=TA_CENTER, spaceBefore=6, spaceAfter=4)
ref = ParagraphStyle("ref", fontName="TNR", fontSize=11, leading=14.5, leftIndent=18, firstLineIndent=-18, spaceAfter=4)

story = []


def P(text):
    story.append(Paragraph(text, body))


def H(text):
    story.append(Paragraph(text, h1))


def bullets(items, numbered=False):
    story.append(ListFlowable([ListItem(Paragraph(t, body), leftIndent=18) for t in items],
                              bulletType="1" if numbered else "bullet", bulletFontName="TNR", bulletFontSize=12, leftIndent=18))


def table(caption, rows, widths):
    story.append(Paragraph(caption, cap))
    data = [[Paragraph(c, cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story.append(t)
    story.append(Spacer(1, 8))


# ------------------------------------------------------------------ title page
story.append(Spacer(1, 1.2 * cm))
logo = REPORT / "DIagram fig tab" / "iub_logo.png"
if logo.exists():
    story.append(Image(str(logo), width=6.5 * cm, height=6.5 * cm, kind="proportional"))
story.append(Spacer(1, 0.8 * cm))
story.append(Paragraph("Senior Project Proposal", centre))
story.append(Spacer(1, 0.5 * cm))
story.append(Paragraph("SARA (Smart Assistive Reading Aid): An Offline Smart Glass for Visually Impaired People that "
                       "Reads Bangla and English Text, Detects Objects, and Recognises and Checks Bangladeshi Taka Notes", title))
story.append(Spacer(1, 1.2 * cm))
for line in ["Prepared by", "<b>MD Abdun Fattah Lam</b>", "ID: 2220014", "<b>Mukut Protim Memo</b>", "ID: 2020086",
             "<b>Tanjila Hossain Tapti</b>", "ID: 2130561", "", "Spring 2026", "", "Supervisor",
             "<b>Mohammad Shidujaman, PhD</b>", "Assistant Professor", "Department of Computer Science and Engineering",
             "School of Engineering, Technology and Sciences", "Independent University, Bangladesh"]:
    story.append(Paragraph(line or "&nbsp;", centre))
story.append(PageBreak())

# ------------------------------------------------------------------ body
H("1. Introduction")
P("A blind person in Bangladesh who buys medicine, hands over a banknote and receives change has to ask someone what "
  "the label says, which note came back, and whether that note is real. The World Health Organization estimates that "
  "at least 2.2 billion people live with some form of vision impairment, and a large share of them live in low- and "
  "middle-income countries [1]. For these people, reading a sign, recognising an object and handling cash are daily "
  "tasks that depend on the help of others.")
P("Camera-based assistive devices can take over part of this help. Commercial products such as OrCam MyEye [2] show "
  "that the idea works, but they are expensive, are built mainly for English, and do not know Bangladeshi Taka. "
  "Low-cost research prototypes built on Raspberry Pi boards usually read English only and speak through an online "
  "service [3, 4]. We propose <i>SARA (Smart Assistive Reading Aid)</i>, a wearable smart glass that speaks Bangla, reads Bangla and English "
  "text, names everyday objects, recognises Taka notes and gives careful advice about counterfeit notes, and that "
  "does its main work on the device without the internet.")

H("2. Problem Statement")
P("A blind or low-vision person in Bangladesh has no affordable wearable device that reads Bangla text aloud, tells "
  "which Taka note is in the hand, and gives trustworthy advice on whether that note may be counterfeit, without a "
  "phone, a screen or an internet connection. Three things make this hard:")
bullets([
    "<b>Language and currency.</b> Existing devices are designed for English and for other currencies. Bangla script, "
    "with its conjuncts and vowel signs, is harder for open OCR engines than English.",
    "<b>Connectivity and cost.</b> Many systems send pictures to a server or need a paid service. This fails where "
    "the connection is poor and adds a cost that many users cannot carry.",
    "<b>Trust.</b> A blind user cannot check the device's answer by sight. A wrong denomination or a wrong "
    "counterfeit verdict can cost money or lead to a false accusation, so the device must say when it is not sure.",
])

H("3. Objectives")
P("The aim is to design, build and evaluate a working prototype of SARA. The specific objectives are:")
bullets([
    "To build a wearable prototype on a Raspberry Pi 5 with a camera, operated only through physical buttons and speech.",
    "To read printed Bangla and English text aloud using on-device OCR and offline text-to-speech.",
    "To announce everyday objects in front of the user in Bangla.",
    "To recognise all nine denominations of Bangladeshi Taka (2, 5, 10, 20, 50, 100, 200, 500 and 1000) in ordinary "
    "camera pictures, using training data that needs no manual labelling.",
    "To study a camera-based check for counterfeit 500 and 1000 Taka notes, and to design spoken feedback that is "
    "safe for a user who cannot verify it.",
    "To measure the accuracy of every function on held-out data, and its speed, memory use and temperature on the "
    "Raspberry Pi 5.",
    "To try the prototype with visually impaired users and collect their feedback.",
], numbered=True)

H("4. Scope")
P("<b>In scope:</b> text reading (Bangla and English), object announcement, Taka denomination recognition, a "
  "counterfeit check for 500 and 1000 Taka notes, Bangla and English speech output, a five-button interface, a "
  "3D-printed spectacle frame, and an evaluation of each function. An optional mode that describes a scene through "
  "an online vision model may be added; it will need the internet and will not be required by the other modes.")
P("<b>Out of scope:</b> navigation and obstacle distance, GPS, handwriting recognition, and voice-command control. "
  "The glass is an aid. It will not replace a white cane, a guide, or the counterfeit-detection equipment of a bank.")

H("5. Related Work and Gap")
P("Reading aids for blind people have a long history, from devices that turned print into vibration to finger-worn "
  "cameras and smart glasses [2, 5]. Raspberry Pi readers combine a camera, an OCR engine and a speech engine at low "
  "cost, but mostly for English and with online speech [3, 4]. Recent glasses add scene description and face "
  "recognition by sending pictures to a remote server [6]. For Bangladeshi currency, detectors have been trained on "
  "a few thousand hand-labelled images [7], and a public dataset of genuine and counterfeit notes with six views per "
  "note is available [8]. Fast object detectors [9] and multilingual OCR engines [10] make on-device operation possible.")
P("What is missing is the combination: one wearable device that reads Bangla, recognises Taka, speaks Bangla and "
  "works offline, together with an honest measurement of how well each part works on the target hardware and with "
  "its users. This project aims to fill that gap.")

H("6. Proposed Methodology")
P("The work will follow four phases: requirement analysis, data and model development, system integration, and "
  "evaluation. Each function will be a separate mode that the user selects with one button.")
bullets([
    "<b>Text reading.</b> EasyOCR [10] with Bangla and English recognisers. The capture button stores the recognised "
    "text and a separate button speaks it, so the user holds the camera still only for a moment. We will test "
    "pre-processing, region selection and post-processing on a test set built for the project.",
    "<b>Object announcement.</b> A COCO-pretrained YOLOv8 detector [9] scans the scene at a fixed interval and speaks "
    "the most confident objects in Bangla, with a waiting time per object so that it does not repeat itself.",
    "<b>Currency recognition.</b> Public note images [11] will be pasted onto real photographs with random rotation, "
    "scale, lighting and blur, so that bounding-box labels are produced automatically. A YOLOv8 detector will be "
    "trained on these images and tested on photographs from other sources.",
    "<b>Counterfeit check.</b> A multi-view classifier will be trained on the JaalTaka dataset [8]. Because a blind "
    "user cannot guarantee how many views are captured, we will study training that keeps the model accurate with "
    "fewer views. The spoken result will be cautious: the glass will ask for a check by hand when it is not sure. "
    "We will also check the banknote watermark in a photograph of the note held against a light, by removing the "
    "effect of the lamp and comparing the watermark with a reference built from genuine notes of the same value.",
    "<b>Speech and control.</b> Offline text-to-speech (Piper and espeak-ng) with separate voices for Bangla and "
    "English segments; five push-buttons for mode, action, read and volume; every change of state is spoken.",
])
P("<b>Evaluation.</b> Models will be trained on a training split, choices will be made on a validation split, and "
  "the test split will be used only for the final figures. Detection will be reported as precision, recall and mean "
  "average precision; reading as character and word error rate; the counterfeit check as accuracy and as the number "
  "of counterfeit notes wrongly passed. Latency, memory and temperature will be measured on the Raspberry Pi 5. "
  "Results that do not support our expectations will be reported as well.")

H("7. Hardware and Software Requirements")
table("Table 1: Hardware and estimated cost", [
    ["Component", "Role", "Estimated cost (BDT)"],
    ["Raspberry Pi 5 (4 to 8 GB) with cooler", "Runs all models and the application", "18,000 – 24,000"],
    ["Raspberry Pi camera module", "Captures the scene", "1,500 – 2,500"],
    ["Five push-buttons", "Mode, action, read, volume up, volume down", "100 – 200"],
    ["Earphone or bone-conduction headset with amplifier", "Speech output", "800 – 1,500"],
    ["MicroSD card (32 to 64 GB)", "Operating system and model files", "800 – 1,200"],
    ["USB power bank", "Portable power", "1,500 – 3,000"],
    ["3D-printed spectacle frame", "Holds the camera and the board", "1,500 – 2,500"],
    ["Wiring, connectors, vibration motor", "Supporting electronics", "400 – 700"],
    ["Total (approximate)", "", "24,600 – 35,600"],
], [6.4, 6.0, 3.6])
P("The costs are our estimates from local market prices and will be confirmed at purchase. Software will be open "
  "source: Python, OpenCV, PyTorch, Ultralytics YOLOv8, EasyOCR, ONNX Runtime, Piper and espeak-ng. Model training "
  "will be done on laptops with NVIDIA GPUs that the team already has.")

H("8. Work Plan")
table("Table 2: Work plan over one semester (16 weeks)", [
    ["Weeks", "Activity", "Output"],
    ["1 – 2", "Literature review, requirement analysis, purchase of parts", "Requirement list, parts in hand"],
    ["3 – 5", "Currency data pipeline and detector training", "Trained Taka detector with test results"],
    ["4 – 7", "Text reading and object modes; speech pipeline", "Working modes on a laptop with a webcam"],
    ["6 – 9", "Counterfeit study on the JaalTaka dataset", "Trained classifier, safe-feedback rule"],
    ["8 – 10", "Frame design and 3D printing; button wiring", "First wearable prototype"],
    ["10 – 12", "Integration and deployment on the Raspberry Pi 5", "Glass running all modes from the buttons"],
    ["12 – 14", "Measurement on the Pi; trial sessions with users", "Latency and accuracy figures, user feedback"],
    ["14 – 16", "Report writing and presentation", "Thesis report and demonstration"],
], [2.0, 8.0, 6.0])

H("9. Expected Outcomes")
bullets([
    "A working prototype of an assistive glass that speaks Bangla, with text, object and currency modes.",
    "A labelled Taka detection dataset produced without manual annotation, and a detector trained on it.",
    "A study of camera-based counterfeit checking, with feedback wording that is safe for a blind user.",
    "Measured accuracy for every function and measured speed on the Raspberry Pi 5.",
    "Feedback from visually impaired users on ease of use, clarity of speech and usefulness.",
    "A thesis report and the source code, so that a later team can continue the work.",
])

H("10. Risks and How We Will Handle Them")
table("Table 3: Main risks", [
    ["Risk", "Plan"],
    ["Models are too slow on the Raspberry Pi 5", "Use small models, quantised ONNX exports, and run heavy tasks only on request"],
    ["Bangla OCR is not accurate enough", "Test pre- and post-processing and other engines; report the measured error honestly"],
    ["A detector trained on synthetic images fails on real photographs", "Use real photographs as backgrounds; test on photographs from independent sources"],
    ["Few counterfeit notes are available", "Use the public JaalTaka dataset; keep the spoken verdict cautious"],
    ["Camera quality on the glass is lower than on a laptop", "Test with the glass camera early; adjust resolution or change the camera"],
    ["Hard to recruit visually impaired participants in time", "Start contact early; report honestly how many sessions were completed"],
], [7.0, 9.0])

H("11. Ethical Considerations")
P("Pictures will be processed on the device and will not be stored or sent anywhere in the offline modes. "
  "Participants in user trials will be told what the device does in spoken form, will take part only with their "
  "consent, and will be recorded under a number and not a name. Photographs of participants will be used only with "
  "their permission. Because a wrong counterfeit verdict could harm the user or another person, the glass will "
  "present its result as advice and will not accuse a note of being counterfeit.")

H("12. Conclusion")
P("SARA aims to give a blind person in Bangladesh a little more independence in three daily tasks: reading, "
  "knowing what is in front of them, and handling cash. The project combines existing open models with data and "
  "interaction designed for Bangla-speaking users, on hardware that is affordable and works without the internet. "
  "We ask for approval to carry out this work as our senior project under the supervision of "
  "Dr. Mohammad Shidujaman.")

H("References")
for r in [
    "[1] World Health Organization, “Blindness and Vision Impairment: Fact Sheet,” 2023. https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment",
    "[2] OrCam Technologies, “OrCam MyEye.” https://www.orcam.com",
    "[3] “Reading Aid and Translator with Raspberry Pi for Blind People,” IEEE conference paper, 2023. https://ieeexplore.ieee.org/abstract/document/10113042",
    "[4] “Vision Voice: A Raspberry Pi-Based Text-to-Audio Converter for the Visually Impaired,” IEEE conference paper, 2024. https://ieeexplore.ieee.org/abstract/document/10664121",
    "[5] J. G. Linvill and J. C. Bliss, “A Direct Translation Reading Aid for the Blind,” Proceedings of the IEEE, vol. 54, no. 1, pp. 40–51, 1966.",
    "[6] D. D. Brilli, E. Georgaras, S. Tsilivaki, N. Melanitis and K. Nikita, “AIris: An AI-powered Wearable Assistive Device for the Visually Impaired,” arXiv:2405.07606, 2024.",
    "[7] M. J. A. Rafi, M. Rony and N. Majadi, “NSTU-BDTAKA: An Open Dataset for Bangladeshi Paper Currency Detection and Recognition,” Data in Brief, vol. 55, 110701, 2024.",
    "[8] B. R. C. Piyas, F. J. Tisha, S. Rahman, S. Islam and B. R. C. Preenon, “JaalTaka: A Benchmark Dataset for Detecting Counterfeit Bangladeshi Banknotes,” Mendeley Data, V2, 2025. doi:10.17632/2m7wk5cy4c.2",
    "[9] G. Jocher, A. Chaurasia and J. Qiu, “Ultralytics YOLOv8,” 2023. https://github.com/ultralytics/ultralytics",
    "[10] JaidedAI, “EasyOCR: Ready-to-use OCR with 80+ Supported Languages.” https://github.com/JaidedAI/EasyOCR",
    "[11] “A Diverse Image Dataset for Bangladeshi Currency Recognition (BanglaTaka),” Mendeley Data. https://data.mendeley.com/datasets/3cv2sypkkh/1",
]:
    story.append(Paragraph(r, ref))


def footer(canvas, doc):
    if doc.page > 1:
        canvas.setFont("TNR", 10)
        canvas.drawCentredString(A4[0] / 2, 1.3 * cm, str(doc.page))


OUT.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm, topMargin=2.2 * cm, bottomMargin=2.2 * cm,
                  title="SARA: Senior Project Proposal", author="MD Abdun Fattah Lam, Mukut Protim Memo, Tanjila Hossain Tapti"
                  ).build(story, onFirstPage=footer, onLaterPages=footer)
print("written", OUT)
