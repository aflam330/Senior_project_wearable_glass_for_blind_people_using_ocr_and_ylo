"""Compile the professional project report to PDF using ReportLab (detailed edition)."""

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, ListFlowable, ListItem,
)

ROOT = Path("C:/Users/memod/OneDrive/Desktop/currency")
ASSETS = ROOT / "results/report_assets"
RESULTS = ROOT / "results/training_v2"
RUNS = ROOT / "runs/detect/train-4"
OUT = ROOT / "Bangla_Currency_Detection_Report.pdf"

NAVY = colors.HexColor("#1F3A5F")
BLUE = colors.HexColor("#2E5EAA")
GREEN = colors.HexColor("#3A7D44")
LIGHT = colors.HexColor("#EAF0F8")
CODEBG = colors.HexColor("#F3F4F6")

styles = getSampleStyleSheet()

def S(name, **kw):
    styles.add(ParagraphStyle(name, parent=styles["Normal"], **kw))

S("Body", fontSize=10, leading=15, alignment=TA_JUSTIFY, spaceAfter=8, fontName="Helvetica")
S("H1", fontSize=15, leading=19, spaceBefore=16, spaceAfter=8, textColor=NAVY, fontName="Helvetica-Bold")
S("H2", fontSize=12, leading=16, spaceBefore=11, spaceAfter=5, textColor=BLUE, fontName="Helvetica-Bold")
S("H3", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=3, textColor=GREEN, fontName="Helvetica-Bold")
S("Caption", fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=colors.grey, fontName="Helvetica-Oblique", spaceAfter=10, spaceBefore=3)
S("TitleBig", fontSize=26, leading=31, alignment=TA_CENTER, textColor=NAVY, fontName="Helvetica-Bold")
S("Subtitle", fontSize=13, leading=18, alignment=TA_CENTER, textColor=BLUE, fontName="Helvetica")
S("Meta", fontSize=10, leading=15, alignment=TA_CENTER, textColor=colors.grey)
S("AbstractBody", fontSize=10, leading=15, alignment=TA_JUSTIFY, fontName="Helvetica", leftIndent=8, rightIndent=8)
S("Ref", fontSize=9, leading=13, alignment=TA_LEFT, spaceAfter=5, fontName="Helvetica", leftIndent=14, firstLineIndent=-14)
S("BulletX", fontSize=10, leading=14, alignment=TA_LEFT, fontName="Helvetica")
S("CodeX", fontSize=8.3, leading=11.5, fontName="Courier", textColor=colors.HexColor("#1A1A1A"))

story = []

def P(txt, s="Body"): story.append(Paragraph(txt, styles[s]))
def sp(h=6): story.append(Spacer(1, h))

def img(path, width_cm, caption=None):
    p = Path(path)
    if not p.exists():
        P(f"[missing image: {p.name}]"); return
    from PIL import Image as PILImage
    iw, ih = PILImage.open(p).size
    w = width_cm * cm; h = w * ih / iw
    story.append(Image(str(p), width=w, height=h))
    if caption: P(caption, "Caption")

def rule():
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CCD6E8"), spaceBefore=4, spaceAfter=8))

def bullets(items):
    lf = ListFlowable(
        [ListItem(Paragraph(t, styles["BulletX"]), value="•", leftIndent=10) for t in items],
        bulletType="bullet", start="•", leftIndent=14, bulletFontSize=9, spaceAfter=8,
    )
    story.append(lf); sp(4)

def table(data, colw, header=True, highlight_last=False, fs=8.7, spanheader=None):
    t = Table(data, colWidths=[c * cm for c in colw], hAlign="CENTER")
    ts = [
        ("FONTSIZE", (0, 0), (-1, -1), fs),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        ts += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT])]
    if highlight_last:
        ts += [("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#D8E6C8")),
               ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")]
    t.setStyle(TableStyle(ts)); story.append(t)

def code_block(lines):
    txt = "<br/>".join(l.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;") for l in lines)
    tbl = Table([[Paragraph(txt, styles["CodeX"])]], colWidths=[16 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODEBG),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D0D0D0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(tbl); sp(4)


# ════════════════════════════════════════ TITLE ════════════════════════════════════════
sp(70)
P("Real-Time Bangladeshi Taka<br/>Banknote Detection", "TitleBig")
sp(10)
P("A YOLOv8 Object-Detection System Trained on<br/>"
  "Synthetically Composited Data with Real-World Backgrounds", "Subtitle")
sp(28)
story.append(HRFlowable(width="55%", thickness=1.4, color=BLUE, hAlign="CENTER"))
sp(22)
P("Detailed Technical Project Report", "Meta")
sp(6)
P("Deep Learning &middot; Computer Vision &middot; Object Detection", "Meta")
sp(36)
img(ASSETS / "architecture.png", 15)
sp(26)
P("Prepared: July 2026", "Meta")
story.append(PageBreak())

# ════════════════════════════════════════ ABSTRACT ════════════════════════════════════════
P("Abstract", "H1"); rule()
P("This project develops a real-time detection system that localises and classifies the nine "
  "denominations of Bangladeshi Taka (BDT) paper currency &mdash; 2, 5, 10, 20, 50, 100, 200, "
  "500 and 1000 Taka &mdash; from a live camera feed. A recurring problem with publicly "
  "available BDT datasets is that they consist exclusively of clean, close-up, full-frame note "
  "scans, which teach a model that &ldquo;the entire frame is the note&rdquo; and fail on a "
  "webcam, where a note appears at an arbitrary scale, angle and position against cluttered "
  "real backgrounds. To bridge this synthetic-to-real domain gap, we designed a copy-paste "
  "compositing pipeline that pastes 5,073 background-free note images onto 1,500 real-world "
  "photographs from the COCO dataset, applying perspective warp, rotation, scale variation, "
  "colour/brightness jitter, motion blur, sensor noise and JPEG re-compression, and "
  "automatically generating exact bounding-box labels. The resulting 22,315-image dataset was "
  "used to fine-tune a COCO-pretrained YOLOv8s detector for 80 epochs. On a held-out test set "
  "of 2,238 images the model achieves mAP@0.5 of 0.995, mAP@0.5:0.95 of 0.847, precision of "
  "0.998 and recall of 0.997, at 7.3 ms per image (&asymp;135 FPS) on an NVIDIA RTX 4060 Laptop "
  "GPU. The system runs live from a standard webcam and includes automatic low-light brightness "
  "correction.", "AbstractBody")
sp(10)
P("<b>Keywords:</b> Object Detection, YOLOv8, Bangladeshi Currency, Synthetic Data, Domain "
  "Randomization, Copy-Paste Augmentation, Transfer Learning, Real-Time Inference.", "Body")

# ════════════════════════════════════════ 1. INTRO ════════════════════════════════════════
P("1. Introduction", "H1"); rule()
P("Automatic recognition of paper currency has practical value in assistive technology for the "
  "visually impaired, automated cash-handling machines, retail self-service and financial "
  "automation. For Bangladeshi Taka in particular, an ideal system must not only <i>classify</i> "
  "which note is present but also <i>detect</i> and localise it within a cluttered scene in real "
  "time, so that it can operate from a phone or webcam held by a user in an uncontrolled "
  "environment.", "Body")
P("Most existing BDT recognition work frames the task as whole-image <i>classification</i> on "
  "clean, centred note images. Such models perform well on their own curated test sets but "
  "degrade sharply in the field because real camera frames contain background clutter, "
  "perspective distortion, motion blur, variable lighting and notes that occupy only a fraction "
  "of the frame. This project deliberately targets that gap by treating the problem as "
  "<b>object detection</b> and by engineering the training data so that its statistics resemble "
  "real webcam imagery rather than studio scans.", "Body")
P("<b>Contributions.</b>", "Body")
bullets([
    "A reproducible synthetic data-generation pipeline that converts a background-free "
    "classification dataset into a large, richly augmented detection dataset with automatically "
    "generated bounding-box labels.",
    "A fine-tuned YOLOv8s detector that attains 0.995 mAP@0.5 on a held-out test set while "
    "running above 100 FPS on a laptop GPU.",
    "A deployable real-time webcam application with automatic low-light correction and a "
    "TorchScript export for dependency-free inference.",
])

# ════════════════════════════════════════ 2. RELATED WORK ════════════════════════════════════════
P("2. Related Work", "H1"); rule()
P("Early Bangladeshi banknote recognition relied on classical neural networks with "
  "hand-engineered features; a neural network using axis-symmetrical masks reported an average "
  "accuracy of 98.57% across eight denominations [4]. More recent approaches use deep "
  "convolutional networks and transfer learning: lightweight CNNs and pretrained backbones "
  "(MobileNet, NASNetMobile, ResNet152v2, EfficientNet) have been applied to real-time "
  "classification for visually impaired users [5,7], and dual-stream MobileNet/EfficientNet "
  "architectures have been proposed for robust recognition [7].", "Body")
P("For the <i>detection</i> formulation, single-stage detectors of the YOLO family are the "
  "dominant choice because of their speed. A YOLOv5-based taka detector trained on 3,111 "
  "annotated images reported precision and recall of 0.95 and 0.93 [1], and a YOLOv8-nano model "
  "with a custom Squeeze-and-Excitation head for multi-currency (USD/EUR/BDT) detection reported "
  "mAP@0.5 of 0.972 and mAP@0.5:0.95 of 0.804 at 3.1 ms per image [2]. Our work differs in its "
  "explicit focus on closing the synthetic-to-real domain gap through domain-randomised "
  "copy-paste compositing, an approach shown to be highly effective for synthetic-to-real "
  "object detection [3]. Whereas [1] and [2] annotate real photographs by hand, we generate an "
  "order of magnitude more labelled data automatically, with pixel-perfect boxes and full "
  "control over the distribution of scales, angles and backgrounds.", "Body")

story.append(PageBreak())

# ════════════════════════════════════════ 3. DATASET ════════════════════════════════════════
P("3. Dataset and Data Engineering", "H1"); rule()
P("This section describes the two raw data sources and, in detail, how they were combined into "
  "a purpose-built detection dataset. Data engineering is the core contribution of the project, "
  "so it is documented step by step.", "Body")

P("3.1 Source Currency Dataset (Foreground)", "H2")
P("The banknote imagery is drawn from <b>&ldquo;A Diverse Image Dataset for Bangladeshi "
  "Currency Recognition&rdquo;</b> (also published as the <i>BanglaTaka</i> dataset), openly "
  "available on Mendeley Data [6]. It contains <b>5,073 background-free images</b> of BDT paper "
  "currency spanning all nine circulating denominations. Images were captured with a variety of "
  "mobile-phone cameras and pre-processed with the CamScanner application, giving clean, cropped "
  "notes on a plain field. Because each image is already a tight crop of a single note, it is "
  "ideal <i>foreground</i> material: the whole image can be treated as the object to be pasted.",
  "Body")
table([
    ["Denomination", "2", "5", "10", "20", "50", "100", "200", "500", "1000", "Total"],
    ["Source images", "445", "660", "419", "480", "416", "565", "423", "1243", "422", "5073"],
], [2.9, 1.15, 1.15, 1.15, 1.15, 1.15, 1.15, 1.15, 1.15, 1.2, 1.3], fs=8.2)
P("Table 1 &mdash; Per-denomination image counts in the source BanglaTaka dataset.", "Caption")

P("3.2 Background Dataset", "H2")
P("Real-world backgrounds were sourced from the <b>COCO val2017</b> split [8] &mdash; a widely "
  "used collection of natural photographs containing desks, hands, indoor rooms, streets and "
  "outdoor scenes. We downloaded <b>1,500</b> images programmatically. Using genuine "
  "photographic backgrounds (rather than solid colours or synthetic noise, as in an earlier "
  "iteration of this project) is the single most important factor in making the model "
  "generalise to real camera frames, because it forces the detector to separate the note from "
  "realistic clutter, texture and lighting.", "Body")

P("3.3 The Domain-Gap Problem", "H2")
P("The reason the two datasets must be combined is a <b>domain gap</b>. A model trained only on "
  "the background-free BanglaTaka images learns that a note fills the entire frame on a plain "
  "background. When such a model is shown a webcam frame &mdash; where the note is small, "
  "rotated, and surrounded by objects &mdash; its assumptions break and detection collapses. "
  "This was observed directly in an early version of the system, which scored highly on "
  "synthetic tests yet failed to detect notes on a live webcam. Compositing the notes onto real "
  "backgrounds re-creates deployment conditions at training time and eliminates the gap.", "Body")

P("3.4 Copy-Paste Compositing Process", "H2")
P("Each source note is pasted onto a randomly chosen, randomly cropped COCO background after a "
  "chain of randomised transformations. Crucially, because we control exactly where and at what "
  "size the note is placed, the corresponding <b>bounding box is computed analytically</b> "
  "&mdash; no manual annotation is required, and the labels are pixel-perfect. Figure 2 shows "
  "the process applied to real assets from our pipeline.", "Body")
img(ASSETS / "compositing.png", 16.5,
    "Figure 2 &mdash; The compositing process on real data: (a) a background-free 100-Taka note "
    "and (b) a COCO background are combined; (c) the note is warped, rotated, scaled and "
    "colour-jittered; (d) it is pasted at a random location and an exact bounding box (green) "
    "with the class label is generated automatically.")

story.append(PageBreak())

P("Each panel of Figure 2 corresponds to a concrete stage of the algorithm:", "Body")
bullets([
    "<b>(a) Source note</b> &mdash; the raw foreground, loaded with an alpha channel so that "
    "only the note (not its rectangular border) is pasted.",
    "<b>(b) Real background</b> &mdash; a random COCO photo, centre-cropped to a square and "
    "resized to 640&times;640 to form the canvas.",
    "<b>(c) Augmented note</b> &mdash; the note after perspective warp, rotation, scaling and "
    "photometric jitter, simulating how a real note looks when held at an angle under different "
    "lighting.",
    "<b>(d) Final composite</b> &mdash; the augmented note pasted at a random position, followed "
    "by frame-level effects (blur, noise, JPEG artefacts). The green box is the auto-generated "
    "ground-truth label written in YOLO format.",
])

P("3.5 Bounding-Box Computation", "H2")
P("YOLO labels store each box as four normalised numbers &mdash; centre-x, centre-y, width and "
  "height, each in the range [0,1] relative to the image dimensions. Given a note of pixel size "
  "(w,&nbsp;h) pasted at top-left corner (x,&nbsp;y) on a canvas of size S=640, the label is "
  "computed directly:", "Body")
code_block([
    "x_center = (x + w/2) / S        # normalised horizontal centre",
    "y_center = (y + h/2) / S        # normalised vertical centre",
    "box_w    =  w / S               # normalised width",
    "box_h    =  h / S               # normalised height",
    "",
    "label line  ->  \"<class_id> <x_center> <y_center> <box_w> <box_h>\"",
])
P("This closed-form label is the key advantage of synthetic compositing: annotation is free, "
  "exact and instantaneous, allowing tens of thousands of labelled examples to be produced "
  "automatically.", "Body")

P("3.6 Augmentation Operations", "H2")
P("The following domain-randomisation operations are applied per composite. Their ranges were "
  "chosen to span &mdash; and slightly exceed &mdash; the variation expected from a hand-held "
  "webcam, so that real conditions fall inside the training distribution.", "Body")
table([
    ["Stage", "Operation", "Range / Setting", "Purpose"],
    ["Geometry", "Perspective warp", "70% prob., corner jitter 8%", "notes held at an angle"],
    ["", "Rotation", "±30°", "arbitrary orientation"],
    ["", "Scale (frac. of frame)", "0.25 – 0.90", "near/far distance"],
    ["", "Random placement", "anywhere", "off-centre framing"],
    ["Photometric", "HSV jitter", "H±18° S×0.5–1.5 V×0.4–1.6", "colour/lighting shifts"],
    ["", "Brightness / Contrast", "0.4–1.6 / 0.7–1.3", "exposure variation"],
    ["Sensor", "Motion blur", "20% prob., kernel 3/5/7", "hand shake"],
    ["", "Gaussian noise", "50% prob., σ≈12", "low-light sensor noise"],
    ["", "JPEG re-compression", "quality 55–92", "webcam compression"],
    ["Balance", "Negative images", "10% of output", "suppress false positives"],
], [2.0, 3.3, 4.5, 4.2], fs=8.2)
P("Table 2 &mdash; Domain-randomisation operations and the real-world variation each targets.", "Caption")

P("3.7 Final Dataset Composition", "H2")
P("Four composites were produced per source note, plus a 10% share of background-only "
  "<i>negative</i> images (empty labels) that teach the model when <i>not</i> to fire. The data "
  "was split 80/10/10 <b>by source note</b> so that no note appears in more than one split, "
  "preventing information leakage between train, validation and test.", "Body")
table([
    ["Split", "Positive Images", "Negative Images", "Total Images", "Note Instances"],
    ["Train", "16,224", "1,620", "17,844", "16,224"],
    ["Validation", "2,016", "217", "2,233", "2,016"],
    ["Test", "2,052", "186", "2,238", "2,052"],
    ["Total", "20,292", "2,023", "22,315", "20,292"],
], [3.0, 3.2, 3.2, 2.9, 3.0], highlight_last=True)
P("Table 3 &mdash; Detection dataset composition after synthetic compositing.", "Caption")

story.append(PageBreak())

# ════════════════════════════════════════ 4. METHODOLOGY ════════════════════════════════════════
P("4. Methodology and Implementation", "H1"); rule()
P("The overall workflow is summarised in Figure 3. Three inputs &mdash; source notes, real "
  "backgrounds and COCO-pretrained YOLOv8s weights &mdash; feed the compositing engine, which "
  "produces a labelled detection dataset used for training, evaluation and deployment.", "Body")
img(ASSETS / "methodology.png", 16, "Figure 3 &mdash; End-to-end methodology pipeline.")

P("4.1 Software Stack and Implementation", "H2")
P("The system was implemented in Python. The compositing engine "
  "(<font face='Courier'>generate_synthetic_dataset.py</font>) uses Pillow and OpenCV for image "
  "transforms and NumPy for array maths. Training, validation and export use the "
  "<b>Ultralytics YOLOv8</b> framework (v8.4.64) on <b>PyTorch 2.6.0</b> with CUDA 12.4. The "
  "real-time application (<font face='Courier'>realtime_detect.py</font>) uses OpenCV for camera "
  "capture and display. All experiments ran on an NVIDIA RTX 4060 Laptop GPU (8&nbsp;GB) with an "
  "AMD Ryzen 7 CPU under Windows 11. The project is organised as a set of reproducible scripts:",
  "Body")
table([
    ["Script", "Role"],
    ["download_backgrounds.py", "Fetch 1,500 COCO val2017 background images"],
    ["generate_synthetic_dataset.py", "Composite notes + backgrounds → labelled YOLO dataset"],
    ["train.py", "Fine-tune YOLOv8s with the augmentation recipe"],
    ["evaluate.py / random_test_check.py", "Test-set metrics, confusion matrix, spot checks"],
    ["realtime_detect.py", "Live webcam detection with low-light correction"],
], [5.8, 9.4], fs=8.7)
P("Table 4 &mdash; Project scripts and their roles.", "Caption")

P("4.2 Detection Algorithm: YOLOv8", "H2")
P("We use <b>YOLO (You Only Look Once)</b>, a <i>single-stage</i> object detector. Unlike "
  "two-stage detectors (e.g. Faster R-CNN) that first propose regions and then classify them, "
  "YOLO performs localisation and classification in a single forward pass over the image, which "
  "is what makes it fast enough for real-time video. The image is processed by a fully "
  "convolutional network that outputs, at three spatial resolutions, a dense grid of "
  "predictions; each prediction carries a class distribution and four box-regression values.",
  "Body")
P("YOLOv8 is <b>anchor-free</b>: rather than matching objects to a fixed set of pre-defined "
  "anchor-box shapes, it predicts box edges directly from each grid cell, which simplifies "
  "training and improves accuracy on unusual aspect ratios (relevant here, since banknotes are "
  "long and thin). Predictions are filtered by a confidence threshold and de-duplicated with "
  "<b>Non-Maximum Suppression (NMS)</b>, which keeps the highest-scoring box among heavily "
  "overlapping detections. We fine-tune the <b>&ldquo;s&rdquo; (small)</b> variant, which "
  "offers the best accuracy/speed trade-off for our hardware.", "Body")

P("4.3 Model Architecture", "H2")
P("YOLOv8s has three parts &mdash; a backbone, a neck and a head (Figure 4). Each is described "
  "below in plain terms.", "Body")
img(ASSETS / "architecture.png", 15.5, "Figure 4 &mdash; YOLOv8s detection architecture.")

P("Input", "H3")
P("The camera frame is resized to 640&times;640 pixels with three colour channels (RGB) and "
  "pixel values scaled to [0,1] before entering the network.", "Body")
P("Backbone &mdash; CSPDarknet", "H3")
P("The backbone is a convolutional feature extractor that progressively shrinks the image while "
  "deepening it, converting raw pixels into rich feature maps. It is built from <b>C2f</b> "
  "blocks (cross-stage partial modules that split, process and re-merge features for efficient "
  "gradient flow) and ends in an <b>SPPF</b> (Spatial Pyramid Pooling &ndash; Fast) module that "
  "pools over several receptive-field sizes so the network sees both fine detail and broad "
  "context. This is where the model learns what a banknote's texture, portrait and numerals look "
  "like.", "Body")
P("Neck &mdash; PAN-FPN", "H3")
P("The neck fuses features from different depths of the backbone. A <b>Feature Pyramid Network "
  "(FPN)</b> carries high-level semantic information from deep layers down to shallow "
  "high-resolution layers, and a <b>Path Aggregation Network (PAN)</b> adds a bottom-up path "
  "back again. Together they let the detector recognise notes at very different sizes &mdash; a "
  "note filling the frame and a small note far from the camera are both handled well.", "Body")
P("Head &mdash; Decoupled, Anchor-Free", "H3")
P("The head makes the final predictions at three scales. It is <b>decoupled</b> &mdash; separate "
  "branches predict the class and the box &mdash; which improves accuracy, and <b>anchor-free</b> "
  "as described above. For our task it outputs, for every candidate location, a probability over "
  "the nine denomination classes plus four numbers defining the box.", "Body")

story.append(PageBreak())

P("4.4 Loss Function", "H2")
P("Training minimises a weighted sum of three losses that jointly teach the model <i>where</i> "
  "and <i>what</i> each note is:", "Body")
bullets([
    "<b>Box regression (CIoU) loss</b> &mdash; penalises mismatch between predicted and true "
    "boxes using Complete-IoU, which accounts for overlap, centre distance and aspect ratio.",
    "<b>Distribution Focal Loss (DFL)</b> &mdash; refines box edges by predicting a distribution "
    "over positions rather than a single point, giving sub-pixel localisation.",
    "<b>Classification (BCE) loss</b> &mdash; binary cross-entropy over the nine classes, "
    "teaching the model which denomination occupies each detected box.",
])
P("The default Ultralytics weightings were used (box 7.5, class 0.5, DFL 1.5).", "Body")

P("4.5 Transfer Learning", "H2")
P("Rather than training from scratch, we initialise from <b>COCO-pretrained weights</b>. The "
  "backbone has already learned general visual primitives (edges, textures, shapes) from "
  "millions of images, so fine-tuning only has to adapt these features to banknotes. This "
  "dramatically speeds convergence &mdash; the model exceeded 0.97 mAP@0.5 within eight epochs "
  "&mdash; and improves final accuracy compared with random initialisation, especially given a "
  "single-class-family task like ours.", "Body")

P("4.6 Training Procedure", "H2")
P("Training ran for 80 epochs at 640&times;640 with a batch size of 16 and the SGD optimiser "
  "(auto-configured by Ultralytics). On top of the offline compositing, YOLOv8's <i>online</i> "
  "augmentation was enabled &mdash; <b>mosaic</b> (stitching four images into one to vary "
  "context and scale) and <b>mixup</b> (blending two images) &mdash; along with HSV and "
  "geometric jitter. Horizontal and vertical flips were <b>disabled</b> because a note's "
  "orientation carries information and mirroring could corrupt printed text. Early-stopping "
  "patience was set to 15 epochs. The full configuration is given in Table 5.", "Body")
table([
    ["Hyperparameter", "Value", "Hyperparameter", "Value"],
    ["Base model", "yolov8s.pt", "Optimizer", "SGD (auto)"],
    ["Epochs", "80", "Initial LR (lr0)", "0.01"],
    ["Image size", "640×640", "Final LR (lrf)", "0.01"],
    ["Batch size", "16", "Momentum", "0.937"],
    ["Early-stop patience", "15", "Weight decay", "0.0005"],
    ["Mosaic / Mixup", "1.0 / 0.15", "Rotation (deg)", "20"],
    ["HSV (h/s/v)", "0.02/0.7/0.4", "Flips (lr/ud)", "0.0 / 0.0"],
    ["Warmup epochs", "3.0", "Box/Cls/DFL gain", "7.5 / 0.5 / 1.5"],
], [3.8, 2.9, 3.6, 2.5])
P("Table 5 &mdash; Key training hyperparameters.", "Caption")

# ════════════════════════════════════════ 5. RESULTS ════════════════════════════════════════
P("5. Results and Analysis", "H1"); rule()
P("5.1 Evaluation Metrics", "H2")
P("Detectors are judged by <b>precision</b> (fraction of detections that are correct), "
  "<b>recall</b> (fraction of true notes found), and <b>mean Average Precision (mAP)</b>. "
  "mAP@0.5 counts a detection as correct if it overlaps the true box by at least 50% "
  "(Intersection-over-Union &ge; 0.5); the stricter <b>mAP@0.5:0.95</b> averages over IoU "
  "thresholds from 0.5 to 0.95 and is far more sensitive to precise localisation &mdash; it is "
  "the headline number for real-world quality.", "Body")

P("5.2 Overall Performance", "H2")
P("Training converged rapidly: mAP@0.5 exceeded 0.97 by epoch 8 and 0.99 by epoch 11, after "
  "which mAP@0.5:0.95 continued to climb as localisation sharpened (Figure 5). Final metrics on "
  "the held-out test set (2,238 images) are given in Table 6.", "Body")
table([
    ["Metric", "Value"],
    ["Precision", "0.998"],
    ["Recall", "0.997"],
    ["mAP@0.5", "0.995"],
    ["mAP@0.5:0.95", "0.847"],
    ["Inference speed", "7.3 ms / image (≈135 FPS)"],
    ["Preprocess / Postprocess", "1.7 ms / 1.7 ms"],
], [5.5, 6.0])
P("Table 6 &mdash; Test-set performance (best checkpoint).", "Caption")
img(RESULTS / "epoch_results.png", 16,
    "Figure 5 &mdash; Training curves: losses (top) and mAP / precision-recall (bottom) vs. epoch.")

story.append(PageBreak())

P("5.3 Per-Class Results", "H2")
P("Performance is uniformly strong across all nine denominations, confirming the model does not "
  "favour high-frequency classes such as 500 Taka (which has the most source images).", "Body")
table([
    ["Class", "Images", "Instances", "Precision", "Recall", "mAP@0.5", "mAP@.5:.95"],
    ["2 Taka",   "180", "180", "0.998", "1.000", "0.995", "0.838"],
    ["5 Taka",   "264", "264", "0.998", "0.996", "0.995", "0.844"],
    ["10 Taka",  "172", "172", "0.995", "0.988", "0.995", "0.848"],
    ["20 Taka",  "192", "192", "0.998", "1.000", "0.995", "0.847"],
    ["50 Taka",  "172", "172", "1.000", "0.993", "0.995", "0.851"],
    ["100 Taka", "228", "228", "0.999", "1.000", "0.995", "0.845"],
    ["200 Taka", "172", "172", "0.999", "1.000", "0.995", "0.843"],
    ["500 Taka", "500", "500", "1.000", "1.000", "0.995", "0.853"],
    ["1000 Taka","172", "172", "0.994", "1.000", "0.995", "0.858"],
    ["All",     "2238","2052", "0.998", "0.997", "0.995", "0.847"],
], [2.4, 1.7, 2.0, 2.0, 1.7, 1.9, 2.3], highlight_last=True)
P("Table 7 &mdash; Per-class detection metrics on the test set.", "Caption")

P("5.4 Confusion Matrix", "H2")
P("The normalised confusion matrix (Figure 6) is almost perfectly diagonal: every denomination "
  "is classified correctly with probability 0.99&ndash;1.00. The only residual error is minor "
  "confusion with the <i>background</i> column &mdash; occasional missed or spurious detections "
  "rather than denomination mix-ups. For a currency reader this is the safest failure mode, "
  "because the system never confuses one note value for another.", "Body")
img(RESULTS / "confusion_matrix_normalized_test.png", 13,
    "Figure 6 &mdash; Normalised confusion matrix on the test set.")

story.append(PageBreak())

P("5.5 Qualitative Detections", "H2")
P("Figure 7 shows a validation batch with predicted boxes, labels and confidence scores. Notes "
  "are tightly localised across a range of scales, rotations and real backgrounds &mdash; "
  "exactly the conditions the compositing pipeline was designed to reproduce.", "Body")
img(RUNS / "val_batch0_pred.jpg", 15.5,
    "Figure 7 &mdash; Example predictions on a validation batch (label + confidence score).")

# ════════════════════════════════════════ 6. COMPARISON ════════════════════════════════════════
P("6. Comparison with Similar Works", "H1"); rule()
P("Table 8 situates our results among comparable Bangladeshi / multi-currency detection and "
  "recognition systems. Direct numerical comparison should be read with care because each study "
  "uses a different dataset, class set and test protocol; the table shows that our system is "
  "competitive with, or exceeds, published detection baselines while explicitly targeting "
  "real-world robustness.", "Body")
table([
    ["Work", "Approach", "Task", "Classes", "mAP@0.5", "mAP@.5:.95", "P / R"],
    ["NN + sym.\nmasks [4]", "Shallow NN", "Classify", "8", "—", "—", "98.6% acc"],
    ["CNN real-time\n[5]", "Lightweight CNN", "Classify", "9", "—", "—", "high acc"],
    ["Dual-stream\n[7]", "MobileNet+\nEfficientNet", "Classify", "9", "—", "—", "robust"],
    ["YOLOv5 taka\n[1]", "YOLOv5", "Detect", "—", "—", "—", "0.95 / 0.93"],
    ["Custom-head\nYOLOv8n [2]", "YOLOv8n+SE", "Detect", "30", "0.972", "0.804", "0.965 / 0.952"],
    ["This work", "YOLOv8s +\nsynth. data", "Detect", "9 (BDT)", "0.995", "0.847", "0.998 / 0.997"],
], [2.2, 2.5, 1.6, 1.5, 1.7, 1.9, 2.2], fs=8.0, highlight_last=True)
P("Table 8 &mdash; Comparison with related currency-recognition systems. "
  "&ldquo;—&rdquo; denotes a metric not reported by that work; &ldquo;acc&rdquo; = classification accuracy.", "Caption")
P("Several observations follow. First, our <b>mAP@0.5:0.95 of 0.847</b> &mdash; the metric most "
  "sensitive to localisation quality and real-world variation &mdash; is higher than the 0.804 "
  "reported by the custom-head YOLOv8n multi-currency detector [2], despite our use of an "
  "unmodified backbone; the gain is attributable to the domain-randomised training data rather "
  "than architectural tricks. Second, our precision and recall (0.998 / 0.997) exceed the "
  "YOLOv5 baseline's 0.95 / 0.93 [1]. Third, unlike the classification-only systems [4,5,7], our "
  "model outputs bounding boxes and therefore supports localisation and multi-note counting, not "
  "just single-note recognition. Finally, our labelled dataset (22,315 images) is far larger "
  "than the hand-annotated 3,111-image set in [1], obtained at zero annotation cost through "
  "automatic compositing.", "Body")

# ════════════════════════════════════════ 7. DEPLOYMENT ════════════════════════════════════════
P("7. Real-Time Deployment", "H1"); rule()
P("The trained model is deployed in a webcam application built on OpenCV. Each frame is passed "
  "through the detector and annotated with live bounding boxes at interactive frame rates "
  "(&asymp;135 FPS inference, well above real-time). A practical issue observed during testing "
  "was that low ambient light produced near-black frames (mean intensity &asymp; 20/255), which "
  "suppressed detections. The application therefore includes an <b>automatic brightness "
  "correction</b> step: when a frame's mean intensity falls below a threshold, a gain-and-gamma "
  "correction is applied before inference, restoring reliable detection in dim conditions. The "
  "model was additionally exported to <b>TorchScript</b> (.pts) so it can run without the "
  "Ultralytics Python dependency in production environments.", "Body")

# ════════════════════════════════════════ 8. LIMITATIONS ════════════════════════════════════════
P("8. Limitations and Future Work", "H1"); rule()
bullets([
    "<b>Synthetic realism.</b> Although real backgrounds and heavy augmentation greatly reduce "
    "the domain gap, composited notes still lack real cast shadows, physical folds and true "
    "occlusion. A modest set of genuine in-the-wild photographs for fine-tuning or validation "
    "would further strengthen real-world guarantees.",
    "<b>Condition and authenticity.</b> The system verifies denomination, not authenticity, and "
    "was trained on notes in good condition. Worn, torn or folded notes and counterfeit "
    "detection (cf. the JaalTaka benchmark) are natural extensions.",
    "<b>Edge deployment.</b> Future work includes INT8 quantisation and export to ONNX / "
    "TensorRT / TFLite for smartphone and embedded deployment, plus an audio-feedback mode to "
    "serve visually impaired users directly.",
])

# ════════════════════════════════════════ 9. CONCLUSION ════════════════════════════════════════
P("9. Conclusion", "H1"); rule()
P("We presented a complete, reproducible pipeline for real-time Bangladeshi Taka detection that "
  "converts a clean classification dataset into a large, domain-randomised detection dataset and "
  "fine-tunes a YOLOv8s detector on it. The resulting model achieves 0.995 mAP@0.5 and 0.847 "
  "mAP@0.5:0.95 on a held-out test set with near-perfect precision and recall, runs above 100 "
  "FPS on a laptop GPU, and operates live from a webcam with automatic low-light correction. The "
  "central lesson is that <i>engineering the training data to match deployment conditions</i> "
  "&mdash; real backgrounds plus aggressive, physically motivated augmentation &mdash; is what "
  "turns a brittle studio-only classifier into a robust real-world detector.", "Body")

# ════════════════════════════════════════ REFERENCES ════════════════════════════════════════
P("References", "H1"); rule()
refs = [
    "[1] YOLOv5-based Bangladeshi Taka detection subset (3,111 annotated images), reporting "
    "precision 0.95 and recall 0.93.",
    "[2] Real-Time Currency Detection and Voice Feedback for Visually Impaired Individuals, "
    "arXiv:2510.20267 &mdash; custom-head YOLOv8n (USD/EUR/BDT), mAP@0.5 0.972, mAP@0.5:0.95 "
    "0.804, 3.1 ms/image.",
    "[3] Nino, Gardi et al., &ldquo;Synthetic-to-Real Object Detection using YOLO and Domain "
    "Randomization Strategies,&rdquo; arXiv:2509.15045.",
    "[4] Bangladeshi Banknote Recognition by Neural Network with Axis-Symmetrical Masks &mdash; "
    "average accuracy 98.57% over eight denominations.",
    "[5] Bangladeshi Banknote Recognition in Real-time using Convolutional Neural Network for "
    "Visually Impaired People.",
    "[6] &ldquo;A Diverse Image Dataset for Bangladeshi Currency Recognition&rdquo; (BanglaTaka), "
    "Mendeley Data, 5,073 images, 9 denominations. https://data.mendeley.com/datasets/3cv2sypkkh/1",
    "[7] Robust and Real-Time Bangladeshi Currency Recognition: A Dual-Stream MobileNet and "
    "EfficientNet Approach, arXiv:2602.07015.",
    "[8] Lin et al., &ldquo;Microsoft COCO: Common Objects in Context,&rdquo; ECCV 2014 "
    "(val2017 split used for backgrounds).",
    "[9] Jocher et al., Ultralytics YOLOv8 (v8.4.64). https://github.com/ultralytics/ultralytics",
]
for r in refs: P(r, "Ref")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CCD6E8")); canvas.setLineWidth(0.6)
    canvas.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
    canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.05 * cm, "Real-Time Bangladeshi Taka Banknote Detection")
    canvas.drawRightString(A4[0] - 2 * cm, 1.05 * cm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
    topMargin=1.8 * cm, bottomMargin=1.9 * cm,
    title="Real-Time Bangladeshi Taka Banknote Detection",
    author="Currency Detection Project",
)
doc.build(story, onLaterPages=footer)
print("PDF written to", OUT)
