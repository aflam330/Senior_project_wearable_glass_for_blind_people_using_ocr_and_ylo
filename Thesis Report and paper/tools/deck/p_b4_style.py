"""build4.js from build3.js: the look and section names of the reference deck (thesis report/slides/demo slide.pptx):
light silver-blue background, formal academic headings, IUB title slide with Supervisor / Submitted By; new title and
subtitle; a Hardware Design slide; citation numbers assigned in order of first use; sources added where needed."""
from pathlib import Path

p = Path(__file__).parent / "build4.js"
s = p.read_text(encoding="utf-8")


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:90], s.count(a))
    s = s.replace(a, b)


rep("""// SARA presentation, third design (26 October 2026): fewer words, larger type, illustrated problem and solution,
// one takeaway per results slide. Built from build2.js; the illustrations are the two pictures in the slides folder.
// node build3.js   -> ../../thesis report/slides/SARA_Presentation.pptx   (then add_transitions.py adds fades)""",
    """// SARA presentation, fourth design (29 October 2026): the look and section names of the reference deck
// (thesis report/slides/demo slide.pptx). Built from build3.js; p_b4_style.py made the changes.
// node build4.js   -> ../../thesis report/slides/SARA_Presentation.pptx   (then add_transitions.py adds fades)""")

# ---------------------------------------------------------------- theme: navy / steel blue on the silver-blue background
rep('headFontFace: "Cambria",', 'headFontFace: "Arial",')
rep('''    dk1: "1C2321", lt1: "FFFFFF", dk2: "0B4D3B", lt2: "EEF5F1",
    accent1: "006A4E", accent2: "D7263D", accent3: "3A8F72", accent4: "E6A817", accent5: "5E6B66", accent6: "B9DCCD",
    hlink: "006A4E", folHlink: "3A8F72",''',
    '''    dk1: "1B2430", lt1: "FFFFFF", dk2: "17324D", lt2: "EAF1F7",
    accent1: "1F5F8B", accent2: "D7263D", accent3: "4A87B5", accent4: "E6A817", accent5: "5B6777", accent6: "BFD4E6",
    hlink: "1F5F8B", folHlink: "4A87B5",''')
s = s.replace('fontFace: "Cambria"', 'fontFace: "Arial"')
for a, b in [('"E4EFEA"', '"E6EEF6"'), ('"CFE2DA"', '"D2E0EC"'), ('"B9C4BF"', '"B8C6D4"'), ('"9FC2B4"', '"9FB4C8"'),
             ('"9DB8AD"', '"9DB3C8"'), ('"DDE7E2"', '"DCE5EE"')]:
    s = s.replace(a, b)

rep('const TITLE = "A Wearable Assistive Robot for Authenticating Bangladeshi Banknotes: Fusing Physical Security Features and Deep Learning for Visually Impaired Users";',
    '''const TITLE = "A Wearable Assistive Glass for Authenticating Bangladeshi Banknotes: Fusing Physical Security Features and Deep Learning for Visually Impaired Users";
const SUBTITLE = "A bilingual (Bangla and English) smart glass that also reads printed Bangla and English text and detects everyday objects, offline on a Raspberry Pi 5";
const BG = path.resolve(__dirname, "assets", "bg_slide.jpg");   // background of the reference deck''')

# ---------------------------------------------------------------- references: numbers in order of first use
rep('''  ["who", "World Health Organization. Blindness and vision impairment: fact sheet, 2023.",''',
    '''  ["who", "World Health Organization. Blindness and vision impairment: fact sheet, updated 10 February 2026.",''')
rep('''  ["mutanen", "Mutanen, J., Jaaskelainen, T., Parkkinen, J. P. S. Luminescent security properties of banknotes. IS&T PICS Conference, 2003, pp. 421-424.", "https://www.imaging.org/common/uploaded%20files/pdfs/Papers/2003/PICS-0-287/8558.pdf"],
];''', '''  ["mutanen", "Mutanen, J., Jaaskelainen, T., Parkkinen, J. P. S. Luminescent security properties of banknotes. IS&T PICS Conference, 2003, pp. 421-424.", "https://www.imaging.org/common/uploaded%20files/pdfs/Papers/2003/PICS-0-287/8558.pdf"],
  ["rpi5", "Raspberry Pi Ltd. Raspberry Pi 5 product brief, 2023.", "https://www.raspberrypi.com/products/raspberry-pi-5/"],
];''')
rep('''const ORDER = ["who", "orcam", "glass", "airis", "craft", "easyocr", "yolo", "coco", "rafdb", "piper", "espeak", "banglataka", "nstu", "jaal", "jaal26", "uddin", "pathak", "mnv2", "uvink", "mutanen", "mlt", "vosk", "esc50"];
REFS.sort((a, b) => ORDER.indexOf(a[0]) - ORDER.indexOf(b[0]));
const NUM = Object.fromEntries(REFS.map((r, i) => [r[0], i + 1]));
const cite = (...keys) => "[" + keys.map((k) => NUM[k]).sort((a, b) => a - b).join(", ") + "]";''',
    '''// numbers are given in the order the slides first cite a source (the slides are built in order)
const NUM = {};
const URL = Object.fromEntries(REFS.map((r) => [r[0], r[2]]));
const cite = (...keys) => "[" + keys.map((k) => {
  if (!(k in URL)) throw new Error("unknown reference " + k);
  if (!NUM[k]) NUM[k] = Object.keys(NUM).length + 1;
  return NUM[k];
}).sort((a, b) => a - b).join(", ") + "]";''')

# ---------------------------------------------------------------- masters: background picture, centred headings
rep('''    title: "TITLE_DARK",
    background: { color: HEX.dk2 },''', '''    title: "TITLE_DARK",
    background: { path: BG },''')
rep('''    title: "CONTENT",
    background: { color: "FFFFFF" },''', '''    title: "CONTENT",
    background: { path: BG },''')
rep('fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },',
    'fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "center", margin: 0 }, text: "" } },')

# ---------------------------------------------------------------- 1. title slide like the reference deck
a = s.index("  let s = pres.addSlide({ masterName: \"TITLE_DARK\", sectionTitle: \"Opening\" });")
b = s.index("  notes(s, \"We present SARA")
s = s[:a] + '''  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Opening" });
  s.addImage({ path: FIG + "iub_logo.png", x: 0.55, y: 0.35, w: 1.25, h: 1.25 * 2071 / 2500, objectName: "IUB logo" });
  s.addText("SARA  ·  Smart Assistive Reading Aid", { x: 1.95, y: 0.5, w: 6.2, h: 0.6, fontSize: 18, bold: true, color: HEX.accent1, valign: "middle", isTextBox: true, margin: 0, objectName: "Project name" });
  s.addText(TITLE, { x: 0.6, y: 1.65, w: 7.9, h: 2.25, fontFace: "Arial", fontSize: 28, bold: true, color: HEX.dk1, isTextBox: true, margin: 0, valign: "top", objectName: "Title" });
  s.addShape(pres.shapes.LINE, { x: 0.6, y: 4.0, w: 7.9, h: 0, line: { color: HEX.accent3, width: 1.5 }, objectName: "Title rule" });
  s.addText(SUBTITLE, { x: 0.6, y: 4.12, w: 7.9, h: 0.85, fontSize: 16, italic: true, color: HEX.accent1, isTextBox: true, margin: 0, valign: "top", objectName: "Subtitle" });
  s.addText([
    { text: "Supervisor:", options: { bold: true, breakLine: true } },
    { text: "Mohammad Shidujaman, PhD", options: { breakLine: true } },
    { text: "Assistant Professor", options: { breakLine: true } },
    { text: "Department of Computer Science & Engineering" },
  ], { x: 0.6, y: 5.25, w: 4.1, h: 1.6, fontSize: 14, color: HEX.dk1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 3, objectName: "Supervisor" });
  s.addText([
    { text: "Submitted By:", options: { bold: true, breakLine: true } },
    { text: "MD Abdun Fattah Lam (2220014)", options: { breakLine: true } },
    { text: "Mukut Protim Memo (2020086)", options: { breakLine: true } },
    { text: "Tanjila Hossain Tapti (2130561)" },
  ], { x: 4.85, y: 5.25, w: 3.7, h: 1.6, fontSize: 14, color: HEX.dk1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 3, objectName: "Submitted by" });
  photo(s, AS + "v2_collage.jpg", 8.85, 0.45, 4.0, 6.6, "Second prototype: front, side and back");
''' + s[b:]

# ---------------------------------------------------------------- slide headings: formal names
for a, b in [
    ('"Three everyday challenges"', '"Introduction"'),
    ('"Why existing solutions fall short"', '"Problem Statement"'),
    ('"SARA in action: ask the glass, not a stranger"', '"Proposed Solution: SARA"'),
    ('"Five modes, started by button or by voice"', '"System Features"'),
    ('"Six research questions"', '"Research Questions"'),
    ('"What existing devices miss"', '"Literature Review: Existing Assistive Devices"'),
    ('"How the glass works"', '"System Architecture"'),
    ('"From a desk prototype to a wearable glass"', '"Evolution of the Prototype"'),
    ('"Taka detection: 22,333 training images, no hand labels"', '"Banknote Detection: Synthetic Training Data"'),
    ('"Notes held in a hand: from 1 in 8 to 1 in 2"', '"Banknote Detection: Hand-Held Notes"'),
    ('"Live tests with real notes"', '"Live Detection Tests"'),
    ('"Counterfeit check: one view is enough with prefix training"', '"Counterfeit Screening: Prefix-Robust Multi-View Training"'),
    ('"Unseen counterfeit prints: a stricter test"', '"Counterfeit Screening: Evaluation on Unseen Prints"'),
    ('"DMWR: a tiny, physics-based watermark checker"', '"Watermark Verification: DMWR"'),
    ('"Safe before decisive: the glass never says \\"counterfeit\\""', '"Safe Verdict Policy"'),
    ('"UV fibre check: the glass carries its own UV lamp"', '"UV Security-Fibre Verification"'),
    ('"Bangla reading: rendered text and real street photos"', '"Bangla and English Text Recognition"'),
    ('"Voice: a natural Bangla voice and spoken commands"', '"Voice Interaction"'),
    ('"Running on the Raspberry Pi 5"', '"Performance on the Raspberry Pi 5"'),
    ('"Testing the worn glass: where the project stands"', '"Prototype Testing and Project Status"'),
    ('"Future work: the roadmap"', '"Future Directions"'),
    ('s.addText("Links", { placeholder: "title" });', 's.addText("Resources and Links", { placeholder: "title" });'),
]:
    rep(a, b)

# ---------------------------------------------------------------- 2. introduction: one sourced line, picture a little smaller
rep('''  ["Money", "Reading", "Surroundings"].forEach((head, i) =>
    s.addText(head, { x: 0.6 + i * 4.03, y: 1.38, w: 4.03, h: 0.55,''', '''  s.addText("Everyday challenges of people with visual impairment: at least 2.2 billion people worldwide have a near or distance vision impairment " + cite("who"),
    { x: 0.6, y: 1.25, w: 12.1, h: 0.45, fontSize: 16, italic: true, color: C.text1, align: "center", isTextBox: true, margin: 0, objectName: "Introduction line" });
  ["Money", "Reading", "Surroundings"].forEach((head, i) =>
    s.addText(head, { x: 1.15 + i * 11.0 / 3, y: 1.78, w: 11.0 / 3, h: 0.5,''')
rep('''  s.addImage({ path: AS + "cartoon_full.jpg", x: 0.6, y: 2.0, w: 12.1, h: 12.1 * 793 / 1983, objectName: "Without and with SARA" });''',
    '''  s.addImage({ path: AS + "cartoon_full.jpg", x: 1.15, y: 2.32, w: 11.0, h: 11.0 * 793 / 1983, objectName: "Without and with SARA" });''')

# ---------------------------------------------------------------- 3. problem statement: wording that matches the sources
rep('''  s.addText("most of them in low- and middle-income regions " + cite("who"),''',
    '''  s.addText("prevalence in low- and middle-income regions is about 4 times higher " + cite("who"),''')
rep('''"OrCam MyEye does not support Bangla " + cite("orcam")''', '''"Bangla is not among OrCam MyEye's listed languages " + cite("orcam")''')
rep('''Sources: ${cite("who")} WHO 2023;''', '''Sources: ${cite("who")} WHO 2026;''')
rep('''most of them in low- and middle-income countries. Three gaps''', '''with a prevalence in low- and middle-income regions about four times that in high-income regions. Three gaps''')
rep('''OrCam MyEye, the best-known commercial reading glass, does not support Bangla.''', '''OrCam MyEye, the best-known commercial reading glass, does not list Bangla among its languages.''')

# ---------------------------------------------------------------- 4. proposed solution: sources inline
rep('''    ["mic", HEX.accent1, "Speak", "\\"Glass, check money\\""],''', '''    ["mic", HEX.accent1, "Speak", "\\"Glass, check money\\" " + cite("vosk")],''')
rep('''    ["speak", HEX.dk2, "Hear the answer", "Bangla speech + vibration"],''', '''    ["speak", HEX.dk2, "Hear the answer", "Bangla speech + vibration " + cite("piper")],''')

# ---------------------------------------------------------------- 5. system features: model sources
rep('''  notes(s, "This is the current prototype''', '''  sources(s, `Models: text ${cite("craft", "easyocr")} CRAFT + EasyOCR; objects ${cite("yolo", "coco")} YOLOv8 on COCO; emotion ${cite("rafdb")} RAF-DB; voice commands ${cite("vosk")} Vosk.`);
  notes(s, "This is the current prototype''')

# ---------------------------------------------------------------- new slide after the architecture: Hardware Design (as in the reference deck)
a = s.index('  notes(s, "The camera picture, the buttons and voice commands go to a mode')
b = s.index("\n", a) + 1
s = s[:b] + '''
  // ================================================================ 8b. hardware design
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Solution" });
  s.addText("Hardware Design", { placeholder: "title" });
  const parts = [
    ["hw_pi5.jpg", "Raspberry Pi 5", "4-core 2.4 GHz, 8 GB; runs every model " + cite("rpi5")],
    ["hw_camera.jpg", "Camera Module 2", "8 MP IMX219, used at 640×480"],
    ["v2_buttons.jpg", "Five push-buttons", "MODE, ACTION, READ, VOL+, VOL−"],
    ["hw_motor.jpg", "Vibration motor", "A buzz when a note is found"],
    ["v2_headphone.jpg", "Bone-conduction headphone", "Speech; the ears stay free"],
    ["hw_powerbank.jpg", "Power bank", "USB-C supply for the Pi 5"],
    ["v2_frame.jpg", "Frame and UV lamp", "3D-printed; UV-A 360–395 nm"],
    ["v2_enclosure.jpg", "Rear enclosure", "Ventilated case for the Pi 5"],
  ];
  parts.forEach(([f, h, t], i) => {
    const x = 0.6 + (i % 4) * 3.1, y = 1.4 + Math.floor(i / 4) * 2.62;
    card(s, x, y, 2.9, 2.45, h + " card", "FFFFFF");
    s.addImage({ path: FIG + f, x: x + 0.2, y: y + 0.12, w: 2.5, h: 1.42, sizing: { type: "contain", w: 2.5, h: 1.42 }, objectName: h + " photo" });
    s.addText(h, { x: x + 0.12, y: y + 1.6, w: 2.66, h: 0.38, fontSize: 15, bold: true, color: C.text2, align: "center", isTextBox: true, margin: 0, objectName: h + " name" });
    s.addText(t, { x: x + 0.12, y: y + 1.98, w: 2.66, h: 0.4, fontSize: 12, color: C.text1, align: "center", isTextBox: true, margin: 0, objectName: h + " detail" });
  });
  sources(s, "Pictures (Wikimedia Commons): Raspberry Pi 5 by SimonWaldherr, CC BY 4.0; Camera Module v2 by the Raspberry Pi Foundation, CC BY-SA 4.0; vibration motor by Raimond Spekking, CC BY-SA 4.0; power bank by Pinguinguy, CC0. Buttons, headphone, frame and enclosure: our prototype.");
  notes(s, "The hardware of the second prototype. A Raspberry Pi 5 runs every model on its CPU. The Camera Module 2 is used at 640 by 480. Five push-buttons on the temple select and start the modes, a vibration motor confirms a found note, and a bone-conduction headphone speaks the result while leaving the ears free. A 3D-printed frame carries the camera and the UV-A lamp, and a ventilated enclosure at the back of the head carries the Pi. A power bank supplies it.");
''' + s[b:]

# ---------------------------------------------------------------- Raspberry Pi slide: cite the board
rep('''  notes(s, "Measured on the Raspberry Pi 5 itself.''', '''  sources(s, `Board: ${cite("rpi5")} Raspberry Pi 5 (8 GB). All timings are ours (benchmark_report/, 1 and 13 October 2026).`);
  notes(s, "Measured on the Raspberry Pi 5 itself.''')

# ---------------------------------------------------------------- thank-you slide like the reference deck
a = s.index('  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Closing" });')
b = s.index('  notes(s, "Thank you; we are happy to take questions.");')
s = s[:a] + '''  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Closing" });
  s.addText("THANK YOU", { x: 0.6, y: 2.35, w: 12.1, h: 1.6, fontFace: "Arial", fontSize: 72, color: HEX.dk1, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "Thanks" });
  s.addText("Questions?", { x: 0.6, y: 4.0, w: 12.1, h: 0.6, fontSize: 26, color: HEX.accent1, align: "center", isTextBox: true, margin: 0, objectName: "Questions" });
  s.addText("SARA  ·  Smart Assistive Reading Aid  ·  Independent University, Bangladesh", { x: 0.6, y: 6.5, w: 12.1, h: 0.4, fontSize: 14, color: HEX.accent5, align: "center", isTextBox: true, margin: 0, objectName: "Closing name" });
''' + s[b:]

# ---------------------------------------------------------------- references slide: in citation order; links from the URL map
rep('''  const half = Math.ceil(REFS.length / 2);
  [REFS.slice(0, half), REFS.slice(half)].forEach((list, col) => {
    s.addText(list.map((r, i) => ({ text: `[${col * half + i + 1}]  ${r[1]}`, options: { breakLine: i < list.length - 1 } })),''',
    '''  const unused = REFS.filter((r) => !NUM[r[0]]).map((r) => r[0]);
  if (unused.length) throw new Error("references never cited: " + unused.join(", "));
  const CITED = REFS.slice().sort((a, b) => NUM[a[0]] - NUM[b[0]]);
  const half = Math.ceil(CITED.length / 2);
  [CITED.slice(0, half), CITED.slice(half)].forEach((list, col) => {
    s.addText(list.map((r, i) => ({ text: `[${NUM[r[0]]}]  ${r[1]}`, options: { breakLine: i < list.length - 1 } })),''')
s = s.replace("{ x: 0.6 + col * 6.15, y: 1.4, w: 5.95, h: 5.4, fontSize: 13,", "{ x: 0.6 + col * 6.15, y: 1.4, w: 5.95, h: 5.4, fontSize: 12,")
import re
s, n = re.subn(r"REFS\[NUM\.(\w+) - 1\]\[2\]", r"URL.\1", s)
assert n == 12, n

p.write_text(s, encoding="utf-8")
print("build4 ok")
