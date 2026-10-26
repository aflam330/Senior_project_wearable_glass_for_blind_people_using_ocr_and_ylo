// SARA presentation, third design (26 October 2026): fewer words, larger type, illustrated problem and solution,
// one takeaway per results slide. Built from build2.js; the illustrations are the two pictures in the slides folder.
// node build3.js   -> ../../thesis report/slides/SARA_Presentation.pptx   (then add_transitions.py adds fades)
const pptxgen = require("pptxgenjs");
const React = require("react");
const RDS = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const path = require("path");
const { applyTheme } = require("C:/Users/LAM/.claude/skills/synced/78df2a5e-005c-4f3e-a592-b1286ff813dd_9e1bb536-53da-4e98-a30a-05e6767fad10/pptx/scripts/apply_theme.js");

const OUT = process.env.DECK_OUT || path.resolve(__dirname, "../../thesis report/slides/SARA_Presentation.pptx");
const FIG = "E:/Final SP/Thesis Report and paper/thesis report/My report/DIagram fig tab/";
const REP = "E:/Final SP/Thesis Report and paper/thesis report/My report/";
const AS = path.resolve(__dirname, "assets") + "/";

const THEME = {
  name: "SARA",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1C2321", lt1: "FFFFFF", dk2: "0B4D3B", lt2: "EEF5F1",
    accent1: "006A4E", accent2: "D7263D", accent3: "3A8F72", accent4: "E6A817", accent5: "5E6B66", accent6: "B9DCCD",
    hlink: "006A4E", folHlink: "3A8F72",
  },
};
const HEX = THEME.colors;
const BN = "Nirmala UI";          // Bangla text

// ------------------------------------------------------------------ icons
async function icon(name, color, size = 256) {
  const svg = RDS.renderToStaticMarkup(React.createElement(fa[name], { color: "#" + color, size: String(size) }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

// ------------------------------------------------------------------ references (numbered as cited)
const REFS = [
  ["who", "World Health Organization. Blindness and vision impairment: fact sheet, 2023.", "https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment"],
  ["orcam", "OrCam Technologies. OrCam MyEye.", "https://www.orcam.com"],
  ["glass", "Ali A., H. et al. A Google Glass based real-time scene analysis for the visually impaired. IEEE Access 9, 2021.", "https://ieeexplore.ieee.org/abstract/document/9648186"],
  ["airis", "Brilli, D. D. et al. AIris: an AI-powered wearable assistive device for the visually impaired. arXiv:2405.07606, 2024.", "https://arxiv.org/abs/2405.07606"],
  ["coco", "Lin, T.-Y. et al. Microsoft COCO: common objects in context. ECCV, 2014.", "https://cocodataset.org"],
  ["banglataka", "A diverse image dataset for Bangladeshi currency recognition (BanglaTaka). Mendeley Data.", "https://data.mendeley.com/datasets/3cv2sypkkh/1"],
  ["nstu", "Rafi, M. J. A., Rony, M., Majadi, N. NSTU-BDTAKA: an open dataset for Bangladeshi paper currency detection and recognition. Data in Brief 55, 2024.", "https://www.sciencedirect.com/science/article/pii/S2352340924006681"],
  ["jaal", "Piyas, B. R. C. et al. JaalTaka: a benchmark dataset for detecting counterfeit Bangladeshi banknotes. Mendeley Data V2, 2025.", "https://data.mendeley.com/datasets/2m7wk5cy4c/2"],
  ["yolo", "Jocher, G., Chaurasia, A., Qiu, J. Ultralytics YOLOv8.", "https://github.com/ultralytics/ultralytics"],
  ["mnv2", "Sandler, M. et al. MobileNetV2: inverted residuals and linear bottlenecks. CVPR, 2018.", "https://arxiv.org/abs/1801.04381"],
  ["easyocr", "JaidedAI. EasyOCR: ready-to-use OCR with 80+ languages.", "https://github.com/JaidedAI/EasyOCR"],
  ["craft", "Baek, Y. et al. Character region awareness for text detection (CRAFT). CVPR, 2019.", "https://arxiv.org/abs/1904.01941"],
  ["mlt", "Nayef, N. et al. ICDAR2019 robust reading challenge on multi-lingual scene text (RRC-MLT-2019). ICDAR, 2019.", "https://rrc.cvc.uab.es/?ch=15"],
  ["rafdb", "Li, S., Deng, W., Du, J. Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild (RAF-DB). CVPR, 2017.", "http://www.whdeng.cn/RAF/model1.html"],
  ["piper", "Rhasspy. Piper: a fast, local neural text-to-speech system.", "https://github.com/rhasspy/piper"],
  ["espeak", "eSpeak NG text-to-speech.", "https://github.com/espeak-ng/espeak-ng"],
  ["vosk", "Alpha Cephei. Vosk offline speech recognition toolkit.", "https://alphacephei.com/vosk/"],
  ["esc50", "Piczak, K. J. ESC: dataset for environmental sound classification. ACM Multimedia, 2015.", "https://github.com/karolpiczak/ESC-50"],
  ["jaal26", "Piyas, B. R. C. et al. JaalTaka: a benchmark dataset for detecting counterfeit Bangladeshi banknotes. Data in Brief 64, 112377, 2026.", "https://doi.org/10.1016/j.dib.2025.112377"],
  ["uddin", "Uddin, M. S., Das, P. P., Roney, M. S. A. Image-based approach for the detection of counterfeit banknotes of Bangladesh. ICIEV, 2016.", "https://doi.org/10.1109/ICIEV.2016.7760162"],
  ["pathak", "Pathak, A. et al. Enhanced counterfeit detection of Bangladesh currency through convolutional neural networks. IJIRCST 12(2), 2024.", "https://doi.org/10.55524/ijircst.2024.12.2.2"],
  ["uvink", "Chowdhury, A. A. et al. Evaluation of deep learning models on UV ink: a fake money detection scheme with RPN. ICCA (ACM), 2022.", "https://doi.org/10.1145/3542954.3542965"],
  ["mutanen", "Mutanen, J., Jaaskelainen, T., Parkkinen, J. P. S. Luminescent security properties of banknotes. IS&T PICS Conference, 2003, pp. 421-424.", "https://www.imaging.org/common/uploaded%20files/pdfs/Papers/2003/PICS-0-287/8558.pdf"],
];
const ORDER = ["who", "orcam", "glass", "airis", "craft", "easyocr", "yolo", "coco", "rafdb", "piper", "espeak", "banglataka", "nstu", "jaal", "jaal26", "uddin", "pathak", "mnv2", "uvink", "mutanen", "mlt", "vosk", "esc50"];
REFS.sort((a, b) => ORDER.indexOf(a[0]) - ORDER.indexOf(b[0]));
const NUM = Object.fromEntries(REFS.map((r, i) => [r[0], i + 1]));
const cite = (...keys) => "[" + keys.map((k) => NUM[k]).sort((a, b) => a - b).join(", ") + "]";

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";              // 13.33 x 7.5 in
  pres.title = "SARA: Smart Assistive Reading Aid";
  pres.author = "MD Abdun Fattah Lam, Mukut Protim Memo, Tanjila Hossain Tapti";
  pres.company = "Independent University, Bangladesh";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  const I = {};
  for (const [k, n, col] of [
    ["store", "FaStore", HEX.dk2], ["pills", "FaPills", HEX.dk2], ["road", "FaWalking", HEX.dk2], ["money", "FaMoneyBillWave", HEX.accent1],
    ["blind", "FaBlind", HEX.dk1], ["friends", "FaUserFriends", HEX.accent5], ["q", "FaQuestion", HEX.accent2],
    ["lang", "FaLanguage", "FFFFFF"], ["wifi", "FaWifi", "FFFFFF"], ["warn", "FaExclamationTriangle", "FFFFFF"],
    ["book", "FaBookOpen", "FFFFFF"], ["cube", "FaCube", "FFFFFF"], ["cash", "FaMoneyBillWave", "FFFFFF"], ["smile", "FaSmile", "FFFFFF"],
    ["globe", "FaGlobe", "FFFFFF"], ["mic", "FaMicrophone", "FFFFFF"], ["cam", "FaCamera", "FFFFFF"], ["chip", "FaMicrochip", "FFFFFF"],
    ["hand", "FaHandPointer", "FFFFFF"], ["speak", "FaVolumeUp", "FFFFFF"], ["shield", "FaShieldAlt", "FFFFFF"], ["therm", "FaThermometerHalf", "FFFFFF"],
    ["git", "FaGithub", HEX.dk2], ["data", "FaDatabase", HEX.dk2], ["tools", "FaTools", HEX.dk2], ["glasses", "FaGlasses", "FFFFFF"],
    ["check", "FaCheckCircle", HEX.accent1], ["link", "FaLink", HEX.dk2], ["users", "FaUsers", "FFFFFF"], ["eye", "FaEye", "FFFFFF"],
  ]) I[k] = await icon(n, col);

  // ---------------------------------------------------------------- layouts
  const FOOT = { x: 0.6, y: 7.02, w: 6.5, h: 0.28, fontSize: 10, color: C.accent5, isTextBox: true, margin: 0 };
  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: HEX.dk2 },
    objects: [],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: "FFFFFF" },
    margin: [0.5, 0.6, 0.6, 0.6],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.38, w: 12.1, h: 0.85, fontFace: "Cambria", fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
      { text: { text: "SARA  ·  Smart Assistive Reading Aid  ·  IUB CSE Senior Project", options: FOOT } },
    ],
    slideNumber: { x: 12.2, y: 7.02, w: 0.5, h: 0.28, fontSize: 10, color: HEX.accent5, align: "right" },
  });

  // helpers ---------------------------------------------------------
  const sources = (s, text) =>
    s.addText(text, { x: 0.6, y: 6.62, w: 12.1, h: 0.32, fontSize: 10, italic: true, color: C.accent5, isTextBox: true, margin: 0, objectName: "Sources" });
  const circleIcon = (s, key, x, y, d, fill, name) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: name + " circle" });
    const p = d * 0.24;
    s.addImage({ data: I[key], x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, objectName: name + " icon" });
  };
  const card = (s, x, y, w, h, name, fill = C.background2) =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName: name });
  const photo = (s, file, x, y, w, h, name) => s.addImage({ path: file, x, y, w, h, sizing: { type: "cover", w, h }, objectName: name });
  const stat = (s, x, y, w, big, small, name, color = C.accent1) => {
    s.addText(big, { x, y, w, h: 0.75, fontFace: "Cambria", fontSize: 36, bold: true, color, isTextBox: true, margin: 0, objectName: name + " value" });
    s.addText(small, { x, y: y + 0.75, w, h: 0.6, fontSize: 13, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: name + " label" });
  };
  const bubble = (s, text, x, y, w, h, fill, color, name, fontFace) =>
    s.addText(text, { shape: pres.shapes.ROUNDED_RECTANGULAR_CALLOUT, x, y, w, h, fill: { color: fill }, line: { color: fill }, color, fontSize: 14,
      bold: true, align: "center", valign: "middle", isTextBox: true, margin: 0.06, fontFace, objectName: name });
  const tag = (s, text, x, y, w, fill, color, name) => s.addText(text, { shape: pres.shapes.ROUNDED_RECTANGLE, x, y, w, h: 0.36, rectRadius: 0.06,
    fill: { color: fill }, line: { color: fill }, color, fontSize: 12, bold: true, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: name });
  const notes = (s, t) => s.addNotes(t);
  // a straight line from (x1, y1) to (x2, y2), x2 >= x1; sizes always positive (negative sizes make PowerPoint refuse the file)
  const connect = (s, x1, y1, x2, y2, name, arrow = false) =>
    s.addShape(pres.shapes.LINE, { x: x1, y: Math.min(y1, y2), w: Math.max(x2 - x1, 0.01), h: Math.abs(y2 - y1), flipV: y2 < y1,
      line: { color: "9DB8AD", width: 1.25, ...(arrow ? { endArrowType: "triangle" } : {}) }, objectName: name });
  const chartFont = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt" };
  const chartFrame = {
    catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.accent5, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    valGridLine: { color: "DDE7E2", size: 0.75 }, catGridLine: { style: "none" }, dataLabelFontSize: 11, dataLabelColor: HEX.dk1,
    legendFontSize: 12, titleFontSize: 13, titleColor: HEX.dk1, ...chartFont,
  };

  // ================================================================ 1. title
  pres.addSection({ title: "Opening" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Opening" });
  photo(s, AS + "v2_collage.jpg", 7.43, 0, 5.9, 7.5, "Second prototype: front, side and back");
  circleIcon(s, "glasses", 0.75, 0.75, 0.75, HEX.accent1, "Glasses");
  s.addText("SARA", { x: 0.75, y: 1.75, w: 6.2, h: 1.0, fontFace: "Cambria", fontSize: 54, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "Title" });
  s.addText([{ text: "Smart Assistive Reading Aid", options: { bold: true, breakLine: true } },
    { text: "Offline AI Smart Glasses for Reading, Object Recognition and Bangladeshi Taka Verification" }],
    { x: 0.75, y: 2.8, w: 6.2, h: 1.8, fontSize: 22, color: "E4EFEA", isTextBox: true, margin: 0, valign: "top", objectName: "Subtitle" });
  s.addText([
    { text: "MD Abdun Fattah Lam (2220014)  ·  Mukut Protim Memo (2020086)", options: { breakLine: true } },
    { text: "Tanjila Hossain Tapti (2130561)", options: { breakLine: true } },
    { text: "Supervisor: Mohammad Shidujaman, PhD, Assistant Professor", options: { breakLine: true } },
    { text: "Department of Computer Science and Engineering, Independent University, Bangladesh" },
  ], { x: 0.75, y: 4.95, w: 6.3, h: 1.6, fontSize: 13, color: "CFE2DA", isTextBox: true, margin: 0, paraSpaceAfter: 4, objectName: "Authors" });
  notes(s, "We present SARA, a smart glass for blind and low-vision people in Bangladesh. It reads Bangla and English text, names objects, recognises Taka notes and checks them carefully, and it works on the device without the internet. The photo is the current prototype.");

  // ================================================================ 2. problem: scenario storyboard
  pres.addSection({ title: "Problem" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
  s.addText("Three everyday challenges", { placeholder: "title" });
  ["Money", "Reading", "Surroundings"].forEach((head, i) =>
    s.addText(head, { x: 0.6 + i * 4.03, y: 1.38, w: 4.03, h: 0.55, fontFace: "Cambria", fontSize: 26, bold: true, color: C.text2, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: head + " label" }));
  s.addImage({ path: AS + "cartoon_full.jpg", x: 0.6, y: 2.0, w: 12.1, h: 12.1 * 793 / 1983, objectName: "Without and with SARA" });
  s.addText("Illustration", { x: 11.2, y: 7.02, w: 0.95, h: 0.28, fontSize: 10, italic: true, color: C.accent5, align: "right", isTextBox: true, margin: 0, objectName: "Illustration note" });
  notes(s, "Three everyday moments; the top row is today, the bottom row is with SARA. At the shop, the blind customer receives change and must trust the shopkeeper about which note it is and whether it is real. At the pharmacy, the medicine label is in Bangla and no affordable device reads it. On the street, they must stop and ask a stranger what is ahead. In every case the answer comes from another person and cannot be checked. With SARA, the glass names the note, reads the label and names what is ahead.");

  // ================================================================ 3. problem in numbers + gaps
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
  s.addText("Why existing solutions fall short", { placeholder: "title" });
  card(s, 0.6, 1.45, 4.2, 4.95, "Statistic card", C.text2);
  s.addText("2.2 billion", { x: 0.95, y: 2.1, w: 3.6, h: 1.0, fontFace: "Cambria", fontSize: 48, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "WHO figure" });
  s.addText("people worldwide have a near or distance vision impairment " + cite("who"),
    { x: 0.95, y: 3.15, w: 3.55, h: 1.2, fontSize: 19, color: "E4EFEA", isTextBox: true, margin: 0, valign: "top", objectName: "WHO label" });
  s.addText("most of them in low- and middle-income regions " + cite("who"),
    { x: 0.95, y: 4.6, w: 3.55, h: 1.2, fontSize: 17, italic: true, color: "CFE2DA", isTextBox: true, margin: 0, valign: "top", objectName: "WHO note" });
  const gaps = [
    ["lang", HEX.accent1, "Limited Bangla support", "OrCam MyEye does not support Bangla " + cite("orcam")],
    ["wifi", HEX.accent3, "Internet dependence", "Research glasses send pictures to a server " + cite("glass", "airis")],
    ["warn", HEX.accent2, "Limited Taka verification", "Reviewed systems do not name Taka notes " + cite("orcam", "glass", "airis")],
  ];
  gaps.forEach(([k, col, h, t], i) => {
    const y = 1.45 + i * 1.7;
    card(s, 5.15, y, 7.55, 1.5, "Gap " + (i + 1));
    circleIcon(s, k, 5.4, y + 0.3, 0.9, col, "Gap " + (i + 1));
    s.addText(h, { x: 6.55, y: y + 0.2, w: 6.0, h: 0.55, fontFace: "Cambria", fontSize: 24, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "Gap " + (i + 1) + " heading" });
    s.addText(t, { x: 6.55, y: y + 0.78, w: 6.0, h: 0.55, fontSize: 18, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "Gap " + (i + 1) + " text" });
  });
  sources(s, `Sources: ${cite("who")} WHO 2023; ${cite("orcam")} OrCam MyEye; ${cite("glass")} Ali A. et al., IEEE Access 2021; ${cite("airis")} Brilli et al., arXiv 2024.`);
  notes(s, "The World Health Organization counts at least 2.2 billion people with a vision impairment, most of them in low- and middle-income countries. Three gaps keep existing help away from Bangladeshi users. OrCam MyEye, the best-known commercial reading glass, does not support Bangla. The research glasses we reviewed send every picture to a cloud server, so without a network there is no answer. And none of the systems we reviewed names Bangladeshi Taka notes, let alone screens them safely: a wrong value or a false accusation of counterfeit has real costs.");

  // ================================================================ 4. with SARA scenario
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
  s.addText("SARA in action: ask the glass, not a stranger", { placeholder: "title" });
  photo(s, AS + "1_20.jpg", 0.6, 1.45, 3.1, 5.1, "SARA prototype, worn");
  s.addText("Our prototype", { x: 0.6, y: 6.6, w: 3.1, h: 0.32, fontSize: 12, italic: true, color: C.accent5, align: "center", isTextBox: true, margin: 0, objectName: "Prototype caption" });
  const steps = [
    ["mic", HEX.accent1, "Speak", "\"Glass, check money\""],
    ["chip", HEX.accent3, "Detect offline", "Raspberry Pi 5, no internet"],
    ["speak", HEX.dk2, "Hear the answer", "Bangla speech + vibration"],
  ];
  steps.forEach(([k, col, h, t], i) => {
    const x = 4.1 + i * 2.95;
    circleIcon(s, k, x, 1.5, 0.72, col, "Step " + (i + 1));
    s.addText(h, { x: x + 0.82, y: 1.47, w: 1.95, h: 0.45, fontFace: "Cambria", fontSize: 18, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "Step " + (i + 1) + " heading" });
    s.addText(t, { x: x + 0.82, y: 1.92, w: 1.95, h: 0.6, fontSize: 14, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "Step " + (i + 1) + " text" });
    if (i < 2) connect(s, x + 2.72, 1.9, x + 2.9, 1.9, "Step arrow " + (i + 1), true);
  });
  const answers = [["cash", "Money", "Names the note and screens it with care"], ["book", "Reading", "Reads Bangla and English labels aloud"], ["eye", "Surroundings", "Names people and objects ahead"]];
  answers.forEach(([k, h, t], i) => {
    const x = 4.1 + i * 2.95;
    card(s, x, 2.85, 2.75, 2.65, h + " card");
    circleIcon(s, k, x + 0.98, 3.05, 0.8, HEX.accent1, h);
    s.addText(h, { x: x + 0.1, y: 3.95, w: 2.55, h: 0.45, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text2, align: "center", isTextBox: true, margin: 0, objectName: h + " label" });
    s.addText(t, { x: x + 0.15, y: 4.45, w: 2.45, h: 0.9, fontSize: 15, color: C.text1, align: "center", isTextBox: true, margin: 0, valign: "top", objectName: h + " answer" });
  });
  s.addText([{ text: "পাঁচশত টাকার নোট", options: { fontFace: BN, fontSize: 24, bold: true, color: HEX.accent1 } },
    { text: "   \"500 Taka note\": the answer the glass speaks", options: { fontSize: 16, color: HEX.dk1 } }],
    { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, valign: "middle", x: 4.1, y: 5.75, w: 8.6, h: 0.75, margin: 12, isTextBox: true, objectName: "Spoken answer" });
  notes(s, "Here is the same moment with SARA. The user says glass, check money. The camera picture is processed on the Raspberry Pi 5 without the internet, and the glass answers in Bangla, for example five hundred Taka note, with a short vibration. The same three steps read a medicine label or name what is ahead.");

  // ================================================================ 5. SARA at a glance
  pres.addSection({ title: "Solution" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Solution" });
  s.addText("Five modes, started by button or by voice", { placeholder: "title" });
  photo(s, AS + "1_17.jpg", 0.6, 1.45, 5.0, 2.81, "Prototype, side view");
  photo(s, AS + "1_16.jpg", 0.6, 4.36, 5.0, 2.14, "Prototype, rear enclosure");
  const modes = [
    ["book", "Text reading", "Bangla and English"],
    ["cube", "Objects", "Named in Bangla"],
    ["cash", "Taka notes", "Value + safe check"],
    ["smile", "Emotion", "Face in front"],
    ["globe", "Online description", "Optional; needs internet"],
  ];
  modes.forEach(([k, h, t], i) => {
    const x = 6.0 + (i % 3) * 2.3 + (i >= 3 ? 1.15 : 0), y = 1.45 + Math.floor(i / 3) * 1.95;
    card(s, x, y, 2.1, 1.8, h + " card");
    circleIcon(s, k, x + 0.7, y + 0.15, 0.7, HEX.accent1, h);
    s.addText(h, { x: x + 0.05, y: y + 0.9, w: 2.0, h: 0.4, fontSize: 16, bold: true, color: C.text2, align: "center", isTextBox: true, margin: 0, objectName: h + " heading" });
    s.addText(t, { x: x + 0.05, y: y + 1.3, w: 2.0, h: 0.4, fontSize: 13, color: C.text1, align: "center", isTextBox: true, margin: 0, objectName: h + " text" });
  });
  card(s, 6.0, 5.45, 6.7, 1.05, "Interaction bar", C.text2);
  circleIcon(s, "hand", 6.2, 5.6, 0.75, HEX.accent3, "Buttons");
  circleIcon(s, "mic", 7.05, 5.6, 0.75, HEX.accent2, "Voice");
  s.addText([{ text: "How the user starts a mode: ", options: { bold: true } }, { text: "five buttons on the frame, or a voice command: \"Glass, read text\"" }],
    { x: 8.0, y: 5.5, w: 4.55, h: 0.95, fontSize: 15, color: "FFFFFF", valign: "middle", isTextBox: true, margin: 0, objectName: "Interaction text" });
  notes(s, "This is the current prototype: the camera and five buttons sit on the frame, a bone-conduction earphone leaves the ears free, and the Raspberry Pi 5 rides in a ventilated enclosure at the back of the head. Five modes cover text, objects, Taka notes, emotion and an optional online description. Every mode can be started with a button or with a voice command that begins with the word glass.");

  // ================================================================ 6. research questions
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Solution" });
  s.addText("Six research questions", { placeholder: "title" });
  const rqs = [
    ["System", "Can it all run on a Raspberry Pi 5?"],
    ["Synthetic data", "Can unlabelled training data detect real notes?"],
    ["Number of views", "Can one view of a note be enough?"],
    ["Unseen prints", "Does the check hold on new counterfeit prints?"],
    ["Safety", "Can it never accuse a genuine note?"],
    ["Bangla OCR", "What matters most for reading Bangla?"],
  ];
  rqs.forEach(([h, t], i) => {
    const x = 0.6 + (i % 3) * 4.1, y = 1.55 + Math.floor(i / 3) * 2.45;
    card(s, x, y, 3.85, 2.2, "RQ" + (i + 1));
    s.addText(String(i + 1), { shape: pres.shapes.OVAL, x: x + 0.25, y: y + 0.25, w: 0.6, h: 0.6, fill: { color: HEX.accent1 }, line: { color: HEX.accent1 },
      color: "FFFFFF", fontSize: 18, bold: true, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "RQ" + (i + 1) + " number" });
    s.addText(h, { x: x + 1.0, y: y + 0.3, w: 2.7, h: 0.5, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "RQ" + (i + 1) + " heading" });
    s.addText(t, { x: x + 0.25, y: y + 1.05, w: 3.4, h: 0.95, fontSize: 20, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "RQ" + (i + 1) + " text" });
  });
  notes(s, "Six questions organise the talk. First, whether a multi-function Bangla glass runs on a Raspberry Pi 5 and how fast. Second, whether a Taka detector trained without hand labels works on real, hand-held photos. Third, why multi-view counterfeit models fail with fewer views and whether training fixes it. Fourth, how much accuracy drops on counterfeit prints never seen in training. Fifth, whether the glass can give counterfeit advice that never accuses a genuine note. Sixth, which part of a Bangla OCR pipeline matters most for a wearable camera.");

  // ================================================================ 7. existing devices
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Solution" });
  s.addText("What existing devices miss", { placeholder: "title" });
  const devs = [["dev_orcam.jpg", "OrCam MyEye " + cite("orcam")], ["dev_googleglass.jpg", "Google Glass analysis " + cite("glass")], ["dev_airis.jpg", "AIris " + cite("airis")]];
  devs.forEach(([f, cap], i) => {
    photo(s, FIG + f, 0.6 + i * 2.75, 4.85, 2.55, 1.35, cap);
    s.addText(cap, { x: 0.6 + i * 2.75, y: 6.24, w: 2.55, h: 0.3, fontSize: 12, color: C.accent5, isTextBox: true, margin: 0, align: "center", objectName: "Caption " + (i + 1) });
  });
  const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.dk2 }, align: "center" } });
  const yes = { text: "Yes", options: { color: HEX.accent1, bold: true, align: "center" } };
  const no = { text: "No", options: { color: HEX.accent2, align: "center" } };
  const part = { text: "Partly", options: { color: HEX.accent5, align: "center" } };
  const rowName = (t, bold) => ({ text: t, options: { bold: !!bold } });
  s.addTable([
    [hdr("System"), hdr("Reads Bangla"), hdr("Knows Taka"), hdr("Works offline"), hdr("Wearable")],
    [rowName("OrCam MyEye " + cite("orcam")), no, no, yes, yes],
    [rowName("Google Glass scene analysis " + cite("glass")), no, no, no, yes],
    [rowName("AIris (2024) " + cite("airis")), no, no, no, yes],
    [rowName("Raspberry Pi reading aids *"), no, no, part, part],
    [{ text: "SARA (ours)", options: { bold: true, fill: { color: HEX.lt2 } } }, { ...yes, options: { ...yes.options, fill: { color: HEX.lt2 } } },
      { ...yes, options: { ...yes.options, fill: { color: HEX.lt2 } } }, { ...yes, options: { ...yes.options, fill: { color: HEX.lt2 } } },
      { ...yes, options: { ...yes.options, fill: { color: HEX.lt2 } } }],
  ], { x: 0.6, y: 1.45, w: 8.05, colW: [3.25, 1.2, 1.2, 1.2, 1.2], fontSize: 16, color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "D5E2DC" }, rowH: 0.52, valign: "middle", objectName: "Comparison table" });
  card(s, 8.95, 1.45, 3.75, 4.95, "Gap card", C.text2);
  s.addText("The research gap", { x: 9.2, y: 1.7, w: 3.3, h: 0.5, fontFace: "Cambria", fontSize: 24, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "Gap heading" });
  s.addText("None of the systems we reviewed combines Bangla, Taka and offline use.",
    { x: 9.2, y: 2.35, w: 3.3, h: 1.5, fontSize: 20, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, valign: "top", objectName: "Gap main" });
  s.addText([
    { text: "Counterfeit tests reuse the same prints", options: { bullet: true, breakLine: true } },
    { text: "Few report speed on the real device", options: { bullet: true } },
  ], { x: 9.2, y: 4.0, w: 3.3, h: 2.2, fontSize: 16, color: "E4EFEA", isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 8, objectName: "Gap list" });
  s.addText("Rows with a reference: from the cited source. * Our assessment of typical hobby builds, not a cited result.", { x: 0.6, y: 4.6, w: 8.05, h: 0.25, fontSize: 11, italic: true, color: C.accent5, isTextBox: true, margin: 0, objectName: "Table note" });
  sources(s, "Photos: OrCam MyEye by Ylip, CC BY-SA 4.0; Google Glass by Ted Eytan, CC BY-SA 2.0 (Wikimedia Commons); AIris from " + cite("airis") + ". Comparison from " + cite("orcam", "glass", "airis") + ".");
  notes(s, "Commercial glasses such as OrCam work offline but not in Bangla and not with Taka. Glass-based research systems send pictures to a server. Raspberry Pi reading aids are cheap but mostly English. No device combines Bangla, Taka and offline use, and counterfeit studies do not check for copies of the same print in training and test.");

  // ================================================================ 8. architecture
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Solution" });
  s.addText("How the glass works", { placeholder: "title" });
  const stages = [
    ["cam", HEX.accent1, "Camera, buttons, voice", ["Camera 640×480", "5 buttons on the frame", "Voice: \"glass, …\""]],
    ["chip", HEX.dk2, "Raspberry Pi 5", ["Mode manager", "All models on the CPU", "Offline, except the online mode"]],
    ["cube", HEX.accent3, "AI modules", ["Text: CRAFT + EasyOCR " + cite("craft", "easyocr"), "Objects: YOLOv8s, COCO " + cite("yolo", "coco"),
      "Taka: YOLOv8s + safe checks", "Emotion: YuNet + EfficientNet " + cite("rafdb"), "Online: cloud model (optional)"]],
    ["speak", HEX.accent1, "Audio + vibration", ["Bangla or English speech " + cite("piper", "espeak"), "Vibration for a note"]],
  ];
  stages.forEach(([k, col, h, lines], i) => {
    const x = 0.6 + i * 3.15;
    card(s, x, 1.45, 2.75, 4.2, h + " card");
    circleIcon(s, k, x + 0.78, 1.7, 1.2, col, h);
    s.addText(h, { x: x + 0.1, y: 3.05, w: 2.55, h: 0.8, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text2, align: "center", valign: "top", isTextBox: true, margin: 0, objectName: h + " heading" });
    s.addText(lines.map((t, j) => ({ text: t, options: { breakLine: j < lines.length - 1 } })),
      { x: x + 0.15, y: 3.9, w: 2.45, h: 2.35, fontSize: i === 2 ? 12 : 14, color: C.text1, align: "center", valign: "top", paraSpaceAfter: 6, isTextBox: true, margin: 0, objectName: h + " details" });
    if (i < 3) s.addShape(pres.shapes.CHEVRON, { x: x + 2.83, y: 2.05, w: 0.28, h: 0.55, fill: { color: HEX.accent6 }, line: { color: HEX.accent6 }, objectName: "Arrow " + (i + 1) });
  });
  s.addText("Everything runs on the glass itself; only the optional online mode uses the internet.", { x: 0.6, y: 5.85, w: 12.1, h: 0.45, fontSize: 18, bold: true, color: C.accent1, align: "center", isTextBox: true, margin: 0, objectName: "Pipeline takeaway" });
  sources(s, `Sources: ${cite("craft")} Baek et al. 2019; ${cite("easyocr")} EasyOCR; ${cite("yolo")} Ultralytics YOLOv8; ${cite("coco")} COCO; ${cite("rafdb")} RAF-DB; ${cite("piper")} Piper; ${cite("espeak")} eSpeak NG.`);
  notes(s, "The camera picture, the buttons and voice commands go to a mode manager on the Raspberry Pi 5. The active mode runs its model on the CPU, and the result is spoken in Bangla or English, with a vibration for a detected note.");

  // ================================================================ 9. prototype evolution
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Solution" });
  s.addText("From a desk prototype to a wearable glass", { placeholder: "title" });
  photo(s, AS + "test_10.jpg", 0.6, 1.45, 2.84, 4.62, "Prototype version 1");
  s.addText([{ text: "Version 1", options: { bold: true, fontSize: 16, color: HEX.dk2, breakLine: true } }, { text: "Board on one temple", options: { fontSize: 13, color: HEX.accent5 } }],
    { x: 0.6, y: 6.12, w: 2.84, h: 0.6, isTextBox: true, margin: 0, valign: "top", objectName: "V1 caption" });
  const v2 = [["1_20.jpg", "Front", "Camera + UV lamp"], ["v2_side_portrait.jpg", "Side", "5 buttons, bone-conduction earphone"], ["1_23.jpg", "Rear", "Pi 5 in a ventilated case"]];
  v2.forEach(([f, h, cap], i) => {
    photo(s, AS + f, 3.69 + i * 3.09, 1.45, 2.84, 4.62, h + ": " + cap);
    s.addText([{ text: h, options: { bold: true, fontSize: 16, color: HEX.dk2, breakLine: true } }, { text: cap, options: { fontSize: 13, color: HEX.accent5 } }],
      { x: 3.69 + i * 3.09, y: 6.12, w: 2.84, h: 0.6, isTextBox: true, margin: 0, valign: "top", objectName: "V2 caption " + (i + 1) });
  });
  s.addText("Version 2", { shape: pres.shapes.ROUNDED_RECTANGLE, x: 3.84, y: 1.6, w: 1.2, h: 0.4, rectRadius: 0.08, fill: { color: HEX.accent1 }, line: { color: HEX.accent1 },
    color: "FFFFFF", fontSize: 13, bold: true, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "V2 tag" });
  notes(s, "The first version put the board and fan on one temple: heavy on one side and loud by the ear. Version two moves the Raspberry Pi into a ventilated enclosure at the back of the head, keeps the camera and a small UV lamp on the front, puts five buttons along the frame and uses a bone-conduction earphone so the ears stay free for traffic sounds.");

  // ================================================================ 10. Taka detection, synthetic data
  pres.addSection({ title: "Currency" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Currency" });
  s.addText("Taka detection: 22,333 training images, no hand labels", { placeholder: "title" });
  s.addImage({ path: FIG + "fig_compositing.jpg", x: 0.6, y: 1.5, w: 8.2, h: 8.2 * 466 / 1600, objectName: "Compositing pipeline" });
  s.addText("Note photos " + cite("banglataka") + " are pasted onto real scenes " + cite("coco") + ", and the program writes every label itself.",
    { x: 0.6, y: 4.2, w: 8.2, h: 0.9, fontSize: 20, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "Compositing text" });
  card(s, 9.2, 1.5, 3.5, 2.3, "Synthetic card");
  tag(s, "SYNTHETIC COMPOSITES", 9.45, 1.7, 2.6, HEX.accent6, HEX.dk2, "Synthetic tag");
  stat(s, 9.45, 2.2, 3.0, "0.995", "mAP@0.5 on 2,282 held-out composites", "mAP");
  card(s, 9.2, 4.0, 3.5, 2.4, "Real card", C.text2);
  tag(s, "REAL PHOTOS", 9.45, 4.2, 1.7, HEX.accent4, HEX.dk1, "Real tag");
  s.addText("94.1%", { x: 9.45, y: 4.7, w: 3.0, h: 0.75, fontFace: "Cambria", fontSize: 40, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "Real value" });
  s.addText("right note value on 1,536 photos taken by other people", { x: 9.45, y: 5.45, w: 3.0, h: 0.8, fontSize: 14, color: "E4EFEA", isTextBox: true, margin: 0, valign: "top", objectName: "Real label" });
  sources(s, `Sources: ${cite("banglataka")} BanglaTaka (Mendeley); ${cite("coco")} Lin et al., COCO, ECCV 2014; detector ${cite("yolo")} Ultralytics YOLOv8. 94.1% with the hand-held detector (next slide).`);
  notes(s, "Public Taka images are clean close-ups, so a model trained on them fails on a live camera. We pasted 5,073 note photos onto real COCO pictures with random angle, scale, light and blur, and let the program write the box labels. The detector scores 0.995 mAP on held-out composites and, the number that matters for users, names 94.1 percent of 1,536 real photographs taken by other people correctly. The two numbers measure different things: the first is on synthetic composites, the second on real photos.");

  // ================================================================ 11. hand-held notes
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Currency" });
  s.addText("Notes held in a hand: from 1 in 8 to 1 in 2", { placeholder: "title" });
  tag(s, "TRAINING: 6,000 SYNTHETIC CLOSE-UPS", 0.6, 1.45, 3.9, HEX.accent6, HEX.dk2, "Training tag");
  s.addImage({ path: REP + "fig58_handheld_closeups.png", x: 0.6, y: 1.9, w: 5.5, h: 2.2, objectName: "Synthetic close-ups" });
  tag(s, "TEST: REAL HAND-HELD PHOTOS", 0.6, 4.3, 3.3, HEX.accent4, HEX.dk1, "Test tag");
  s.addText("11.8% → 47.5%", { x: 0.6, y: 4.75, w: 7.0, h: 0.85, fontFace: "Cambria", fontSize: 44, bold: true, color: C.accent1, isTextBox: true, margin: 0, objectName: "Hand-held takeaway" });
  s.addChart(pres.charts.BAR, [
    { name: "Earlier detector", labels: ["Laptop", "Raspberry Pi (8-bit)"], values: [11.8, 8.7] },
    { name: "Hand-held detector", labels: ["Laptop", "Raspberry Pi (8-bit)"], values: [47.5, 33.5] },
  ], { x: 7.9, y: 1.45, w: 4.8, h: 4.4, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent5, HEX.accent1], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.0\"%\"", valAxisMaxVal: 60, valAxisMinVal: 0, valAxisLabelFormatCode: "0\"%\"", showLegend: true, legendPos: "b",
    showTitle: true, title: "Right note value, real hand-held photos (NSTU, 1,127)", ...chartFrame, objectName: "Hand-held chart" });
  s.addText("Right note value on the laptop. Trade-off: with 3 notes in view, the total is right 97.0% (was 99.2%).",
    { x: 0.6, y: 5.65, w: 7.0, h: 0.8, fontSize: 14, italic: true, color: C.accent2, isTextBox: true, margin: 0, valign: "top", objectName: "Costs" });
  sources(s, `Sources: test photos ${cite("nstu")} Rafi et al., NSTU-BDTAKA, Data in Brief 2024; detector ${cite("yolo")} YOLOv8. Our results: paper_evidence/HANDHELD_FINETUNE.md.`);
  notes(s, "A blind user holds the note close, folded and under the fingers, and our training pictures showed whole notes in scenes. We generated six thousand hand-held close-ups from training notes only and fine-tuned the detector. On the independent NSTU hand-held test, the right value rises from 11.8 to 47.5 percent on the laptop and from 8.7 to 33.5 percent on the Pi. There are costs: one-taka notes are named more often by mistake (36.6 percent), and with three notes in view the total is right 97.0 percent of the time instead of 99.2.");

  // ================================================================ 12. live tests
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Currency" });
  s.addText("Live tests with real notes", { placeholder: "title" });
  s.addText("Informal demonstrations, not a measured accuracy study", { shape: pres.shapes.ROUNDED_RECTANGLE, x: 0.6, y: 1.4, w: 12.1, h: 0.5, rectRadius: 0.08,
    fill: { color: "FFF4D6" }, line: { color: HEX.accent4 }, fontSize: 16, bold: true, color: HEX.dk1, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "Demo banner" });
  ["test_2.jpg", "test_4.jpg", "test_7.jpg", "test_14.jpg", "test_16.jpg", "test_8.jpg"].forEach((f, i) =>
    photo(s, AS + f, 0.6 + (i % 3) * 4.08, 2.05 + Math.floor(i / 3) * 2.25, 3.94, 2.12, "Live test " + (i + 1)));
  s.addText("Six note values (10 to 1,000 Taka) named at 0.63–0.89 confidence, including worn and folded notes. Laptop webcam and the first glass prototype.",
    { x: 0.6, y: 6.55, w: 12.1, h: 0.4, fontSize: 14, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "Live test note" });
  notes(s, "These are live sessions with our own notes, with a laptop webcam and the first glass prototype. Ten, fifty, one hundred, two hundred, five hundred and one thousand Taka were named at 0.63 to 0.89 confidence, including worn and folded notes and two or three notes at once. Boxes are loose and can overlap when notes are close. These photos illustrate behaviour; we did not count an error rate in these sessions.");

  // ================================================================ 13. prefix training
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Currency" });
  s.addText("Counterfeit check: one view is enough with prefix training", { placeholder: "title" });
  s.addChart(pres.charts.LINE, [
    { name: "Fixed six-view training", labels: ["1", "2", "3", "4", "5", "6"], values: [71.2, 73.1, 85.7, 90.5, 90.5, 98.1] },
    { name: "Prefix training (ours)", labels: ["1", "2", "3", "4", "5", "6"], values: [98.2, 98.4, 98.6, 98.6, 98.7, 98.7] },
  ], { x: 0.6, y: 1.45, w: 7.6, h: 5.0, chartColors: [HEX.accent5, HEX.accent1], lineSize: 3, lineDataSymbolSize: 8, valAxisMinVal: 60, valAxisMaxVal: 100,
    valAxisLabelFormatCode: "0\"%\"", showLegend: true, legendPos: "b", showTitle: true, title: "Test accuracy by number of views (mean of 3 seeds)",
    catAxisTitle: "Views captured", showCatAxisTitle: true, catAxisTitleFontSize: 12, catAxisTitleColor: HEX.accent5, ...chartFrame, objectName: "Prefix chart" });
  card(s, 8.6, 1.45, 4.1, 4.95, "Takeaway card", C.text2);
  s.addText("With a single view", { x: 8.9, y: 1.75, w: 3.5, h: 0.45, fontSize: 18, color: "CFE2DA", isTextBox: true, margin: 0, objectName: "One view label" });
  s.addText("71.2%", { x: 8.9, y: 2.25, w: 3.5, h: 0.7, fontFace: "Cambria", fontSize: 36, bold: true, color: "B9C4BF", isTextBox: true, margin: 0, objectName: "Before value" });
  s.addText("rises to", { x: 8.9, y: 2.95, w: 3.5, h: 0.45, fontSize: 18, italic: true, color: "CFE2DA", isTextBox: true, margin: 0, objectName: "Rises to" });
  s.addText("98.2%", { x: 8.9, y: 3.4, w: 3.5, h: 0.95, fontFace: "Cambria", fontSize: 54, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "After value" });
  s.addText("Held-out JaalTaka test notes, never used for training or for any choice", { x: 8.9, y: 5.55, w: 3.5, h: 0.7, fontSize: 12, italic: true, color: "CFE2DA", isTextBox: true, margin: 0, valign: "top", objectName: "Split note" });
  s.addText("Training on random first views: one view is enough.",
    { x: 8.9, y: 4.5, w: 3.5, h: 1.0, fontSize: 18, color: "E4EFEA", isTextBox: true, margin: 0, valign: "top", objectName: "Prefix takeaway" });
  sources(s, `Data: ${cite("jaal", "jaal26")} Piyas et al., JaalTaka dataset and article, 2025-2026 (1,390 notes, six views each; counterfeits seized by the Rapid Action Battalion).`);
  notes(s, "A blind user cannot promise to capture six views of a note. A model trained only with all six views collapses to 71 percent with one view. Training on random prefixes of the views, that is the first k views for a random k, keeps it at about 98 percent from one to six views, with no loss at six views and stable results across three seeds.");

  // ================================================================ 14. unseen prints
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Currency" });
  s.addText("Unseen counterfeit prints: a stricter test", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [{ name: "6 views", labels: ["Frozen ResNet-50", "Prefix network", "Fine-tuned ResNet-50", "Prefix + watermark"], values: [83.8, 89.3, 94.4, 95.0] }],
    { x: 0.6, y: 1.45, w: 7.6, h: 5.0, barDir: "bar", chartColors: [HEX.accent5, HEX.accent3, HEX.accent5, HEX.accent1], showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: "0.0\"%\"", valAxisMinVal: 70, valAxisMaxVal: 100, valAxisLabelFormatCode: "0\"%\"", showLegend: false, showTitle: true,
      title: "Accuracy on counterfeit prints never seen in training (6 views, 3 seeds)", ...chartFrame, objectName: "Unseen prints chart" });
  card(s, 8.6, 1.45, 4.1, 2.55, "Main result", C.text2);
  s.addText("95.0%", { x: 8.9, y: 1.65, w: 3.5, h: 1.0, fontFace: "Cambria", fontSize: 54, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "Unseen value" });
  s.addText("on counterfeit prints never seen in training, with the watermark check", { x: 8.9, y: 2.7, w: 3.5, h: 1.1, fontSize: 16, color: "E4EFEA", isTextBox: true, margin: 0, valign: "top", objectName: "Unseen label" });
  s.addText([{ text: "Why a stricter split: ", options: { bold: true } }, { text: "279 of 322 readable 500 Taka counterfeits share one serial number, so the usual split tests copies of known prints." }],
    { x: 8.6, y: 4.25, w: 4.1, h: 2.1, fontSize: 15, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "Unseen text" });
  sources(s, `Data: ${cite("jaal")} JaalTaka; serial-number audit and print-disjoint split are ours. Earlier Taka counterfeit work: ${cite("uddin")} SVM on security features; ${cite("pathak")} CNNs; ${cite("uvink")} UV flecks + RPN.`);
  notes(s, "Most counterfeit notes in JaalTaka are copies of a few prints, so the usual test measures recognition of known reprints; a serial-number lookup alone scores 90.2 percent. On truly unseen prints our network drops to about 89 percent; adding the watermark check brings it to 95 percent.");

  // ================================================================ 15. DMWR
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Currency" });
  s.addText("DMWR: a tiny, physics-based watermark checker", { placeholder: "title" });
  s.addImage({ path: FIG + "fig_dmwr_steps.png", x: 0.6, y: 1.45, w: 7.6, h: 7.6 * 960 / 1800, objectName: "DMWR steps" });
  s.addText([
    { text: "1. Remove the lamp:  Q = I / (G∗I)", options: { breakLine: true } },
    { text: "2. Genuine portrait:  T = median of Q over genuine notes", options: { breakLine: true } },
    { text: "3. Residual R = Q − T → two small CNN streams → \"watermark clear?\"" },
  ], { x: 0.6, y: 5.6, w: 7.6, h: 0.95, fontSize: 14, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "DMWR steps text" });
  card(s, 8.6, 1.45, 4.1, 5.0, "DMWR stats");
  s.addText("91.5%", { x: 8.9, y: 1.7, w: 3.6, h: 1.0, fontFace: "Cambria", fontSize: 54, bold: true, color: C.accent1, isTextBox: true, margin: 0, objectName: "DMWR value" });
  s.addText("on unseen counterfeit prints (3 seeds, ±0.3)", { x: 8.9, y: 2.7, w: 3.6, h: 0.7, fontSize: 16, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "DMWR value label" });
  stat(s, 8.9, 3.65, 3.6, "46× smaller", "than MobileNetV2 " + cite("mnv2") + " (47,818 parameters, 0.21 MB), and as accurate: 92.9%, p ≥ 0.375", "DMWR size");
  sources(s, `Baseline ${cite("mnv2")} Sandler et al., MobileNetV2, CVPR 2018; data ${cite("jaal")} JaalTaka. DMWR (denomination-matched watermark residual) is our method.`);
  notes(s, "Our watermark method divides the back-lit photo by a blurred copy to remove the lamp, subtracts a genuine portrait of the note value and classifies the residual with two tiny networks. With under fifty thousand parameters it reaches 91.5 percent on unseen prints, statistically level with a MobileNetV2 46 times larger.");

  // ================================================================ 16. safe verdict
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Currency" });
  s.addText("Safe before decisive: the glass never says \"counterfeit\"", { placeholder: "title" });
  circleIcon(s, "cash", 0.8, 2.45, 1.2, HEX.dk2, "Note");
  s.addText("Note captured", { x: 0.45, y: 3.75, w: 1.9, h: 0.4, fontSize: 14, align: "center", color: C.text1, isTextBox: true, margin: 0, objectName: "Note label" });
  s.addText("Score above the safe threshold?", { shape: pres.shapes.DIAMOND, x: 2.9, y: 1.95, w: 2.8, h: 2.2, fill: { color: HEX.lt2 }, line: { color: HEX.accent1, width: 1.5 },
    fontSize: 15, bold: true, color: HEX.dk2, align: "center", valign: "middle", isTextBox: true, margin: 0.2, objectName: "Decision" });
  connect(s, 2.0, 3.05, 2.9, 3.05, "Arrow in", true);
  s.addText("\"Likely genuine\"", { shape: pres.shapes.ROUNDED_RECTANGLE, x: 6.6, y: 1.6, w: 3.0, h: 0.9, rectRadius: 0.1, fill: { color: HEX.accent1 }, line: { color: HEX.accent1 },
    color: "FFFFFF", fontSize: 18, bold: true, align: "center", valign: "middle", isTextBox: true, objectName: "Genuine box" });
  s.addText("\"Check by hand\" + guided watermark check", { shape: pres.shapes.ROUNDED_RECTANGLE, x: 6.6, y: 3.6, w: 3.0, h: 1.0, rectRadius: 0.1, fill: { color: HEX.accent4 }, line: { color: HEX.accent4 },
    color: HEX.dk1, fontSize: 16, bold: true, align: "center", valign: "middle", isTextBox: true, objectName: "Check box" });
  connect(s, 5.7, 3.05, 6.6, 2.05, "Arrow yes", true);
  connect(s, 5.7, 3.05, 6.6, 4.1, "Arrow no", true);
  s.addText("yes", { x: 5.85, y: 2.05, w: 0.6, h: 0.3, fontSize: 12, color: C.accent5, isTextBox: true, margin: 0, objectName: "Yes label" });
  s.addText("no", { x: 5.85, y: 3.75, w: 0.6, h: 0.3, fontSize: 12, color: C.accent5, isTextBox: true, margin: 0, objectName: "No label" });
  card(s, 10.0, 1.45, 2.7, 1.95, "Safety stat");
  stat(s, 10.25, 1.7, 2.3, "0 / 88", "test counterfeits passed as \"likely genuine\"", "Safety");
  s.addText([{ text: "\"Likely genuine\": ", options: { bold: true, color: HEX.accent1 } }, { text: "the score is higher than every counterfeit seen in validation (57.5% of genuine notes).", options: { breakLine: true } },
    { text: "\"Check by hand\": ", options: { bold: true, color: "B07D00" } }, { text: "the glass is not sure. It never means \"counterfeit\"." }],
    { x: 0.6, y: 4.75, w: 12.1, h: 1.0, fontSize: 18, color: C.text1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 6, objectName: "Safety text" });
  s.addText("A screening aid, not a guaranteed counterfeit detector", { shape: pres.shapes.ROUNDED_RECTANGLE, x: 0.6, y: 5.9, w: 12.1, h: 0.5, rectRadius: 0.08,
    fill: { color: "FFF4D6" }, line: { color: HEX.accent4 }, fontSize: 16, bold: true, color: HEX.dk1, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "Screening banner" });
  sources(s, `Test notes: ${cite("jaal")} JaalTaka.`);
  notes(s, "Because a blind user cannot verify the answer, the glass never says a note is counterfeit. It says likely genuine only above a threshold set from validation counterfeits; otherwise it asks for a check by hand. The threshold is the highest score of any validation counterfeit, so a new counterfeit passes with probability about 1 in n plus 1, roughly 1.1 percent. No test counterfeit passed, and 57.5 percent of genuine notes are confirmed from close-up views.");

  // ================================================================ 17. UV fibres
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Currency" });
  s.addText("UV fibre check: the glass carries its own UV lamp", { placeholder: "title" });
  s.addImage({ path: FIG + "fig_uv_fibres.jpg", x: 0.6, y: 1.45, w: 12.1, h: 12.1 * 520 / 1816, objectName: "UV fibres" });
  stat(s, 0.6, 5.1, 3.6, "85.6%", "of 111 real UV photos of genuine notes accepted", "UV genuine");
  stat(s, 4.6, 5.1, 3.6, "88.6%", "of SIMULATED fibre-free sheets rejected (not real counterfeits)", "UV sheets", C.accent5);
  sources(s, `The glass's lamp emits 360-395 nm, covering the 365 nm used for banknote checks; banknote paper stays dark while its fibres glow ${cite("mutanen")}. Prior Taka UV work: ${cite("uvink")} (96.3%). Fibre check and photos are ours.`);
  s.addText([{ text: "Prototype, off by default", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: "The rejection result is simulated; it must be validated on real counterfeit notes", options: { fontSize: 14 } }],
    { shape: pres.shapes.ROUNDED_RECTANGLE, x: 8.6, y: 5.0, w: 4.1, h: 1.4, rectRadius: 0.1, fill: { color: "FBE9EB" }, line: { color: HEX.accent2, width: 1.25 },
      color: HEX.accent2, isTextBox: true, margin: 12, valign: "middle", objectName: "UV caveat" });
  notes(s, "Under a UV lamp, genuine 200, 500 and 1000 Taka notes show glowing security fibres. Our check removes the lamp's own blue light and counts thin glowing marks. It accepts 85.6 percent of real genuine photos and rejects 88.6 percent of simulated fibre-free sheets. It stays off by default until real counterfeits are tested.");

  // ================================================================ 18. text reading
  pres.addSection({ title: "Reading and voice" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Reading and voice" });
  s.addText("Bangla reading: rendered text and real street photos", { placeholder: "title" });
  const ocrOpts = (x, w, title, max, name) => ({ x, y: 1.95, w, h: 4.5, barDir: "col", barGapWidthPct: 50, chartColors: [HEX.accent5, HEX.accent1], showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\"%\"", valAxisMinVal: 0, valAxisMaxVal: max, valAxisLabelFormatCode: "0\"%\"", showLegend: true, legendPos: "b",
    showTitle: false, ...chartFrame, objectName: name });
  s.addText("Rendered text (clean)", { x: 0.6, y: 1.45, w: 3.4, h: 0.45, fontFace: "Cambria", fontSize: 19, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "Rendered heading" });
  s.addChart(pres.charts.BAR, [{ name: "Before", labels: ["All scripts"], values: [9.3] }, { name: "Now", labels: ["All scripts"], values: [2.3] }],
    ocrOpts(0.6, 3.4, "", 12, "Rendered chart"));
  s.addText("Real word photos (hard)", { x: 4.3, y: 1.45, w: 4.3, h: 0.45, fontFace: "Cambria", fontSize: 19, bold: true, color: C.accent2, isTextBox: true, margin: 0, objectName: "Real heading" });
  s.addChart(pres.charts.BAR, [{ name: "Before", labels: ["Bangla", "Latin"], values: [39.2, 62.5] }, { name: "Now", labels: ["Bangla", "Latin"], values: [28.7, 50.6] }],
    ocrOpts(4.3, 4.3, "", 70, "Real chart"));
  s.addText("Character error rate, lower is better. Measured on evaluation datasets, not yet through the glass.", { x: 0.6, y: 6.32, w: 8.0, h: 0.3, fontSize: 12, bold: true, italic: true, color: C.accent5, isTextBox: true, margin: 0, objectName: "Chart note" });
  s.addText("What helped", { x: 8.95, y: 1.45, w: 3.75, h: 0.45, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "Helped heading" });
  s.addText([
    { text: "Reading the main text block only", options: { bullet: true, breakLine: true } },
    { text: "Tuned detector and clean-up", options: { bullet: true, breakLine: true } },
    { text: "Recogniser fine-tuned on real words", options: { bullet: true } },
  ], { x: 8.95, y: 1.95, w: 3.75, h: 2.2, fontSize: 17, color: C.text1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 8, objectName: "Helped list" });
  s.addText("What did not", { x: 8.95, y: 4.3, w: 3.75, h: 0.45, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "Not heading" });
  s.addText("Binarisation, Tesseract, TrOCR, PaddleOCR, dictionaries",
    { x: 8.95, y: 4.8, w: 3.75, h: 1.3, fontSize: 17, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "Not list" });
  sources(s, `Engine ${cite("easyocr")} EasyOCR with ${cite("craft")} CRAFT; real words from ${cite("mlt")} ICDAR 2019 MLT (1,000 test words). Every choice made on validation data.`);
  notes(s, "On our rendered Bangla and English benchmark the error fell from 9.3 to 2.3 percent. Real street words are much harder: a recogniser fine-tuned on real words and blended with the original weights lowered the error from 39.2 to 28.7 percent for Bangla and from 62.5 to 50.6 percent for Latin script. What helped: reading only the main block of text, tuned detector settings with CLAHE and clean-up rules, and the recogniser fine-tuned on real words and blended fifty-fifty with the original weights. Binarisation, multi-scale reading, Tesseract, TrOCR, PaddleOCR, dictionaries, smaller pictures and a rotation retry did not help.");

  // ================================================================ 19. voice
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Reading and voice" });
  s.addText("Voice: a natural Bangla voice and spoken commands", { placeholder: "title" });
  card(s, 0.6, 1.45, 4.0, 4.4, "Speech card");
  circleIcon(s, "speak", 0.9, 1.7, 0.85, HEX.accent1, "Voice");
  s.addText("Speaking", { x: 1.95, y: 1.85, w: 2.5, h: 0.5, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "Speaking heading" });
  s.addText([
    { text: "Natural Piper voice, Bangla and English " + cite("piper", "espeak"), options: { bullet: true, breakLine: true } },
    { text: "No broken sound when the Pi is busy", options: { bullet: true, breakLine: true } },
    { text: "\"পড়ছি\" (reading) while text is captured", options: { bullet: true } },
  ], { x: 0.9, y: 2.8, w: 3.5, h: 2.9, fontSize: 17, color: C.text1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 10, objectName: "Speaking list" });
  s.addText("Command results: synthetic voices on a laptop, not human trials on the glasses", { shape: pres.shapes.ROUNDED_RECTANGLE, x: 0.6, y: 6.0, w: 12.1, h: 0.52, rectRadius: 0.08,
    fill: { color: "FFF4D6" }, line: { color: HEX.accent4 }, fontSize: 16, bold: true, color: HEX.dk1, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "Voice banner" });
  s.addChart(pres.charts.BAR, [{ name: "Commands carried out", labels: ["Clean", "Whispered", "Street noise 0 dB", "People talking 10 dB"], values: [100, 98.9, 82.5, 54.3] }],
    { x: 4.9, y: 1.45, w: 5.0, h: 4.4, barDir: "col", chartColors: [HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\"%\"",
      valAxisMinVal: 0, valAxisMaxVal: 110, valAxisLabelFormatCode: "0\"%\"", showLegend: false, showTitle: true,
      title: "\"Glass, …\" commands, 12 unseen synthetic voices", ...chartFrame, objectName: "Voice chart" });
  stat(s, 10.2, 1.6, 2.5, "212 → 13", "false commands in 5,040 ordinary sentences, thanks to the wake word", "False commands");
  stat(s, 10.2, 3.55, 2.5, "0.86 s", "from the end of speech to the action (median)", "Response");
  sources(s, `Recogniser ${cite("vosk")} Vosk; noise from ${cite("esc50")} Piczak, ESC-50; voices ${cite("piper", "espeak")}. Results: paper_evidence/VOICE_COMMANDS_FAST.md.`);
  notes(s, "The glass now speaks with a natural Piper voice, with eSpeak NG as a fallback, and each sentence is rendered before it plays, which fixed the broken sound on the Pi; a short word and a soft tick tell the user that a long capture is running. These command results come from synthetic voices on a laptop, not from people wearing the glass. Commands start with the word glass. On twelve unseen synthetic voices, clean commands are always carried out, whispered ones 98.9 percent, with street noise as loud as the voice 82.5 percent, and with people talking nearby about half. The wake word cut false commands from 212 to 13.");

  // ================================================================ 20. Raspberry Pi 5
  pres.addSection({ title: "Device" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Device" });
  s.addText("Running on the Raspberry Pi 5", { placeholder: "title" });
  const pis = [["399 ms", "currency check, end to end"], ["64 ms", "watermark check"], ["562 ms", "object scan (YOLOv8s)"], ["680 ms", "emotion, face in view"],
    ["18.3 s", "3-line sign, text found once"], ["1.48 GB", "peak memory, 30-minute run"]];
  pis.forEach(([v, l], i) => {
    const x = 0.6 + (i % 3) * 2.75, y = 1.45 + Math.floor(i / 3) * 2.45;
    card(s, x, y, 2.55, 2.2, "Pi stat " + (i + 1));
    stat(s, x + 0.25, y + 0.3, 2.1, v, l, "Pi " + (i + 1), i === 4 ? C.accent2 : C.accent1);
  });
  card(s, 8.95, 1.45, 3.75, 4.65, "Heat card", C.text2);
  circleIcon(s, "therm", 9.25, 1.7, 0.8, HEX.accent2, "Heat");
  s.addText("Key finding: heat", { x: 10.2, y: 1.85, w: 2.4, h: 0.5, fontFace: "Cambria", fontSize: 19, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "Heat heading" });
  s.addText("88.7 °C", { x: 9.25, y: 2.65, w: 3.3, h: 0.8, fontFace: "Cambria", fontSize: 44, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "Heat peak" });
  s.addText("peak temperature; the Pi throttled in 76% of samples (idle 71 °C)", { x: 9.25, y: 3.45, w: 3.25, h: 0.9, fontSize: 15, color: "E4EFEA", isTextBox: true, margin: 0, valign: "top", objectName: "Heat label" });
  s.addText("Response: a thermal governor slows background scans when hot; better cooling is future work.",
    { x: 9.25, y: 4.55, w: 3.25, h: 1.3, fontSize: 14, italic: true, color: "CFE2DA", isTextBox: true, margin: 0, valign: "top", objectName: "Heat response" });
  s.addText("Measured on a Raspberry Pi 5 (8 GB) on 13 October 2026, board at its temperature limit; memory from the 1 October run. Changes after 13 October are not yet timed on the Pi.",
    { x: 0.6, y: 6.25, w: 12.1, h: 0.4, fontSize: 12, italic: true, color: C.accent5, isTextBox: true, margin: 0, objectName: "Pi caveat" });
  notes(s, "Measured on the Raspberry Pi 5 itself. The currency check takes 0.4 seconds, the watermark check 64 milliseconds. Text reading is the slow part at about 18 seconds for a short sign. The board was hot in this run, so we added a thermal governor; the newest changes still need to be timed on the Pi.");

  // ================================================================ 21. testing the worn glass
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Device" });
  s.addText("Testing the worn glass: where the project stands", { placeholder: "title" });
  [["1_5.jpg", 0.6], ["1_18.jpg", 3.15], ["1_1.jpg", 5.7]].forEach(([f, x], i) => photo(s, AS + f, x, 1.45, 2.4, 4.27, "Worn glass " + (i + 1)));
  s.addText("Prototype version 2 on team members: fit, weight and button reach are checked before user trials.",
    { x: 0.6, y: 5.85, w: 7.5, h: 0.7, fontSize: 14, italic: true, color: C.accent5, isTextBox: true, margin: 0, valign: "top", objectName: "Worn caption" });
  const status = [
    ["DONE", HEX.accent1, "FFFFFF", HEX.lt2, "Engineering tests", "Each part tested on held-out data, on a laptop and the Pi 5"],
    ["IN PROGRESS", HEX.accent4, HEX.dk1, "FFF8E6", "Participant trials", "Sessions with a visually impaired participant; no user study complete yet"],
    ["LIMITATION", HEX.accent2, "FFFFFF", "FBE9EB", "Glass camera", "First field test: currency unreliable, Bangla text not read (640×480, fixed focus)"],
  ];
  status.forEach(([t, col, tc, fill, h, body], i) => {
    const y = 1.45 + i * 1.68;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.45, y, w: 4.25, h: 1.55, rectRadius: 0.1, fill: { color: fill }, line: { color: col, width: 1.25 }, objectName: h + " card" });
    tag(s, t, 8.65, y + 0.15, 1.45, col, tc, h + " status");
    s.addText(h, { x: 10.2, y: y + 0.12, w: 2.4, h: 0.42, fontFace: "Cambria", fontSize: 18, bold: true, color: C.text2, isTextBox: true, margin: 0, valign: "middle", objectName: h + " heading" });
    s.addText(body, { x: 8.65, y: y + 0.62, w: 3.9, h: 0.85, fontSize: 14, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: h + " text" });
  });
  notes(s, "The second prototype on team members. The first field test with the glass camera was not good enough: currency was unreliable and Bangla text was not read, probably because of the low-resolution fixed-focus camera. Sessions with a visually impaired participant have started, with a Bangla voice questionnaire, but no user study is complete yet, so we make no claims about users.");

  // ================================================================ 22. next steps
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Device" });
  s.addText("Future work: the roadmap", { placeholder: "title" });
  const nexts = [
    ["cam", "Camera", "Higher resolution or autofocus"],
    ["users", "User study", "8–12 participants, including their own voice commands"],
    ["shield", "More banknotes", "Unseen counterfeit prints; UV check on real fakes"],
    ["therm", "Speed and cooling", "Time the new code on the Pi; better cooling; battery"],
  ];
  s.addShape(pres.shapes.RECTANGLE, { x: 1.2, y: 2.6, w: 10.9, h: 0.08, fill: { color: HEX.accent6 }, line: { color: HEX.accent6 }, objectName: "Roadmap line" });
  nexts.forEach(([k, h, t], i) => {
    const x = 0.6 + i * 3.08;
    circleIcon(s, k, x + 0.79, 1.85, 1.6, i % 2 ? HEX.accent3 : HEX.accent1, h);
    s.addText(String(i + 1), { shape: pres.shapes.OVAL, x: x + 2.05, y: 1.75, w: 0.5, h: 0.5, fill: { color: HEX.dk2 }, line: { color: "FFFFFF", width: 2 },
      color: "FFFFFF", fontSize: 14, bold: true, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: h + " number" });
    s.addText(h, { x, y: 3.7, w: 3.18, h: 0.55, fontFace: "Cambria", fontSize: 24, bold: true, color: C.text2, align: "center", isTextBox: true, margin: 0, objectName: h + " heading" });
    s.addText(t, { x: x + 0.15, y: 4.35, w: 2.88, h: 1.2, fontSize: 17, color: C.text1, align: "center", isTextBox: true, margin: 0, valign: "top", objectName: h + " text" });
  });
  s.addText([{ text: "Done: ", options: { bold: true, color: HEX.accent1 } }, { text: "engineering tests     " }, { text: "In progress: ", options: { bold: true, color: "B07D00" } }, { text: "participant trials     " }, { text: "Future: ", options: { bold: true, color: HEX.dk2 } }, { text: "the four steps above" }],
    { shape: pres.shapes.ROUNDED_RECTANGLE, x: 0.6, y: 5.6, w: 12.1, h: 0.65, rectRadius: 0.1, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, fontSize: 17, color: HEX.dk1, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "Status line" });
  notes(s, "Our roadmap has four steps. First, a better glass camera with higher resolution or autofocus, and a test set taken through the glass. Second, the user study with eight to twelve participants: task time, accuracy, SUS and NASA-TLX, including voice commands spoken by people and Bangla command words. Third, more banknotes: about a hundred unseen counterfeit prints with a bank or the police, and the UV check on real counterfeits. Fourth, speed and cooling: timing the new code on the Pi, better cooling and battery life.");

  // ================================================================ 23. conclusion
  pres.addSection({ title: "Closing" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Closing" });
  s.addText("Conclusion", { placeholder: "title" });
  photo(s, AS + "v2_collage.jpg", 0.6, 1.4, 4.1, 5.2, "Second prototype");
  const concl = [
    ["glasses", "Built", "An offline glass that speaks Bangla, reads, names objects and Taka on a Raspberry Pi 5"],
    ["shield", "New methods", "Prefix training, a safe verdict, the DMWR watermark check, hand-held fine-tuning"],
    ["users", "Next", "A better camera and the user trials"],
  ];
  concl.forEach(([k, h, t], i) => {
    const y = 1.4 + i * 1.78;
    card(s, 5.1, y, 7.6, 1.6, h + " card");
    circleIcon(s, k, 5.35, y + 0.4, 0.8, i === 2 ? HEX.accent3 : HEX.accent1, h);
    s.addText(h, { x: 6.4, y: y + 0.2, w: 6.1, h: 0.5, fontFace: "Cambria", fontSize: 22, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: h + " heading" });
    s.addText(t, { x: 6.4, y: y + 0.75, w: 6.1, h: 0.75, fontSize: 17, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: h + " text" });
  });
  notes(s, "To sum up: SARA turns three everyday moments, money, reading and the street, into questions the user can ask the glass. It is an assistive glass that speaks Bangla and runs on a Raspberry Pi 5, tested part by part on held-out data, with new methods for counterfeit checking and hand-held notes. Next come a better camera and the user trials.");

  // ================================================================ 24. thank you
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Closing" });
  photo(s, AS + "1_20.jpg", 8.73, 0, 4.6, 7.5, "Prototype worn");
  s.addText("Thank you", { x: 0.75, y: 2.4, w: 7.5, h: 1.2, fontFace: "Cambria", fontSize: 54, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "Thanks" });
  s.addText("Questions are welcome", { x: 0.75, y: 3.65, w: 7.5, h: 0.6, fontSize: 24, color: "CFE2DA", isTextBox: true, margin: 0, objectName: "Questions" });
  s.addText("SARA  ·  Smart Assistive Reading Aid", { x: 0.75, y: 6.4, w: 7.5, h: 0.4, fontSize: 14, color: "9FC2B4", isTextBox: true, margin: 0, objectName: "Closing name" });
  notes(s, "Thank you; we are happy to take questions.");

  // ================================================================ 25. references
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Closing" });
  s.addText("References", { placeholder: "title" });
  const half = Math.ceil(REFS.length / 2);
  [REFS.slice(0, half), REFS.slice(half)].forEach((list, col) => {
    s.addText(list.map((r, i) => ({ text: `[${col * half + i + 1}]  ${r[1]}`, options: { breakLine: i < list.length - 1 } })),
      { x: 0.6 + col * 6.15, y: 1.4, w: 5.95, h: 5.25, fontSize: 11, color: C.text1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 5, objectName: "References " + (col + 1) });
  });
  notes(s, "The sources cited on the slides, numbered as in the slides.");

  // ================================================================ 26. links
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Closing" });
  s.addText("Links", { placeholder: "title" });
  const groups = [
    ["git", "Our code and results", [["SARA: glass software for the Raspberry Pi 5", "https://github.com/aflam330/SARA"],
      ["Research repository: experiments, results, thesis", "https://github.com/aflam330/Senior_project_wearable_glass_for_blind_people_using_ocr_and_ylo"]]],
    ["data", "Datasets", [["JaalTaka (counterfeit notes)", REFS[NUM.jaal - 1][2]], ["BanglaTaka (note photos)", REFS[NUM.banglataka - 1][2]],
      ["NSTU-BDTAKA (hand-held notes)", REFS[NUM.nstu - 1][2]], ["Microsoft COCO", REFS[NUM.coco - 1][2]], ["ICDAR 2019 MLT", REFS[NUM.mlt - 1][2]],
      ["ESC-50 environmental sounds", REFS[NUM.esc50 - 1][2]]]],
    ["tools", "Open-source tools", [["Ultralytics YOLOv8", REFS[NUM.yolo - 1][2]], ["EasyOCR", REFS[NUM.easyocr - 1][2]], ["Piper text-to-speech", REFS[NUM.piper - 1][2]],
      ["Vosk speech recognition", REFS[NUM.vosk - 1][2]], ["eSpeak NG", REFS[NUM.espeak - 1][2]], ["WHO vision fact sheet", REFS[NUM.who - 1][2]]]],
  ];
  groups.forEach(([k, h, items], g) => {
    const x = 0.6 + g * 4.1;
    card(s, x, 1.45, 3.85, 5.1, h + " card");
    s.addImage({ data: I[k], x: x + 0.25, y: 1.65, w: 0.5, h: 0.5, objectName: h + " icon" });
    s.addText(h, { x: x + 0.9, y: 1.68, w: 2.8, h: 0.45, fontFace: "Cambria", fontSize: 18, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: h + " heading" });
    const runs = [];
    items.forEach(([t, u], i) => {
      runs.push({ text: t, options: { hyperlink: { url: u, tooltip: u }, color: HEX.accent1, bold: true, fontSize: 13, breakLine: true } });
      runs.push({ text: u.replace(/^https?:\/\//, ""), options: { color: HEX.accent5, fontSize: 10, breakLine: i < items.length - 1, paraSpaceAfter: 7 } });
    });
    s.addText(runs, { x: x + 0.25, y: 2.35, w: 3.4, h: 4.05, isTextBox: true, margin: 0, valign: "top", fit: "none", objectName: h + " links" });
  });
  notes(s, "Our code is public on GitHub: the glass software and the research repository with every experiment. The datasets and open-source tools we used are linked here.");

  await pres.writeFile({ fileName: OUT });
  if (!process.env.NOTHEME) await applyTheme(OUT, THEME);
  console.log("written", OUT);
})();
