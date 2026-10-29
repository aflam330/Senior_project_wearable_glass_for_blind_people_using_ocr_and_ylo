// SARA presentation, fourth design (29 October 2026): the look and section names of the reference deck
// (thesis report/slides/demo slide.pptx). Built from build3.js; p_b4_style.py made the changes.
// node build4.js   -> ../../thesis report/slides/SARA_Presentation.pptx   (then add_transitions.py adds fades)
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
  headFontFace: "Arial",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2430", lt1: "FFFFFF", dk2: "17324D", lt2: "EAF1F7",
    accent1: "1F5F8B", accent2: "D7263D", accent3: "4A87B5", accent4: "E6A817", accent5: "5B6777", accent6: "BFD4E6",
    hlink: "1F5F8B", folHlink: "4A87B5",
  },
};
const HEX = THEME.colors;
const BN = "Nirmala UI";          // Bangla text
const TITLE = "A Wearable Assistive Glass for Authenticating Bangladeshi Banknotes: Fusing Physical Security Features and Deep Learning for Visually Impaired Users";
const SUBTITLE = "A bilingual (Bangla and English) smart glass that also reads printed Bangla and English text and detects everyday objects, offline on a Raspberry Pi 5";
const BG = path.resolve(__dirname, "assets", "bg_slide.jpg");   // background of the reference deck

// ------------------------------------------------------------------ icons
async function icon(name, color, size = 256) {
  const svg = RDS.renderToStaticMarkup(React.createElement(fa[name], { color: "#" + color, size: String(size) }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

// ------------------------------------------------------------------ references (numbered as cited)
const REFS = [
  ["who", "World Health Organization. Blindness and vision impairment: fact sheet, updated 10 February 2026.", "https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment"],
  ["orcam", "OrCam Technologies. OrCam MyEye.", "https://www.orcam.com"],
  ["glass", "Ali A., H. et al. A Google Glass based real-time scene analysis for the visually impaired. IEEE Access 9, 2021.", "https://ieeexplore.ieee.org/abstract/document/9648186"],
  ["airis", "Brilli, D. D. et al. AIris: an AI-powered wearable assistive device for the visually impaired. arXiv:2405.07606, 2024.", "https://arxiv.org/abs/2405.07606"],
  ["coco", "Lin, T.-Y. et al. Microsoft COCO: common objects in context. ECCV, 2014.", "https://cocodataset.org"],
  ["banglataka", "Nuhash, M. N. I., Akter, S., Mojumdar, M. U. A diverse image dataset for Bangladeshi currency recognition (BanglaTaka). Mendeley Data V1, 2024.", "https://data.mendeley.com/datasets/3cv2sypkkh/1"],
  ["nstu", "Rafi, M. J. A., Rony, M., Majadi, N. NSTU-BDTAKA: an open dataset for Bangladeshi paper currency detection and recognition. Data in Brief 55, 2024.", "https://www.sciencedirect.com/science/article/pii/S2352340924006681"],
  ["jaal", "Piyas, B. R. C. et al. JaalTaka: a benchmark dataset for detecting counterfeit Bangladeshi banknotes. Mendeley Data V2, 2025.", "https://data.mendeley.com/datasets/2m7wk5cy4c/2"],
  ["yolo", "Jocher, G., Qiu, J., Chaurasia, A. Ultralytics YOLO, version 8.0.0, 2023.", "https://github.com/ultralytics/ultralytics"],
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
  ["rpiaid", "Adusumilli, V. et al. Reading aid and translator with Raspberry Pi for blind people. ICACCS, 2023, pp. 327-331.", "https://doi.org/10.1109/ICACCS57279.2023.10113042"],
  ["uddin", "Uddin, M. S., Das, P. P., Roney, M. S. A. Image-based approach for the detection of counterfeit banknotes of Bangladesh. ICIEV, 2016.", "https://doi.org/10.1109/ICIEV.2016.7760162"],
  ["pathak", "Pathak, A. et al. Enhanced counterfeit detection of Bangladesh currency through convolutional neural networks. IJIRCST 12(2), 2024.", "https://doi.org/10.55524/ijircst.2024.12.2.2"],
  ["uvink", "Chowdhury, A. A. et al. Evaluation of deep learning models on UV ink: a fake money detection scheme with RPN. ICCA (ACM), 2022.", "https://doi.org/10.1145/3542954.3542965"],
  ["mutanen", "Mutanen, J., Jaaskelainen, T., Parkkinen, J. P. S. Luminescent security properties of banknotes. IS&T PICS Conference, 2003, pp. 421-424.", "https://www.imaging.org/common/uploaded%20files/pdfs/Papers/2003/PICS-0-287/8558.pdf"],
  ["rpi5", "Raspberry Pi Ltd. Raspberry Pi 5 product brief, 2023.", "https://www.raspberrypi.com/products/raspberry-pi-5/"],
];
// numbers are given in the order the slides first cite a source (the slides are built in order)
const NUM = {};
const URL = Object.fromEntries(REFS.map((r) => [r[0], r[2]]));
const cite = (...keys) => "[" + keys.map((k) => {
  if (!(k in URL)) throw new Error("unknown reference " + k);
  if (!NUM[k]) NUM[k] = Object.keys(NUM).length + 1;
  return NUM[k];
}).sort((a, b) => a - b).join(", ") + "]";

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";              // 13.33 x 7.5 in
  pres.title = TITLE;
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
    background: { path: BG },
    objects: [],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { path: BG },
    margin: [0.5, 0.6, 0.6, 0.6],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.38, w: 12.1, h: 0.85, fontFace: "Arial", fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "center", margin: 0 }, text: "" } },
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
    s.addText(big, { x, y, w, h: 0.75, fontFace: "Arial", fontSize: 36, bold: true, color, isTextBox: true, margin: 0, objectName: name + " value" });
    s.addText(small, { x, y: y + 0.75, w, h: 0.75, fontSize: 15, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: name + " label" });
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
      line: { color: "9DB3C8", width: 1.25, ...(arrow ? { endArrowType: "triangle" } : {}) }, objectName: name });
  const chartFont = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt" };
  const chartFrame = {
    catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.accent5, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    valGridLine: { color: "DCE5EE", size: 0.75 }, catGridLine: { style: "none" }, dataLabelFontSize: 11, dataLabelColor: HEX.dk1,
    legendFontSize: 12, titleFontSize: 13, titleColor: HEX.dk1, ...chartFont,
  };

  // ================================================================ 1. title
  pres.addSection({ title: "Opening" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Opening" });
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
  notes(s, "We present SARA, a smart glass for blind and low-vision people in Bangladesh. It reads Bangla and English text, names objects, recognises Taka notes and checks them carefully, and it works on the device without the internet. The photo is the current prototype.");

  // ================================================================ 2. introduction
  pres.addSection({ title: "Problem" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
  s.addText("Introduction", { placeholder: "title" });
  const B = (text, opts = {}) => ({ text, options: { bullet: { indent: 18 }, breakLine: true, paraSpaceAfter: 8, ...opts } });
  const Q = (head, text) => [
    { text: head + ": ", options: { bullet: { indent: 18 }, indentLevel: 1, bold: true, color: HEX.accent1, paraSpaceAfter: 4 } },
    { text, options: { breakLine: true } },
  ];
  s.addText([
    B("At least 2.2 billion people have a near or distance vision impairment, and its prevalence in low- and middle-income regions is about four times that in high-income regions " + cite("who")),
    B("In Bangladesh, three everyday questions usually require help from another person:", { paraSpaceAfter: 4 }),
    ...Q("Money", "which note was handed back, and is it genuine?"),
    ...Q("Reading", "what does a medicine label, sign or letter say?"),
    ...Q("Surroundings", "what or who is ahead?"),
    B("In each case the blind person depends on an answer they cannot verify", { paraSpaceBefore: 4 }),
    { text: "SARA is an offline, bilingual assistive glass built for this setting", options: { bullet: { indent: 18 }, bold: true, color: HEX.dk2 } },
  ], { x: 0.6, y: 1.45, w: 6.3, h: 5.05, fontSize: 18, color: HEX.dk1, valign: "top", isTextBox: true, margin: 0, objectName: "Introduction points" });
  const cw = 5.6, ch = cw * 793 / 1983;
  ["Money", "Reading", "Surroundings"].forEach((head, i) =>
    s.addText(head, { x: 7.1 + i * cw / 3, y: 1.75, w: cw / 3, h: 0.45, fontFace: "Arial", fontSize: 18, bold: true, color: C.text2, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: head + " label" }));
  s.addImage({ path: AS + "cartoon_full.jpg", x: 7.1, y: 2.25, w: cw, h: ch, objectName: "Without and with SARA" });
  s.addText("Top row: today, an answer from another person.  Bottom row: with SARA, the glass answers. (Illustration)",
    { x: 7.1, y: 2.35 + ch, w: cw, h: 0.6, fontSize: 12, italic: true, color: C.accent5, align: "center", isTextBox: true, margin: 0, valign: "top", objectName: "Illustration caption" });
  notes(s, "At least 2.2 billion people have a vision impairment, and it is about four times as common in low- and middle-income regions. In Bangladesh, three everyday questions usually need another person: which note was handed back and whether it is genuine, what a medicine label, sign or letter says, and what or who is ahead. In each case the blind person depends on an answer they cannot check. SARA is an offline, bilingual glass built for exactly this setting.");

  // flowchart helpers ------------------------------------------------
  const node = (s, runs, x, y, w, h, fill, color, name, fontSize = 14) =>
    s.addText(runs, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x, y, w, h, fill: { color: fill }, line: { color: fill },
      color, fontSize, align: "center", valign: "middle", isTextBox: true, margin: 0.08, objectName: name });
  const hb = (head, body, headSize) => [{ text: head, options: { bold: true, breakLine: true, ...(headSize ? { fontSize: headSize } : {}) } }, { text: body }];
  // an arrow from (x1, y1) to (x2, y2) in any direction; sizes stay positive
  const arrow = (s, x1, y1, x2, y2, name, head = true) =>
    s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.01), h: Math.abs(y2 - y1),
      flipH: x2 < x1, flipV: y2 < y1, line: { color: HEX.accent5, width: 1.75, ...(head ? { endArrowType: "triangle" } : {}) }, objectName: name });

  // picture-flowchart helpers (rich-picture style: a labelled box with a picture, labelled arrows)
  for (const [k, n, col] of [["shopkeeper", "FaUserTie", HEX.dk2], ["pharmacist", "FaUserNurse", HEX.dk2], ["stranger", "FaUser", HEX.dk2], ["qmark", "FaQuestionCircle", "FFFFFF"]])
    I[k] = await icon(n, col);
  const pnode = (s, head, x, y, w, h, name, { img, iconKey, headH = 0.32, fs = 12 } = {}) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: "FFFFFF" }, line: { color: HEX.accent6, width: 1 }, objectName: name + " box" });
    s.addText(head, { shape: pres.shapes.RECTANGLE, x: x + 0.06, y: y + 0.06, w: w - 0.12, h: headH, fill: { color: HEX.lt2 }, line: { color: HEX.accent3, width: 0.75 },
      fontSize: fs, bold: true, color: HEX.dk2, align: "center", valign: "middle", isTextBox: true, margin: 0.02, objectName: name + " label" });
    const iy = y + headH + 0.12, ih = h - headH - 0.18;
    if (img) photo(s, img, x + 0.06, iy, w - 0.12, ih, name + " picture");
    if (iconKey) {
      const d = Math.min(ih, 0.8);
      s.addShape(pres.shapes.OVAL, { x: x + (w - d) / 2, y: iy + (ih - d) / 2, w: d, h: d, fill: { color: HEX.lt2 }, line: { color: HEX.accent6 }, objectName: name + " circle" });
      s.addImage({ data: I[iconKey], x: x + (w - d) / 2 + d * 0.2, y: iy + (ih - d) / 2 + d * 0.2, w: d * 0.6, h: d * 0.6, objectName: name + " icon" });
    }
  };
  const lab = (s, text, x, y, w, h = 0.4, align = "center") =>
    s.addText(text, { x, y, w, h, fontSize: 11, italic: true, color: HEX.dk1, align, valign: "middle", isTextBox: true, margin: 0, objectName: "Arrow label: " + text });

  // ================================================================ 3. problem statement: picture flowchart
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
  s.addText("Problem Statement", { placeholder: "title" });
  pnode(s, "Visually impaired user", 0.6, 1.95, 1.45, 2.3, "User", { img: AS + "flow_user.jpg", headH: 0.5 });
  const scenes = [
    ["Money: at the shop", "cartoon_problem_1.jpg", "receives change", "Shopkeeper", "shopkeeper", "must trust"],
    ["Reading: at the pharmacy", "cartoon_problem_2.jpg", "needs a label read", "Family or pharmacist", "pharmacist", "asks"],
    ["Surroundings: on the street", "cartoon_problem_3.jpg", "walks outside", "A stranger", "stranger", "asks the way"],
  ];
  scenes.forEach(([h, img, a1, who, ik, a2], i) => {
    const y = 1.4 + i * 1.2, my = y + 0.575;
    pnode(s, h, 3.45, y, 2.0, 1.15, h, { img: AS + img, headH: 0.3, fs: 11 });
    pnode(s, who, 6.45, y, 1.6, 1.15, who, { iconKey: ik, headH: 0.3, fs: 11 });
    arrow(s, 2.05, 3.1, 3.45, my, h + " arrow 1");
    lab(s, a1, 2.1, [1.65, 2.72, 4.1][i], 1.35, 0.3, "left");
    arrow(s, 5.45, my, 6.45, my, h + " arrow 2");
    lab(s, a2, 5.45, my - 0.34, 1.0, 0.3);
    arrow(s, 8.05, my, 9.1, 3.175, h + " arrow 3");
  });
  lab(s, "gives an answer", 8.05, 1.45, 1.05, 0.45);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.1, y: 1.4, w: 3.6, h: 3.55, rectRadius: 0.06, fill: { color: HEX.dk2 }, line: { color: HEX.dk2 }, objectName: "Outcome box" });
  circleIcon(s, "qmark", 10.5, 1.55, 0.8, HEX.accent2, "Outcome");
  s.addText([
    { text: "An answer they cannot verify", options: { bold: true, fontSize: 19, breakLine: true, paraSpaceAfter: 8 } },
    { text: "A wrong note or a fake note can go unnoticed", options: { bullet: { indent: 14 }, breakLine: true, paraSpaceAfter: 5 } },
    { text: "A label or letter can be misread", options: { bullet: { indent: 14 }, breakLine: true, paraSpaceAfter: 5 } },
    { text: "Every task waits for someone else", options: { bullet: { indent: 14 } } },
  ], { x: 9.25, y: 2.45, w: 3.35, h: 2.4, color: "FFFFFF", fontSize: 14, valign: "top", isTextBox: true, margin: 0, objectName: "Outcome text" });
  s.addText("Existing aids do not close this gap", { x: 0.6, y: 5.05, w: 8.0, h: 0.3, fontFace: "Arial", fontSize: 15, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "Aids heading" });
  const aids = [
    ["OrCam MyEye " + cite("orcam"), FIG + "dev_orcam.jpg", "Offline, but no Bangla and no Taka"],
    ["Research glasses " + cite("glass", "airis"), FIG + "dev_googleglass.jpg", "Pictures go to a server"],
    ["Pi reading aids " + cite("rpiaid"), FIG + "hw_pi5.jpg", "Mostly English-only"],
    ["Counterfeit studies " + cite("uddin", "pathak", "uvink", "jaal", "jaal26"), FIG + "jaal_real_v1.jpg", "Not on a wearable"],
  ];
  aids.forEach(([h, img, t], i) => {
    const x = 0.6 + i * 2.4;
    pnode(s, h, x, 5.4, 2.25, 1.2, "Aid " + (i + 1), { headH: 0.3, fs: 11 });
    photo(s, img, x + 0.06, 5.78, 0.95, 0.76, "Aid " + (i + 1) + " picture");
    s.addText(t, { x: x + 1.07, y: 5.78, w: 1.12, h: 0.76, fontSize: 11, color: HEX.dk1, valign: "middle", isTextBox: true, margin: 0, objectName: "Aid " + (i + 1) + " text" });
  });
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 10.15, y: 5.75, w: 0.4, h: 0.5, fill: { color: HEX.accent5 }, line: { color: HEX.accent5 }, objectName: "Aids to gap" });
  node(s, hb("The gap", "No reviewed device is offline, speaks Bangla and names Taka notes", 15), 10.65, 5.4, 2.05, 1.2, HEX.accent2, "FFFFFF", "Gap", 12);
  s.addText("Illustration", { x: 11.2, y: 7.02, w: 0.95, h: 0.28, fontSize: 10, italic: true, color: C.accent5, align: "right", isTextBox: true, margin: 0, objectName: "Illustration note" });
  notes(s, "This picture shows the problem. A visually impaired user receives change at a shop, needs a medicine label read at the pharmacy, or walks outside. Each time the answer comes from another person: the shopkeeper they must trust, a family member or pharmacist, or a stranger. So they end up with an answer they cannot verify: a wrong or fake note can go unnoticed, a label can be misread, and every task waits for someone else. Existing aids do not close this gap. OrCam MyEye works offline but does not list Bangla and does not name Taka notes. The research glasses send pictures to a server. Low-cost Raspberry Pi reading aids are mostly English-only. Studies on Bangladeshi counterfeit notes, and the JaalTaka benchmark, have not been brought to a wearable. No reviewed device is offline, speaks Bangla and names Taka notes.");

  // ================================================================ 4. proposed solution: picture flowchart
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
  s.addText("Proposed Solution: SARA", { placeholder: "title" });
  pnode(s, "User wearing SARA (staged)", 0.6, 1.4, 2.3, 1.45, "User", { img: AS + "flow_exchange.jpg" });
  arrow(s, 2.9, 2.2, 4.25, 2.2, "Ask arrow");
  lab(s, "button, or \"Glass, check money\" " + cite("vosk"), 2.9, 1.55, 1.35, 0.6);
  pnode(s, "Glass camera", 4.25, 1.4, 2.1, 1.45, "Camera", { img: FIG + "v2_frame.jpg" });
  arrow(s, 6.35, 2.2, 7.5, 2.2, "Capture arrow");
  lab(s, "captures a picture", 6.35, 1.6, 1.15, 0.55);
  pnode(s, "Raspberry Pi 5 " + cite("rpi5"), 7.5, 1.4, 2.1, 1.45, "Pi 5", { img: FIG + "hw_pi5.jpg" });
  s.addText([{ text: "Offline: ", options: { bold: true, color: HEX.accent1 } }, { text: "text, objects, Taka, emotion", options: { breakLine: true } },
    { text: "Online (optional): ", options: { bold: true, color: HEX.accent5 } }, { text: "scene description only" }],
    { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x: 10.1, y: 1.6, w: 2.6, h: 1.05, fill: { color: HEX.lt2 }, line: { color: HEX.accent6 }, fontSize: 12, color: HEX.dk1, valign: "middle", isTextBox: true, margin: 0.1, objectName: "Offline note" });
  const picModes = [
    ["Taka recognition " + cite("yolo"), FIG + "live_1000_500.jpg"],
    ["Authenticity: watermark + UV", FIG + "fig_uv_fibres.jpg"],
    ["Text reading " + cite("craft", "easyocr"), AS + "cartoon_sara_2.jpg"],
    ["Objects and people " + cite("yolo", "coco"), AS + "cartoon_sara_3.jpg"],
  ];
  const px = (i) => 0.6 + i * 3.233;
  arrow(s, 8.55, 2.85, 8.55, 3.05, "Pi to modes", false);
  lab(s, "processes offline", 8.65, 2.86, 1.5, 0.22, "left");
  arrow(s, px(0) + 1.2, 3.05, px(3) + 1.2, 3.05, "Mode bus", false);
  picModes.forEach(([h, img], i) => {
    pnode(s, h, px(i), 3.25, 2.4, 1.45, "Mode " + (i + 1), { img });
    arrow(s, px(i) + 1.2, 3.05, px(i) + 1.2, 3.25, "Mode " + (i + 1) + " arrow in");
    arrow(s, px(i) + 1.2, 4.7, px(i) + 1.2, 5.05, "Mode " + (i + 1) + " arrow out");
  });
  pnode(s, "Safe verdict (money)", px(0), 5.05, px(1) + 2.4 - px(0), 1.5, "Verdict");
  circleIcon(s, "shield", px(0) + 0.2, 5.5, 0.85, HEX.accent3, "Verdict");
  s.addText("\"Likely genuine\" or \"check by hand whether it is genuine\"; it never says \"counterfeit\"",
    { x: px(0) + 1.2, y: 5.45, w: 4.3, h: 1.0, fontSize: 14, color: HEX.dk1, valign: "middle", isTextBox: true, margin: 0, objectName: "Verdict text" });
  arrow(s, px(1) + 2.4, 5.8, px(2), 5.8, "Verdict to answer");
  pnode(s, "Spoken answer in Bangla or English " + cite("piper") + " + a short vibration", px(2), 5.05, px(3) + 2.4 - px(2), 1.5, "Answer");
  photo(s, FIG + "hw_earphone.jpg", px(2) + 0.06, 5.43, 1.0, 1.06, "Headphone");
  photo(s, AS + "cartoon_sara_1.jpg", px(2) + 1.14, 5.43, 1.94, 1.06, "Hearing the answer");
  s.addText([{ text: "পাঁচশত টাকার নোট", options: { fontFace: BN, fontSize: 16, bold: true, color: HEX.accent1, breakLine: true } },
    { text: "\"500 Taka note\": known without asking anyone", options: { fontSize: 11, color: HEX.dk1 } }],
    { x: px(2) + 3.18, y: 5.43, w: px(3) + 2.4 - px(2) - 3.24, h: 1.06, valign: "middle", isTextBox: true, margin: 0, objectName: "Spoken example" });
  notes(s, "This picture is the same moment with SARA. The user presses a button or says glass, check money. The glass camera captures a picture, and the Raspberry Pi 5 processes it on the device without the internet. Depending on the mode, it recognises the Taka note, checks the watermark and the UV security fibres, reads Bangla or English text, or names objects and people ahead. For money, the safe verdict says likely genuine or check by hand, and never says counterfeit. The answer is spoken in Bangla or English through the bone-conduction headphone with a short vibration, for example five hundred Taka note, so the user knows without asking anyone.");

  // ================================================================ 5. SARA at a glance
  pres.addSection({ title: "Solution" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Solution" });
  s.addText("System Features", { placeholder: "title" });
  s.addImage({ path: AS + "v2_hardware_collage.jpg", x: 0.6, y: 1.45, w: 5.1, h: 5.1 * 1612 / 2412, objectName: "Prototype version 2 hardware" });
  s.addText("Prototype version 2: the complete unit, the UV light on, the Pi 5 in the rear enclosure, buttons and power bank",
    { x: 0.6, y: 5.0, w: 5.1, h: 0.7, fontSize: 12, italic: true, color: C.accent5, isTextBox: true, margin: 0, valign: "top", objectName: "Hardware caption" });
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
    s.addText(h, { x: x + 0.05, y: y + 0.9, w: 2.0, h: 0.4, fontSize: 17, bold: true, color: C.text2, align: "center", isTextBox: true, margin: 0, objectName: h + " heading" });
    s.addText(t, { x: x + 0.05, y: y + 1.3, w: 2.0, h: 0.4, fontSize: 14, color: C.text1, align: "center", isTextBox: true, margin: 0, objectName: h + " text" });
  });
  card(s, 6.0, 5.45, 6.7, 1.05, "Interaction bar", C.text2);
  circleIcon(s, "hand", 6.2, 5.6, 0.75, HEX.accent3, "Buttons");
  circleIcon(s, "mic", 7.05, 5.6, 0.75, HEX.accent2, "Voice");
  s.addText([{ text: "How the user starts a mode: ", options: { bold: true } }, { text: "five buttons on the frame, or a voice command: \"Glass, read text\"" }],
    { x: 8.0, y: 5.5, w: 4.55, h: 0.95, fontSize: 15, color: "FFFFFF", valign: "middle", isTextBox: true, margin: 0, objectName: "Interaction text" });
  sources(s, `Models: text ${cite("craft", "easyocr")} CRAFT + EasyOCR; objects ${cite("yolo", "coco")} YOLOv8 on COCO; emotion ${cite("rafdb")} RAF-DB; voice commands ${cite("vosk")} Vosk.`);
  notes(s, "This is the current prototype: the camera and five buttons sit on the frame, a bone-conduction earphone leaves the ears free, and the Raspberry Pi 5 rides in a ventilated enclosure at the back of the head. Five modes cover text, objects, Taka notes, emotion and an optional online description. Every mode can be started with a button or with a voice command that begins with the word glass.");

  // ================================================================ 6. research questions
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Solution" });
  s.addText("Research Questions", { placeholder: "title" });
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
    s.addText(h, { x: x + 1.0, y: y + 0.3, w: 2.7, h: 0.5, fontFace: "Arial", fontSize: 20, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "RQ" + (i + 1) + " heading" });
    s.addText(t, { x: x + 0.25, y: y + 1.05, w: 3.4, h: 0.95, fontSize: 20, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "RQ" + (i + 1) + " text" });
  });
  notes(s, "Six questions organise the talk. First, whether a multi-function Bangla glass runs on a Raspberry Pi 5 and how fast. Second, whether a Taka detector trained without hand labels works on real, hand-held photos. Third, why multi-view counterfeit models fail with fewer views and whether training fixes it. Fourth, how much accuracy drops on counterfeit prints never seen in training. Fifth, whether the glass can give counterfeit advice that never accuses a genuine note. Sixth, which part of a Bangla OCR pipeline matters most for a wearable camera.");

  // ================================================================ 7. existing devices
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Solution" });
  s.addText("Literature Review: Existing Assistive Devices", { placeholder: "title" });
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
  s.addText("The research gap", { x: 9.2, y: 1.7, w: 3.3, h: 0.5, fontFace: "Arial", fontSize: 24, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "Gap heading" });
  s.addText("None of the systems we reviewed combines Bangla, Taka and offline use.",
    { x: 9.2, y: 2.35, w: 3.3, h: 1.5, fontSize: 20, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, valign: "top", objectName: "Gap main" });
  s.addText([
    { text: "Counterfeit tests reuse the same prints", options: { bullet: true, breakLine: true } },
    { text: "Few report speed on the real device", options: { bullet: true } },
  ], { x: 9.2, y: 4.0, w: 3.3, h: 2.2, fontSize: 16, color: "E6EEF6", isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 8, objectName: "Gap list" });
  s.addText("Rows with a reference: from the cited source. * Our assessment of typical hobby builds, not a cited result.", { x: 0.6, y: 4.6, w: 8.05, h: 0.25, fontSize: 11, italic: true, color: C.accent5, isTextBox: true, margin: 0, objectName: "Table note" });
  sources(s, "Photos: OrCam MyEye by Ylip, CC BY-SA 4.0; Google Glass by Ted Eytan, CC BY-SA 2.0 (Wikimedia Commons); AIris from " + cite("airis") + ". Comparison from " + cite("orcam", "glass", "airis") + ".");
  notes(s, "Commercial glasses such as OrCam work offline but not in Bangla and not with Taka. Glass-based research systems send pictures to a server. Raspberry Pi reading aids are cheap but mostly English. No device combines Bangla, Taka and offline use, and counterfeit studies do not check for copies of the same print in training and test.");

  // ================================================================ 8. architecture (stream diagram)
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Solution" });
  s.addText("System Architecture", { placeholder: "title" });
  const GREY = "8A96A3";
  const aline = (x1, y1, x2, y2, name, head = true) => arrow(s, x1, y1, x2, y2, name, head);
  // input: a stack of camera frames
  s.addText([{ text: "Input", options: { bold: true, fontSize: 16, breakLine: true } }, { text: "camera frame, 640×480", options: { fontSize: 11 } }],
    { x: 0.4, y: 1.35, w: 1.9, h: 0.55, color: HEX.dk1, align: "center", isTextBox: true, margin: 0, objectName: "Input heading" });
  ["live_100.jpg", "live_three.jpg", "live_1000_500.jpg"].forEach((f, i) =>
    s.addImage({ path: FIG + f, x: 0.45 + i * 0.14, y: 2.0 + i * 0.16, w: 1.55, h: 1.2, sizing: { type: "cover", w: 1.55, h: 1.2 }, line: { color: "FFFFFF", width: 1.5 }, objectName: "Input frame " + (i + 1) }));
  s.addText([{ text: "Trigger", options: { bold: true, breakLine: true } }, { text: "5 buttons, or voice: \"glass, …\" " + cite("vosk") }],
    { shape: pres.shapes.RECTANGLE, x: 0.45, y: 4.05, w: 1.85, h: 0.9, fill: { color: "FFFFFF" }, line: { color: GREY, width: 1 }, fontSize: 11, color: HEX.dk1, align: "center", valign: "middle", isTextBox: true, margin: 0.05, objectName: "Trigger" });
  aline(2.33, 2.95, 2.45, 2.95, "Input to manager");
  aline(2.3, 4.5, 2.45, 4.5, "Trigger to manager");
  // mode manager: tall block
  s.addText("Mode manager: starts one stream", { shape: pres.shapes.CUBE, x: 2.45, y: 1.5, w: 0.6, h: 4.65, fill: { color: "E3E7EC" }, line: { color: GREY, width: 1 },
    vert: "vert270", fontSize: 12, bold: true, color: HEX.dk1, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "Mode manager" });
  const streams = [
    ["Text-reading stream", "F2B8B0", ["Resize", "canvas 2560"], [["CRAFT", "text detector " + cite("craft")], ["Text block", "main block + clean-up"], ["EasyOCR", "fine-tuned, blended " + cite("easyocr")]]],
    ["Taka stream", "F7CB9B", ["Resize", "640"], [["YOLOv8s", "Taka detector " + cite("yolo")], ["Watermark", "4 windows + DMWR"], ["UV check", "security fibres"]]],
    ["Object stream", "AFCBEB", ["Resize", "640"], [["YOLOv8s", "COCO objects " + cite("yolo", "coco")], ["Announce", "names new objects"]]],
    ["Emotion stream", "BFE0B6", ["Face", "YuNet detector"], [["EfficientNet-V2-S", "RAF-DB " + cite("rafdb")], ["Expression", "facial expression"]]],
  ];
  const CX = [4.35, 5.95, 7.55], CW = 1.3, CH = 0.6;
  streams.forEach(([title, col, [rh, rs], cubes], r) => {
    const c = 2.05 + r * 1.15;
    s.addText(title, { x: 3.2, y: c - 0.6, w: 3.0, h: 0.25, fontFace: "Arial", fontSize: 12, bold: true, color: HEX.dk2, isTextBox: true, margin: 0, objectName: title });
    aline(3.05, c, 3.2, c, title + " in");
    s.addText(rh, { shape: pres.shapes.RECTANGLE, x: 3.2, y: c - 0.25, w: 0.85, h: 0.5, fill: { color: "FFFFFF" }, line: { color: GREY, width: 1 }, fontSize: 11, bold: true, color: HEX.dk1, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: title + " resize" });
    s.addText("[" + rs + "]", { x: 3.05, y: c + 0.28, w: 1.15, h: 0.25, fontSize: 10.5, color: HEX.dk1, align: "center", isTextBox: true, margin: 0, objectName: title + " resize size" });
    let x0 = 4.05;
    cubes.forEach(([on, under], k) => {
      aline(x0, c, CX[k], c, title + " arrow " + (k + 1));
      s.addText(on, { shape: pres.shapes.CUBE, x: CX[k], y: c - CH / 2, w: CW, h: CH, fill: { color: col }, line: { color: GREY, width: 0.75 },
        fontSize: on.length > 12 ? 9.5 : 11, bold: true, color: HEX.dk1, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: title + " block " + (k + 1) });
      s.addText(under, { x: CX[k] - 0.1, y: c + 0.3, w: CW + 0.2, h: 0.25, fontSize: 10.5, color: HEX.dk1, align: "center", isTextBox: true, margin: 0, objectName: title + " block " + (k + 1) + " label" });
      x0 = CX[k] + CW;
    });
    aline(x0, c, 9.2, c, title + " out", false);
  });
  aline(9.2, 2.05, 9.2, 5.5, "Collector", false);
  s.addShape(pres.shapes.OVAL, { x: 9.12, y: 3.7, w: 0.16, h: 0.16, fill: { color: GREY }, line: { color: GREY }, objectName: "Collector dot" });
  aline(9.28, 3.78, 9.45, 3.78, "To answer");
  const slabs = [["Answer sentence: Bangla or English", "E3E7EC", 9.45], ["Speech: Piper " + cite("piper") + ", eSpeak NG " + cite("espeak"), "F5E6A8", 10.2]];
  slabs.forEach(([t, col, x], i) => {
    s.addText(t, { shape: pres.shapes.CUBE, x, y: 1.75, w: 0.55, h: 4.1, fill: { color: col }, line: { color: GREY, width: 1 },
      vert: "vert270", fontSize: 12, bold: true, color: HEX.dk1, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "Slab " + (i + 1) });
    if (i === 0) aline(10.0, 3.78, 10.2, 3.78, "Answer to speech");
  });
  aline(10.75, 3.78, 10.95, 3.78, "Speech to output");
  // output panel
  s.addText("Spoken Output", { x: 10.95, y: 1.35, w: 1.95, h: 0.35, fontFace: "Arial", fontSize: 15, bold: true, color: HEX.dk1, align: "center", isTextBox: true, margin: 0, objectName: "Output heading" });
  s.addShape(pres.shapes.RECTANGLE, { x: 10.95, y: 1.75, w: 1.95, h: 4.1, fill: { color: "FFFFFF" }, line: { color: GREY, width: 1 }, objectName: "Output panel" });
  const outs = [
    [{ text: "পাঁচশত টাকার নোট", options: { fontFace: BN, fontSize: 13, bold: true, color: HEX.accent1 } }],
    [{ text: "\"Likely genuine\" or \"check by hand\"", options: {} }],
    [{ text: "Bangla or English text read aloud", options: {} }],
    [{ text: "Objects ahead named", options: {} }],
    [{ text: "Facial expression", options: {} }],
    [{ text: "Vibration for a note", options: { italic: true } }],
  ];
  outs.forEach((runs, i) =>
    s.addText(runs, { shape: pres.shapes.RECTANGLE, x: 11.05, y: 1.85 + i * 0.66, w: 1.75, h: 0.56, fill: { color: HEX.lt2 }, line: { color: HEX.accent6, width: 0.75 },
      fontSize: 11, color: HEX.dk1, align: "center", valign: "middle", isTextBox: true, margin: 0.03, objectName: "Output " + (i + 1) }));
  s.addText("All streams run on the Raspberry Pi 5 CPU, offline. The optional online scene-description mode is not shown.",
    { x: 3.2, y: 6.2, w: 7.6, h: 0.35, fontSize: 12, italic: true, color: HEX.accent1, isTextBox: true, margin: 0, objectName: "Architecture note" });
  sources(s, `Sources: ${cite("vosk")} Vosk; ${cite("yolo")} Ultralytics YOLOv8; ${cite("craft")} Baek et al. 2019; ${cite("easyocr")} EasyOCR; ${cite("coco")} COCO; ${cite("piper")} Piper; ${cite("rafdb")} RAF-DB; ${cite("espeak")} eSpeak NG.`);
  notes(s, "The architecture, drawn as streams. A camera frame of 640 by 480 comes in on the left, and a button press or a voice command starting with glass tells the mode manager which stream to start. The text-reading stream runs the CRAFT detector at a canvas size of 2560, keeps the main text block, and reads it with our fine-tuned EasyOCR recogniser. The Taka stream finds the note with YOLOv8s at 640, then checks four watermark windows with DMWR and counts the UV security fibres. The object stream runs YOLOv8s trained on COCO and names new objects. The emotion stream finds a face with YuNet and classifies the expression with EfficientNet-V2-S trained on RAF-DB. Whatever stream ran, its result becomes one Bangla or English sentence, which Piper speaks, with eSpeak NG as a fallback, and a short vibration confirms a note.");

  // ================================================================ 8b. hardware design (wiring diagram)
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Solution" });
  s.addText("Hardware Design", { placeholder: "title" });
  // a wire through the points [[x, y], ...], drawn as horizontal and vertical segments
  const wire = (pts, color, name, width = 2.25, dash) => {
    for (let k = 0; k + 1 < pts.length; k++) {
      const [x1, y1] = pts[k], [x2, y2] = pts[k + 1];
      s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.01), h: Math.max(Math.abs(y2 - y1), 0.01),
        line: { color, width, ...(dash ? { dashType: dash } : {}) }, objectName: name + " " + (k + 1) });
    }
  };
  const wlab = (t, x, y, w, align = "left") => s.addText(t, { x, y, w, h: 0.22, fontSize: 10, color: HEX.dk1, align, isTextBox: true, margin: 0, objectName: "Wire label: " + t });
  const comp = (file, label, x, y, w, h, name) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: "FFFFFF" }, line: { color: HEX.accent6, width: 1 }, objectName: name + " card" });
    s.addImage({ path: FIG + file, x: x + 0.08, y: y + 0.08, w: w - 0.16, h: h - 0.55, sizing: { type: "contain", w: w - 0.16, h: h - 0.55 }, objectName: name + " photo" });
    s.addText(label, { x: x + 0.05, y: y + h - 0.45, w: w - 0.1, h: 0.4, fontSize: 11, bold: true, color: HEX.dk2, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: name + " label" });
  };
  const RED = "D7263D", BLACK = "1B2430", USB = "5B6777";
  // the board
  s.addImage({ path: FIG + "hw_pi5.jpg", x: 4.85, y: 2.75, w: 3.6, h: 2.7, objectName: "Raspberry Pi 5" });
  s.addText("Raspberry Pi 5, 8 GB " + cite("rpi5") + ", in the rear enclosure", { x: 4.35, y: 5.48, w: 4.6, h: 0.3, fontSize: 13, bold: true, color: HEX.dk2, align: "center", isTextBox: true, margin: 0, objectName: "Pi label" });
  // left: camera, headphone, power bank
  comp("hw_webcam.jpg", "USB webcam (our board)", 0.6, 1.35, 2.0, 1.55, "Camera");
  wire([[2.6, 2.1], [3.9, 2.1], [3.9, 3.6], [4.85, 3.6]], USB, "Webcam USB cable", 4);
  wlab("USB cable", 2.7, 1.85, 1.2);
  comp("v2_headphone.jpg", "Bone-conduction headphone", 0.6, 3.1, 2.0, 1.45, "Headphone");
  wire([[2.6, 3.85], [3.55, 3.85], [3.55, 4.3], [4.85, 4.3]], "7B4FA0", "Audio", 2.25, "dash");
  wlab("audio out", 2.7, 3.6, 1.0);
  comp("hw_powerbank.jpg", "Power bank", 0.6, 4.75, 2.0, 1.45, "Power bank");
  wire([[2.6, 5.3], [4.3, 5.3], [4.3, 5.0], [4.85, 5.0]], RED, "Power", 3.5);
  wlab("USB-C, 5 V", 2.7, 5.05, 1.4);
  // right: buttons, vibration motor, frame with the UV lamp
  comp("v2_buttons.jpg", "Five push-buttons", 10.7, 1.35, 2.0, 1.85, "Buttons");
  const pins = [["MODE · BCM 17", "1F5F8B"], ["ACTION · BCM 27", "2E9E5B"], ["READ · BCM 24", "E6A817"], ["VOL+ · BCM 22", "4A87B5"], ["VOL− · BCM 23", "B5651D"]];
  pins.forEach(([t, col], k) => {
    const y = 1.5 + k * 0.22, x = 7.0 + k * 0.2;
    wire([[10.7, y], [x, y], [x, 2.85]], col, t + " wire", 2);
    wlab(t, 8.95, y - 0.21, 1.7, "right");
  });
  wire([[10.7, 2.62], [6.75, 2.62], [6.75, 2.85]], BLACK, "Ground", 2);
  wlab("GND", 9.6, 2.4, 1.05, "right");
  comp("hw_motor.jpg", "Vibration motor", 10.7, 3.4, 2.0, 1.35, "Motor");
  wire([[10.7, 3.95], [9.3, 3.95], [9.3, 2.74], [8.2, 2.74], [8.2, 2.85]], "C2185B", "Motor signal", 2);
  wire([[10.7, 4.2], [9.55, 4.2], [9.55, 2.62]], BLACK, "Motor ground", 2);
  s.addShape(pres.shapes.OVAL, { x: 9.5, y: 2.57, w: 0.1, h: 0.1, fill: { color: BLACK }, line: { color: BLACK }, objectName: "Ground junction" });
  wlab("BCM 13 (PWM)", 9.6, 3.7, 1.05, "right");
  wire([[8.45, 4.75], [9.35, 4.75]], USB, "USB", 4);
  s.addText("USB microphone (optional)", { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.05, x: 9.35, y: 4.5, w: 1.2, h: 0.5, fill: { color: "FFFFFF" }, line: { color: HEX.accent6 },
    fontSize: 10, color: HEX.dk1, align: "center", valign: "middle", isTextBox: true, margin: 0.03, objectName: "Microphone" });
  comp("v2_frame.jpg", "Frame + UV-A lamp", 10.7, 4.95, 2.0, 1.3, "Frame");
  s.addText("Lamp 360–395 nm, switched by hand; not wired to the Pi", { x: 8.75, y: 5.3, w: 1.85, h: 0.65, fontSize: 10, italic: true, color: HEX.dk1, align: "right", valign: "middle", isTextBox: true, margin: 0, objectName: "Lamp note" });
  // legend
  const legend = [[RED, "5 V power", 3.5], [USB, "USB", 4], [HEX.accent1, "GPIO input, to GND with pull-up", 2], [BLACK, "ground", 2]];
  legend.forEach(([col, t, wd], k) => {
    const x = [3.0, 4.55, 6.3, 9.1][k];
    s.addShape(pres.shapes.LINE, { x, y: 6.0, w: 0.35, h: 0, line: { color: col, width: wd }, objectName: "Legend line " + (k + 1) });
    s.addText(t, { x: x + 0.42, y: 5.88, w: k === 2 ? 2.4 : 1.3, h: 0.25, fontSize: 10, color: HEX.dk1, isTextBox: true, margin: 0, objectName: "Legend " + (k + 1) });
  });
  s.addText("Schematic, not to scale; pin numbers are those set in the code.", { x: 3.0, y: 6.2, w: 6.0, h: 0.25, fontSize: 10, italic: true, color: C.accent5, isTextBox: true, margin: 0, objectName: "Schematic note" });
  sources(s, "Pictures (Wikimedia Commons): Raspberry Pi 5 by SimonWaldherr, CC BY 4.0; vibration motor by Raimond Spekking, CC BY-SA 4.0; power bank by Pinguinguy, CC0. Webcam board, buttons, headphone and frame: our prototype.");
  notes(s, "This is how the second prototype is wired. The Raspberry Pi 5 sits in the rear enclosure and runs every model. The camera is a USB webcam plugged into a USB port of the Pi. The five push-buttons each connect one GPIO pin to ground, BCM 17 for MODE, 27 for ACTION, 24 for READ, 22 for volume up and 23 for volume down, with internal pull-up resistors. The vibration motor is driven by PWM from BCM 13. Speech goes to the audio output and the bone-conduction headphone, and a USB microphone can be plugged in for voice commands. A power bank supplies 5 volts through the USB-C port. The UV lamp on the frame is switched by hand and is not wired to the Pi.");

  // ================================================================ 9. prototype evolution
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Solution" });
  s.addText("Evolution of the Prototype", { placeholder: "title" });
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
  s.addText("Banknote Detection: Synthetic Training Data", { placeholder: "title" });
  s.addImage({ path: FIG + "fig_compositing.jpg", x: 0.6, y: 1.5, w: 8.2, h: 8.2 * 466 / 1600, objectName: "Compositing pipeline" });
  s.addText("Note photos " + cite("banglataka") + " are pasted onto real scenes " + cite("coco") + ", and the program writes every label itself.",
    { x: 0.6, y: 4.2, w: 8.2, h: 0.9, fontSize: 20, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "Compositing text" });
  card(s, 9.2, 1.5, 3.5, 2.3, "Synthetic card");
  tag(s, "SYNTHETIC COMPOSITES", 9.45, 1.7, 2.6, HEX.accent6, HEX.dk2, "Synthetic tag");
  stat(s, 9.45, 2.2, 3.0, "0.995", "mAP@0.5 on 2,282 held-out composites", "mAP");
  card(s, 9.2, 4.0, 3.5, 2.4, "Real card", C.text2);
  tag(s, "REAL PHOTOS", 9.45, 4.2, 1.7, HEX.accent4, HEX.dk1, "Real tag");
  s.addText("94.1%", { x: 9.45, y: 4.7, w: 3.0, h: 0.75, fontFace: "Arial", fontSize: 40, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "Real value" });
  s.addText("right note value on 1,536 photos taken by other people (laptop)", { x: 9.45, y: 5.45, w: 3.0, h: 0.8, fontSize: 14, color: "E6EEF6", isTextBox: true, margin: 0, valign: "top", objectName: "Real label" });
  sources(s, `Sources: ${cite("banglataka")} BanglaTaka (Mendeley); ${cite("coco")} Lin et al., COCO, ECCV 2014; detector ${cite("yolo")} Ultralytics YOLOv8. 94.1% with the hand-held detector (next slide).`);
  notes(s, "Public Taka images are clean close-ups, so a model trained on them fails on a live camera. We pasted 5,073 note photos onto real COCO pictures with random angle, scale, light and blur, and let the program write the box labels. The detector scores 0.995 mAP on held-out composites and, the number that matters for users, names 94.1 percent of 1,536 real photographs taken by other people correctly. The two numbers measure different things: the first is on synthetic composites, the second on real photos.");

  // ================================================================ 11. hand-held notes
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Currency" });
  s.addText("Banknote Detection: Hand-Held Notes", { placeholder: "title" });
  tag(s, "TRAINING: 6,000 SYNTHETIC CLOSE-UPS", 0.6, 1.45, 3.9, HEX.accent6, HEX.dk2, "Training tag");
  s.addImage({ path: REP + "fig58_handheld_closeups.png", x: 0.6, y: 1.9, w: 5.5, h: 2.2, objectName: "Synthetic close-ups" });
  tag(s, "TEST: REAL HAND-HELD PHOTOS", 0.6, 4.3, 3.3, HEX.accent4, HEX.dk1, "Test tag");
  s.addText("11.8% → 47.5%", { x: 0.6, y: 4.75, w: 7.0, h: 0.85, fontFace: "Arial", fontSize: 44, bold: true, color: C.accent1, isTextBox: true, margin: 0, objectName: "Hand-held takeaway" });
  s.addChart(pres.charts.BAR, [
    { name: "Earlier detector", labels: ["Laptop", "Raspberry Pi (8-bit)"], values: [11.8, 8.7] },
    { name: "Hand-held detector", labels: ["Laptop", "Raspberry Pi (8-bit)"], values: [47.5, 33.5] },
  ], { x: 7.9, y: 1.45, w: 4.8, h: 4.4, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent5, HEX.accent1], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.0\"%\"", valAxisMaxVal: 60, valAxisMinVal: 0, valAxisLabelFormatCode: "0\"%\"", showLegend: true, legendPos: "b",
    showTitle: true, title: "Right note value, real hand-held photos (NSTU, 1,127)", ...chartFrame, objectName: "Hand-held chart" });
  s.addText("Since 29 Oct the Pi runs the laptop's full-precision detector. Trade-off: with 3 notes in view, the total is right 97.0% (was 99.2%).",
    { x: 0.6, y: 5.65, w: 7.0, h: 0.8, fontSize: 14, italic: true, color: C.accent2, isTextBox: true, margin: 0, valign: "top", objectName: "Costs" });
  sources(s, `Sources: test photos ${cite("nstu")} Rafi et al., NSTU-BDTAKA, Data in Brief 2024; detector ${cite("yolo")} YOLOv8. Our results: paper_evidence/HANDHELD_FINETUNE.md.`);
  notes(s, "A blind user holds the note close, folded and under the fingers, and our training pictures showed whole notes in scenes. We generated six thousand hand-held close-ups from training notes only and fine-tuned the detector. On the independent NSTU hand-held test, the right value rises from 11.8 to 47.5 percent on the laptop and from 8.7 to 33.5 percent on the Pi. There are costs: one-taka notes are named more often by mistake (36.6 percent), and with three notes in view the total is right 97.0 percent of the time instead of 99.2.");

  // ================================================================ 12. live tests
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Currency" });
  s.addText("Live Detection Tests", { placeholder: "title" });
  s.addText("Informal demonstrations, not a measured accuracy study", { shape: pres.shapes.ROUNDED_RECTANGLE, x: 0.6, y: 1.35, w: 12.1, h: 0.42, rectRadius: 0.08,
    fill: { color: "FFF4D6" }, line: { color: HEX.accent4 }, fontSize: 15, bold: true, color: HEX.dk1, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "Demo banner" });
  const liveRows = [
    ["Software: live detection on a laptop webcam (detector output shown)", ["live_sw_1.jpg", "live_sw_3.jpg", "live_sw_4.jpg", "live_sw_5.jpg"]],
    ["Second prototype (v2): live tests with the glass (staged)", ["live_v2_16.jpg", "live_v2_18.jpg", "live_v2_24.jpg", "live_v2_29.jpg"]],
  ];
  liveRows.forEach(([label, files], r) => {
    const y = 1.88 + r * 2.25;
    s.addText(label, { x: 0.6, y, w: 12.1, h: 0.3, fontFace: "Arial", fontSize: 14, bold: true, color: HEX.dk2, isTextBox: true, margin: 0, objectName: "Live row " + (r + 1) + " label" });
    files.forEach((f, i) => photo(s, AS + f, 0.6 + i * 3.06, y + 0.33, 2.92, 1.85, "Live row " + (r + 1) + " photo " + (i + 1)));
  });
  s.addText("Screenshots: 50, 200, 500 and 1,000 Taka named at 0.63–0.89 confidence, two or three notes at once. The v2 photos do not show the detector output.",
    { x: 0.6, y: 6.42, w: 12.1, h: 0.4, fontSize: 13, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "Live test note" });
  notes(s, "The top row is the software on a laptop webcam: the detector draws a box with the note value and its confidence. Fifty, two hundred, five hundred and one thousand Taka were named at 0.63 to 0.89 confidence, with two or three notes at once. The bottom row is the second prototype in live tests: a team member holds a note up in front of the glass camera; these scenes are staged and the photos do not show the detector output. We did not count an error rate in these sessions.");

  // ================================================================ 13. prefix training
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Currency" });
  s.addText("Counterfeit Screening: Prefix-Robust Multi-View Training", { placeholder: "title" });
  s.addChart(pres.charts.LINE, [
    { name: "Fixed six-view training", labels: ["1", "2", "3", "4", "5", "6"], values: [71.2, 73.1, 85.7, 90.5, 90.5, 98.1] },
    { name: "Prefix training (ours)", labels: ["1", "2", "3", "4", "5", "6"], values: [98.2, 98.4, 98.6, 98.6, 98.7, 98.7] },
  ], { x: 0.6, y: 1.45, w: 7.6, h: 5.0, chartColors: [HEX.accent5, HEX.accent1], lineSize: 3, lineDataSymbolSize: 8, valAxisMinVal: 60, valAxisMaxVal: 100,
    valAxisLabelFormatCode: "0\"%\"", showLegend: true, legendPos: "b", showTitle: true, title: "Test accuracy by number of views (mean of 3 seeds)",
    catAxisTitle: "Views captured", showCatAxisTitle: true, catAxisTitleFontSize: 12, catAxisTitleColor: HEX.accent5, ...chartFrame, objectName: "Prefix chart" });
  card(s, 8.6, 1.45, 4.1, 4.95, "Takeaway card", C.text2);
  s.addText("With a single view", { x: 8.9, y: 1.75, w: 3.5, h: 0.45, fontSize: 18, color: "D2E0EC", isTextBox: true, margin: 0, objectName: "One view label" });
  s.addText("71.2%", { x: 8.9, y: 2.25, w: 3.5, h: 0.7, fontFace: "Arial", fontSize: 36, bold: true, color: "B8C6D4", isTextBox: true, margin: 0, objectName: "Before value" });
  s.addText("rises to", { x: 8.9, y: 2.95, w: 3.5, h: 0.45, fontSize: 18, italic: true, color: "D2E0EC", isTextBox: true, margin: 0, objectName: "Rises to" });
  s.addText("98.2%", { x: 8.9, y: 3.4, w: 3.5, h: 0.95, fontFace: "Arial", fontSize: 54, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "After value" });
  s.addText("Held-out JaalTaka test notes, never used for training or for any choice", { x: 8.9, y: 5.55, w: 3.5, h: 0.7, fontSize: 12, italic: true, color: "D2E0EC", isTextBox: true, margin: 0, valign: "top", objectName: "Split note" });
  s.addText("Training on random first views: one view is enough.",
    { x: 8.9, y: 4.5, w: 3.5, h: 1.0, fontSize: 18, color: "E6EEF6", isTextBox: true, margin: 0, valign: "top", objectName: "Prefix takeaway" });
  sources(s, `Data: ${cite("jaal", "jaal26")} Piyas et al., JaalTaka dataset and article, 2025-2026 (1,390 notes, six views each; counterfeits seized by the Rapid Action Battalion).`);
  notes(s, "A blind user cannot promise to capture six views of a note. A model trained only with all six views collapses to 71 percent with one view. Training on random prefixes of the views, that is the first k views for a random k, keeps it at about 98 percent from one to six views, with no loss at six views and stable results across three seeds.");

  // ================================================================ 14. unseen prints
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Currency" });
  s.addText("Counterfeit Screening: Evaluation on Unseen Prints", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [{ name: "6 views", labels: ["Frozen ResNet-50", "Prefix network", "Fine-tuned ResNet-50", "Prefix + watermark"], values: [83.8, 89.3, 94.4, 95.0] }],
    { x: 0.6, y: 1.45, w: 7.6, h: 5.0, barDir: "bar", chartColors: [HEX.accent5, HEX.accent3, HEX.accent5, HEX.accent1], showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: "0.0\"%\"", valAxisMinVal: 70, valAxisMaxVal: 100, valAxisLabelFormatCode: "0\"%\"", showLegend: false, showTitle: true,
      title: "Accuracy on counterfeit prints never seen in training (6 views, 3 seeds)", ...chartFrame, objectName: "Unseen prints chart" });
  card(s, 8.6, 1.45, 4.1, 2.55, "Main result", C.text2);
  s.addText("95.0%", { x: 8.9, y: 1.65, w: 3.5, h: 1.0, fontFace: "Arial", fontSize: 54, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "Unseen value" });
  s.addText("on counterfeit prints never seen in training, with the watermark check", { x: 8.9, y: 2.7, w: 3.5, h: 1.1, fontSize: 16, color: "E6EEF6", isTextBox: true, margin: 0, valign: "top", objectName: "Unseen label" });
  s.addText([{ text: "Why a stricter split: ", options: { bold: true } }, { text: "279 of 322 readable 500 Taka counterfeits share one serial number, so the usual split tests copies of known prints." }],
    { x: 8.6, y: 4.25, w: 4.1, h: 2.1, fontSize: 15, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "Unseen text" });
  sources(s, `Data: ${cite("jaal")} JaalTaka; serial-number audit and print-disjoint split are ours. Earlier Taka counterfeit work: ${cite("uddin")} SVM on security features; ${cite("pathak")} CNNs; ${cite("uvink")} UV flecks + RPN.`);
  notes(s, "Most counterfeit notes in JaalTaka are copies of a few prints, so the usual test measures recognition of known reprints; a serial-number lookup alone scores 90.2 percent. On truly unseen prints our network drops to about 89 percent; adding the watermark check brings it to 95 percent.");

  // ================================================================ 15. DMWR
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Currency" });
  s.addText("Watermark Verification: DMWR", { placeholder: "title" });
  s.addImage({ path: FIG + "fig_dmwr_steps.png", x: 0.6, y: 1.45, w: 7.6, h: 7.6 * 960 / 1800, objectName: "DMWR steps" });
  s.addText([
    { text: "1. Remove the lamp:  Q = I / (G∗I)", options: { breakLine: true } },
    { text: "2. Genuine portrait:  T = median of Q over genuine notes", options: { breakLine: true } },
    { text: "3. Residual R = Q − T → two small CNN streams → \"watermark clear?\"" },
  ], { x: 0.6, y: 5.6, w: 7.6, h: 0.95, fontSize: 14, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "DMWR steps text" });
  card(s, 8.6, 1.45, 4.1, 5.0, "DMWR stats");
  s.addText("91.5%", { x: 8.9, y: 1.7, w: 3.6, h: 1.0, fontFace: "Arial", fontSize: 54, bold: true, color: C.accent1, isTextBox: true, margin: 0, objectName: "DMWR value" });
  s.addText("on unseen counterfeit prints (3 seeds, ±0.3)", { x: 8.9, y: 2.7, w: 3.6, h: 0.7, fontSize: 16, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "DMWR value label" });
  stat(s, 8.9, 3.65, 3.6, "46× smaller", "than MobileNetV2 " + cite("mnv2") + " (47,818 parameters, 0.21 MB), and as accurate: 92.9%, p ≥ 0.375", "DMWR size");
  sources(s, `Baseline ${cite("mnv2")} Sandler et al., MobileNetV2, CVPR 2018; data ${cite("jaal")} JaalTaka. DMWR (denomination-matched watermark residual) is our method.`);
  notes(s, "Our watermark method divides the back-lit photo by a blurred copy to remove the lamp, subtracts a genuine portrait of the note value and classifies the residual with two tiny networks. With under fifty thousand parameters it reaches 91.5 percent on unseen prints, statistically level with a MobileNetV2 46 times larger.");

  // ================================================================ 16. safe verdict
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Currency" });
  s.addText("Safe Verdict Policy", { placeholder: "title" });
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
  s.addText("UV Security-Fibre Verification", { placeholder: "title" });
  s.addImage({ path: FIG + "fig_uv_fibres.jpg", x: 0.6, y: 1.45, w: 12.1, h: 12.1 * 520 / 1816, objectName: "UV fibres" });
  stat(s, 0.6, 5.1, 3.6, "85.6%", "of 111 real UV photos of genuine notes accepted", "UV genuine");
  stat(s, 4.6, 5.1, 3.6, "88.6%", "of SIMULATED fibre-free sheets rejected (not real counterfeits)", "UV sheets", C.accent5);
  sources(s, `The glass's lamp emits 360-395 nm, covering the 365 nm used for banknote checks; banknote paper stays dark while its fibres glow ${cite("mutanen")}. Prior Taka UV work: ${cite("uvink")} (96.3%). Fibre check and photos are ours.`);
  s.addText([{ text: "Speaks under the lamp", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: "\"Likely genuine\" or \"check by hand\", never \"counterfeit\". Not yet tested on real counterfeits or through the glass camera", options: { fontSize: 13 } }],
    { shape: pres.shapes.ROUNDED_RECTANGLE, x: 8.6, y: 5.0, w: 4.1, h: 1.4, rectRadius: 0.1, fill: { color: "FBE9EB" }, line: { color: HEX.accent2, width: 1.25 },
      color: HEX.accent2, isTextBox: true, margin: 12, valign: "middle", objectName: "UV caveat" });
  notes(s, "Under a UV lamp, genuine 200, 500 and 1000 Taka notes show glowing security fibres. Our check removes the lamp's own blue light and counts thin glowing marks. It accepts 85.6 percent of real genuine photos and rejects 88.6 percent of simulated fibre-free sheets. Since 29 October currency mode runs it whenever the picture is taken under the lamp, which it recognises from the colour: all 47 UV test photos were recognised and none of 2,282 ordinary photos were mistaken for one. It then says likely genuine or check by hand, never counterfeit, because no counterfeit note has been photographed under UV and it has not been tested through the glass camera.");

  // ================================================================ 18. text reading
  pres.addSection({ title: "Reading and voice" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Reading and voice" });
  s.addText("Bangla and English Text Recognition", { placeholder: "title" });
  const ocrOpts = (x, w, title, max, name) => ({ x, y: 1.95, w, h: 4.5, barDir: "col", barGapWidthPct: 50, chartColors: [HEX.accent5, HEX.accent1], showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\"%\"", valAxisMinVal: 0, valAxisMaxVal: max, valAxisLabelFormatCode: "0\"%\"", showLegend: true, legendPos: "b",
    showTitle: false, ...chartFrame, objectName: name });
  s.addText("Rendered text (clean)", { x: 0.6, y: 1.45, w: 3.4, h: 0.45, fontFace: "Arial", fontSize: 19, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "Rendered heading" });
  s.addChart(pres.charts.BAR, [{ name: "Before", labels: ["All scripts"], values: [9.3] }, { name: "Now", labels: ["All scripts"], values: [2.3] }],
    ocrOpts(0.6, 3.4, "", 12, "Rendered chart"));
  s.addText("Real word photos (hard)", { x: 4.3, y: 1.45, w: 4.3, h: 0.45, fontFace: "Arial", fontSize: 19, bold: true, color: C.accent2, isTextBox: true, margin: 0, objectName: "Real heading" });
  s.addChart(pres.charts.BAR, [{ name: "Before", labels: ["Bangla", "Latin"], values: [39.2, 62.5] }, { name: "Now", labels: ["Bangla", "Latin"], values: [28.7, 50.6] }],
    ocrOpts(4.3, 4.3, "", 70, "Real chart"));
  s.addText("Character error rate, lower is better. Measured on evaluation datasets, not yet through the glass.", { x: 0.6, y: 6.32, w: 8.0, h: 0.3, fontSize: 12, bold: true, italic: true, color: C.accent5, isTextBox: true, margin: 0, objectName: "Chart note" });
  s.addText("What helped", { x: 8.95, y: 1.45, w: 3.75, h: 0.45, fontFace: "Arial", fontSize: 20, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "Helped heading" });
  s.addText([
    { text: "Reading the main text block only", options: { bullet: true, breakLine: true } },
    { text: "Tuned detector and clean-up", options: { bullet: true, breakLine: true } },
    { text: "Recogniser fine-tuned on real words", options: { bullet: true, breakLine: true } },
    { text: "Sharpest frame of 1 s: shaken text 28.0% → 3.8%", options: { bullet: true } },
  ], { x: 8.95, y: 1.95, w: 3.75, h: 2.3, fontSize: 15, color: C.text1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 8, objectName: "Helped list" });
  s.addText("What did not", { x: 8.95, y: 4.3, w: 3.75, h: 0.45, fontFace: "Arial", fontSize: 20, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "Not heading" });
  s.addText("Binarisation, Tesseract, TrOCR, PaddleOCR, dictionaries",
    { x: 8.95, y: 4.8, w: 3.75, h: 1.3, fontSize: 17, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "Not list" });
  sources(s, `Engine ${cite("easyocr")} EasyOCR with ${cite("craft")} CRAFT; real words from ${cite("mlt")} ICDAR 2019 MLT (1,000 test words). Every choice made on validation data.`);
  notes(s, "On our rendered Bangla and English benchmark the error fell from 9.3 to 2.3 percent. Real street words are much harder: a recogniser fine-tuned on real words and blended with the original weights lowered the error from 39.2 to 28.7 percent for Bangla and from 62.5 to 50.6 percent for Latin script. What helped: reading only the main block of text, tuned detector settings with CLAHE and clean-up rules, and the recogniser fine-tuned on real words and blended fifty-fifty with the original weights. Binarisation, multi-scale reading, Tesseract, TrOCR, PaddleOCR, dictionaries, smaller pictures and a rotation retry did not help. For shaky hands, the glass now watches the camera for one second after a press and reads the sharpest frame; on validation text with simulated shake the error fell from 28.0 to 3.8 percent, and from 50.4 to 3.5 percent when the press itself shook the camera most.");

  // ================================================================ 19. voice
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Reading and voice" });
  s.addText("Voice Interaction", { placeholder: "title" });
  card(s, 0.6, 1.45, 4.0, 4.4, "Speech card");
  circleIcon(s, "speak", 0.9, 1.7, 0.85, HEX.accent1, "Voice");
  s.addText("Speaking", { x: 1.95, y: 1.85, w: 2.5, h: 0.5, fontFace: "Arial", fontSize: 20, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "Speaking heading" });
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
  s.addText("Performance on the Raspberry Pi 5", { placeholder: "title" });
  const pis = [["422 ms", "currency check, end to end"], ["51 ms", "watermark check"], ["462 ms", "object scan (YOLOv8s)"], ["536 ms", "emotion, face in view"],
    ["5.4 s", "3-line sign (18.7 s on 1 October)"], ["2.76 GB", "peak memory of the app (of 8 GB)"]];
  pis.forEach(([v, l], i) => {
    const x = 0.6 + (i % 3) * 2.75, y = 1.45 + Math.floor(i / 3) * 2.45;
    card(s, x, y, 2.55, 2.2, "Pi stat " + (i + 1));
    stat(s, x + 0.25, y + 0.3, 2.1, v, l, "Pi " + (i + 1), i === 4 ? C.accent2 : C.accent1);
  });
  card(s, 8.95, 1.45, 3.75, 4.65, "Heat card", C.text2);
  circleIcon(s, "therm", 9.25, 1.7, 0.8, HEX.accent2, "Heat");
  s.addText("Board temperature", { x: 10.2, y: 1.85, w: 2.4, h: 0.5, fontFace: "Arial", fontSize: 19, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "Heat heading" });
  s.addText("57.8–62.3 °C", { x: 9.25, y: 2.65, w: 3.3, h: 0.8, fontFace: "Arial", fontSize: 34, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "Heat peak" });
  s.addText("measured on the board with a thermometer while the glass was running", { x: 9.25, y: 3.45, w: 3.25, h: 0.9, fontSize: 15, color: "E6EEF6", isTextBox: true, margin: 0, valign: "top", objectName: "Heat label" });
  s.addText("The processor's own sensor reads higher: in 30 minutes of non-stop load it peaked at 87.0 °C, with the soft limit active in 55% of samples.",
    { x: 9.25, y: 4.55, w: 3.25, h: 1.3, fontSize: 14, italic: true, color: "D2E0EC", isTextBox: true, margin: 0, valign: "top", objectName: "Heat response" });
  s.addText("Timings from the 29 October 2026 benchmark on the Pi 5 (8 GB). The Pi now runs the full-precision note detector: about 0.3 s more per check, not yet timed.",
    { x: 0.6, y: 6.25, w: 12.1, h: 0.4, fontSize: 12, italic: true, color: C.accent5, isTextBox: true, margin: 0, objectName: "Pi caveat" });
  sources(s, `Board: ${cite("rpi5")} Raspberry Pi 5 (8 GB), fan and heat sink, C270 webcam. All timings are ours (benchmark_report/2026-10-29/); board-surface temperature measured by the team with a thermometer.`);
  notes(s, "Measured on the Raspberry Pi 5 itself on 29 October. The currency check takes 0.42 seconds, the watermark check 51 milliseconds, an object scan 0.46 seconds and emotion with a face 0.54 seconds. Text reading is now about three times faster than on 1 October: a three-line sign in 5.4 seconds, and on a twelve-line page the first line is spoken after 4.4 seconds. With a thermometer on the board while the glass was running, we measured 57.8 to 62.3 degrees Celsius on its surface. The processor's own sensor reads higher: in thirty minutes of non-stop load it peaked at 87 degrees and the soft temperature limit was active about half the time, so heat under continuous load is still a limit. Since that run the Pi uses the same full-precision note detector as the laptop, which costs about 0.3 seconds per check and has not been timed yet.");

  // ================================================================ 21. testing the worn glass
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Device" });
  s.addText("Prototype Testing and Project Status", { placeholder: "title" });
  s.addImage({ path: AS + "v2_use_collage.jpg", x: 0.6, y: 1.45, w: 7.3, h: 7.3 * 1612 / 2412, objectName: "Prototype version 2 in use" });
  s.addText("Prototype version 2 in a staged shop exchange by team members; illustration only, no error rate counted.",
    { x: 0.6, y: 6.4, w: 7.5, h: 0.4, fontSize: 14, italic: true, color: C.accent5, isTextBox: true, margin: 0, valign: "top", objectName: "Worn caption" });
  const status = [
    ["DONE", HEX.accent1, "FFFFFF", HEX.lt2, "Engineering checks", "Each part tested on held-out data, on a laptop and the Pi 5"],
    ["PRELIMINARY", HEX.accent2, "FFFFFF", "FBE9EB", "First field test", "Glass camera: currency unreliable, Bangla text not read"],
    ["IN PROGRESS", HEX.accent4, HEX.dk1, "FFF8E6", "Participant trials", "Sessions with a visually impaired participant have started"],
    ["NONE YET", HEX.accent5, "FFFFFF", "F3F5F4", "Full user study", "No results about users are claimed"],
  ];
  status.forEach(([t, col, tc, fill, h, body], i) => {
    const y = 1.45 + i * 1.27;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.45, y, w: 4.25, h: 1.17, rectRadius: 0.1, fill: { color: fill }, line: { color: col, width: 1.25 }, objectName: h + " card" });
    tag(s, t, 8.65, y + 0.12, 1.45, col, tc, h + " status");
    s.addText(h, { x: 10.2, y: y + 0.09, w: 2.45, h: 0.42, fontFace: "Arial", fontSize: 17, bold: true, color: C.text2, isTextBox: true, margin: 0, valign: "middle", objectName: h + " heading" });
    s.addText(body, { x: 8.65, y: y + 0.58, w: 3.9, h: 0.55, fontSize: 14, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: h + " text" });
  });
  notes(s, "The second prototype on team members. The first field test with the glass camera was not good enough: currency was unreliable and Bangla text was not read, probably because of the low-resolution fixed-focus camera. Sessions with a visually impaired participant have started, with a Bangla voice questionnaire, but no user study is complete yet, so we make no claims about users.");

  // ================================================================ 22. next steps
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Device" });
  s.addText("Future Directions", { placeholder: "title" });
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
    s.addText(h, { x, y: 3.7, w: 3.18, h: 0.55, fontFace: "Arial", fontSize: 24, bold: true, color: C.text2, align: "center", isTextBox: true, margin: 0, objectName: h + " heading" });
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
    ["glasses", "Built", "An offline Raspberry Pi 5 glass that reads Bangla and English, names objects and Taka notes, and speaks Bangla"],
    ["shield", "Investigated", "Synthetic training data, prefix training, unseen prints, a safe verdict, DMWR, hand-held notes, OCR, voice"],
    ["users", "Still to validate", "Real counterfeits (also under UV), people's voices, the glass camera, a completed user study"],
  ];
  concl.forEach(([k, h, t], i) => {
    const y = 1.4 + i * 1.78;
    card(s, 5.1, y, 7.6, 1.6, h + " card");
    circleIcon(s, k, 5.35, y + 0.4, 0.8, i === 2 ? HEX.accent3 : HEX.accent1, h);
    s.addText(h, { x: 6.4, y: y + 0.2, w: 6.1, h: 0.5, fontFace: "Arial", fontSize: 22, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: h + " heading" });
    s.addText(t, { x: 6.4, y: y + 0.75, w: 6.1, h: 0.75, fontSize: 17, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: h + " text" });
  });
  notes(s, "To sum up: SARA turns three everyday moments, money, reading and the street, into questions the user can ask the glass. It is an assistive glass that speaks Bangla and runs on a Raspberry Pi 5, tested part by part on held-out data, with new methods for counterfeit checking and hand-held notes. Still to be validated: real counterfeit notes, including under the UV lamp; voice commands spoken by people; the glass camera; and a completed user study.");

  // ================================================================ 24. thank you
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Closing" });
  s.addText("THANK YOU", { x: 0.6, y: 2.35, w: 12.1, h: 1.6, fontFace: "Arial", fontSize: 72, color: HEX.dk1, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "Thanks" });
  s.addText("Questions?", { x: 0.6, y: 4.0, w: 12.1, h: 0.6, fontSize: 26, color: HEX.accent1, align: "center", isTextBox: true, margin: 0, objectName: "Questions" });
  s.addText("SARA  ·  Smart Assistive Reading Aid  ·  Independent University, Bangladesh", { x: 0.6, y: 6.5, w: 12.1, h: 0.4, fontSize: 14, color: HEX.accent5, align: "center", isTextBox: true, margin: 0, objectName: "Closing name" });
  notes(s, "Thank you; we are happy to take questions.");

  // ================================================================ 25. references
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Closing" });
  s.addText("References", { placeholder: "title" });
  const unused = REFS.filter((r) => !NUM[r[0]]).map((r) => r[0]);
  if (unused.length) throw new Error("references never cited: " + unused.join(", "));
  const CITED = REFS.slice().sort((a, b) => NUM[a[0]] - NUM[b[0]]);
  const half = Math.ceil(CITED.length / 2);
  [CITED.slice(0, half), CITED.slice(half)].forEach((list, col) => {
    s.addText(list.map((r, i) => ({ text: `[${NUM[r[0]]}]  ${r[1]}`, options: { breakLine: i < list.length - 1 } })),
      { x: 0.6 + col * 6.15, y: 1.4, w: 5.95, h: 5.4, fontSize: 12, color: C.text1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 5, objectName: "References " + (col + 1) });
  });
  notes(s, "The sources cited on the slides, numbered as in the slides.");

  // ================================================================ 26. links
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Closing" });
  s.addText("Resources and Links", { placeholder: "title" });
  const groups = [
    ["git", "Our code and results", [["SARA: glass software for the Raspberry Pi 5", "https://github.com/aflam330/SARA"],
      ["Research repository: experiments, results, thesis", "https://github.com/aflam330/Senior_project_wearable_glass_for_blind_people_using_ocr_and_ylo"]]],
    ["data", "Datasets", [["JaalTaka (counterfeit notes)", URL.jaal], ["BanglaTaka (note photos)", URL.banglataka],
      ["NSTU-BDTAKA (hand-held notes)", URL.nstu], ["Microsoft COCO", URL.coco], ["ICDAR 2019 MLT", URL.mlt],
      ["ESC-50 environmental sounds", URL.esc50]]],
    ["tools", "Open-source tools", [["Ultralytics YOLOv8", URL.yolo], ["EasyOCR", URL.easyocr], ["Piper text-to-speech", URL.piper],
      ["Vosk speech recognition", URL.vosk], ["eSpeak NG", URL.espeak], ["WHO vision fact sheet", URL.who]]],
  ];
  groups.forEach(([k, h, items], g) => {
    const x = 0.6 + g * 4.1;
    card(s, x, 1.45, 3.85, 5.1, h + " card");
    s.addImage({ data: I[k], x: x + 0.25, y: 1.65, w: 0.5, h: 0.5, objectName: h + " icon" });
    s.addText(h, { x: x + 0.9, y: 1.68, w: 2.8, h: 0.45, fontFace: "Arial", fontSize: 18, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: h + " heading" });
    const runs = [];
    items.forEach(([t, u], i) => {
      runs.push({ text: t, options: { hyperlink: { url: u, tooltip: u }, color: HEX.accent1, bold: true, fontSize: 15, breakLine: true } });
      runs.push({ text: u.replace(/^https?:\/\//, ""), options: { color: HEX.accent5, fontSize: 11, breakLine: i < items.length - 1, paraSpaceAfter: 9 } });
    });
    s.addText(runs, { x: x + 0.25, y: 2.35, w: 3.4, h: 4.05, isTextBox: true, margin: 0, valign: "top", fit: "none", objectName: h + " links" });
  });
  notes(s, "Our code is public on GitHub: the glass software and the research repository with every experiment. The datasets and open-source tools we used are linked here.");

  await pres.writeFile({ fileName: OUT });
  if (!process.env.NOTHEME) await applyTheme(OUT, THEME);
  console.log("written", OUT);
})();
