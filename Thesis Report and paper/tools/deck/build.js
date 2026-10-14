// SARA: project presentation (pptxgenjs, structured deck)
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const md = require("react-icons/md");
const path = require("path");
const fs = require("fs");
const { applyTheme } = require("C:/Users/LAM/.claude/skills/synced/78df2a5e-005c-4f3e-a592-b1286ff813dd_9e1bb536-53da-4e98-a30a-05e6767fad10/pptx/scripts/apply_theme.js");

const FIG = "E:/Final SP/Thesis Report and paper/thesis report/My report/DIagram fig tab/";
const OUT_DIR = "E:/Final SP/Thesis Report and paper/thesis report/slides/";
const OUT = OUT_DIR + "SARA_Presentation.pptx";

const THEME = {
  name: "SARA",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F23", lt1: "FFFFFF", dk2: "0B3D2E", lt2: "EAF4EF",
    accent1: "0B6E4F", accent2: "E4572E", accent3: "F2A541", accent4: "3D8B7D",
    accent5: "6B7B76", accent6: "1C7293", hlink: "1C7293", folHlink: "6B7B76",
  },
};
const HEX = THEME.colors;

async function icon(Comp, hex, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
  pres.title = "SARA";
  pres.author = "MD Abdun Fattah Lam, Mukut Protim Memo, Tanjila Hossain Tapti";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  // ---------------- layouts
  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.5, w: 7.4, h: 1.4, fontSize: 54, bold: true, color: C.background1, align: "left", valign: "bottom", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.05, w: 7.4, h: 1.7, fontSize: 22, color: "A8D5BA", align: "left", valign: "top", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: C.background1 },
    margin: [0.5, 0.6, 0.6, 0.6],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.text2, align: "left", valign: "middle", margin: 0 }, text: "" } },
      { text: { text: "SARA  |  IUB CSE Senior Project", options: { x: 0.6, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
    ],
    slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
  });

  // ---------------- icons
  const I = {};
  const want = {
    text: [md.MdTextFields, HEX.lt1], obj: [md.MdCategory, HEX.lt1], money: [md.MdPayments, HEX.lt1],
    shield: [md.MdVerifiedUser, HEX.lt1], mood: [md.MdMood, HEX.lt1], cloud: [md.MdCloudQueue, HEX.lt1],
    lang: [md.MdTranslate, HEX.lt1], wifi: [md.MdWifiOff, HEX.lt1], trust: [md.MdHelpOutline, HEX.lt1],
    cam: [md.MdPhotoCamera, HEX.lt1], touch: [md.MdTouchApp, HEX.lt1], chip: [md.MdMemory, HEX.lt1],
    voice: [md.MdRecordVoiceOver, HEX.lt1], warn: [md.MdWarningAmber, HEX.lt1], group: [md.MdGroups, HEX.lt1],
    speed: [md.MdSpeed, HEX.lt1], thermo: [md.MdThermostat, HEX.lt1], battery: [md.MdBatteryAlert, HEX.lt1],
    data: [md.MdDataset, HEX.lt1], eye: [md.MdVisibilityOff, HEX.lt1],
  };
  for (const [k, [comp, col]] of Object.entries(want)) I[k] = await icon(comp, col);

  // icon in a coloured circle: the deck's visual motif
  function badge(slide, key, x, y, d, fill, name) {
    slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: name + " circle" });
    const p = d * 0.22;
    slide.addImage({ data: I[key], x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, objectName: name + " icon" });
  }
  function numBadge(slide, n, x, y, d, fill) {
    slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: "number " + n });
    slide.addText(String(n), { x, y, w: d, h: d, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  }
  function card(slide, x, y, w, h, name) {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: name });
  }
  function photo(slide, file, x, y, w, h, name) {
    slide.addImage({ path: FIG + file, x, y, w, h, sizing: { type: "cover", w, h }, objectName: name });
  }
  const body = (t, o) => Object.assign({ fontSize: 16, color: C.text1, margin: 0, isTextBox: true, valign: "top" }, o, { });

  // ======================================================== 1. title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  s.addText("SARA", { placeholder: "title" });
  s.addText("Smart Assistive Reading Aid: An Offline Smart Glass for Visually Impaired People that Reads Bangla and English Text, Detects Objects, and Recognises and Checks Bangladeshi Taka Notes", { placeholder: "body" });
  s.addText([
    { text: "MD Abdun Fattah Lam (2220014)  ·  Mukut Protim Memo (2020086)  ·  Tanjila Hossain Tapti (2130561)", options: { breakLine: true } },
    { text: "Supervisor: Mohammad Shidujaman, PhD, Assistant Professor", options: { breakLine: true } },
    { text: "Department of Computer Science and Engineering, Independent University, Bangladesh  ·  Spring 2026" },
  ], { x: 0.7, y: 5.0, w: 7.4, h: 1.4, fontSize: 14, color: C.background1, margin: 0, isTextBox: true, paraSpaceAfter: 6 });
  photo(s, "worn_front.jpg", 8.75, 0.7, 4.0, 6.1, "Prototype worn");
  s.addNotes("Good morning. We are presenting SARA, a smart glass for blind and low-vision people in Bangladesh. It reads Bangla and English text, names objects, recognises Taka notes and gives careful advice about counterfeit notes, and it does this on the device without the internet. The photo shows our first prototype being worn.");

  // ======================================================== 2. problem
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("Reading, recognising and paying still need a helper", { placeholder: "title" });
  s.addText([
    { text: "A blind person buys medicine and receives change. To know what the label says, which note came back and whether it is real, they must ask someone, and they cannot check the answer.", options: { breakLine: true } },
  ], body("", { x: 0.6, y: 1.5, w: 5.2, h: 2.2, fontSize: 18 }));
  s.addText("2.2 billion", { x: 0.6, y: 4.1, w: 5.2, h: 1.0, fontSize: 60, bold: true, color: C.accent1, margin: 0, isTextBox: true, fontFace: "Cambria" });
  s.addText("people live with some form of vision impairment (WHO)", body("", { x: 0.6, y: 5.15, w: 5.2, h: 0.6, fontSize: 16, color: C.accent5 }));
  const probs = [
    ["lang", "Built for English", "Existing devices do not read Bangla or know Taka notes."],
    ["wifi", "Need internet or money", "Many send pictures to a server or need a phone and a paid service."],
    ["trust", "Answers cannot be checked", "A wrong note or a false \"counterfeit\" costs money or causes an accusation."],
  ];
  probs.forEach(([k, h, t], i) => {
    const y = 1.5 + i * 1.75;
    card(s, 6.4, y, 6.3, 1.5, "problem card " + (i + 1));
    badge(s, k, 6.65, y + 0.3, 0.9, i === 2 ? HEX.accent2 : HEX.accent1, "problem " + (i + 1));
    s.addText(h, body("", { x: 7.8, y: y + 0.22, w: 4.7, h: 0.45, fontSize: 20, bold: true, color: C.text2 }));
    s.addText(t, body("", { x: 7.8, y: y + 0.7, w: 4.7, h: 0.7, fontSize: 15 }));
  });
  s.addNotes("The problem in one sentence: there is no affordable wearable device that reads Bangla, recognises Taka and gives trustworthy advice about counterfeit notes without a phone or the internet. Three things make it hard: language, connectivity and cost, and trust, because a blind user cannot check what the device says.");

  // ======================================================== 3. solution
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("SARA: five modes, five buttons, Bangla speech", { placeholder: "title" });
  photo(s, "proto_front.jpg", 0.6, 1.45, 3.9, 5.2, "Prototype photo");
  const modes = [
    ["text", "Text reading", "Bangla and English OCR, spoken right after capture"],
    ["obj", "Objects", "Names objects in front of the user every 1.5 s"],
    ["money", "Currency", "All nine Taka notes, with a cautious counterfeit check"],
    ["mood", "Emotion", "Facial expression of the person in front"],
    ["cloud", "Online description", "Optional; the only mode that needs the internet"],
  ];
  modes.forEach(([k, h, t], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 4.9 + col * 3.95, y = 1.45 + row * 1.75;
    card(s, x, y, 3.75, 1.55, "mode card " + (i + 1));
    badge(s, k, x + 0.2, y + 0.3, 0.8, k === "cloud" ? HEX.accent5 : HEX.accent1, "mode " + (i + 1));
    s.addText(h, body("", { x: x + 1.15, y: y + 0.2, w: 2.45, h: 0.45, fontSize: 18, bold: true, color: C.text2 }));
    s.addText(t, body("", { x: x + 1.15, y: y + 0.65, w: 2.45, h: 0.8, fontSize: 14 }));
  });
  s.addText("Controlled by MODE, ACTION, READ and two volume buttons. Every change is spoken.",
    body("", { x: 8.85, y: 5.0, w: 3.85, h: 1.4, fontSize: 15, italic: true, color: C.accent1 }));
  s.addNotes("This is our solution. A camera sits above the bridge of a 3D-printed frame and a Raspberry Pi 5 runs everything. The user cycles five modes with one button and presses ACTION to capture. Only the online mode needs the internet; the other four run on the device.");

  // ======================================================== 4. research questions
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("Six research questions", { placeholder: "title" });
  const rqs = [
    ["System", "Can a multi-function Bangla glass run on a Raspberry Pi 5, and how fast?"],
    ["Synthetic data", "Can a Taka detector trained without hand labels work on real photographs?"],
    ["Number of views", "Why do multi-view counterfeit models fail with fewer views, and can training fix it?"],
    ["Unseen prints", "How much does accuracy drop on counterfeit prints never seen in training?"],
    ["Safety", "Can the glass give counterfeit advice that never accuses a genuine note?"],
    ["Bangla OCR", "Which part of a Bangla OCR pipeline matters most for a wearable camera?"],
  ];
  rqs.forEach(([h, t], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 6.2, y = 1.55 + row * 1.7;
    numBadge(s, i + 1, x, y + 0.05, 0.7, HEX.accent1);
    s.addText(h, body("", { x: x + 0.95, y, w: 5.0, h: 0.45, fontSize: 20, bold: true, color: C.text2 }));
    s.addText(t, body("", { x: x + 0.95, y: y + 0.5, w: 5.0, h: 0.9, fontSize: 16 }));
  });
  s.addNotes("These six questions organise the rest of the talk: the system on the Pi, synthetic training data for currency, the number of views, unseen counterfeit prints, a safe spoken verdict, and Bangla OCR.");

  // ======================================================== 5. gap
  pres.addSection({ title: "Background" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Background" });
  s.addText("What existing devices miss", { placeholder: "title" });
  const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, fontSize: 16, align: "center", valign: "middle" } });
  const cell = (t, good) => ({ text: t, options: { fontSize: 15, align: "center", valign: "middle", bold: good, color: good ? HEX.accent1 : (t === "No" ? HEX.accent2 : HEX.dk1) } });
  const rows = [
    [hdr("System"), hdr("Reads Bangla"), hdr("Knows Taka"), hdr("Works offline"), hdr("Wearable")],
    [{ text: "OrCam MyEye", options: { fontSize: 15 } }, cell("No"), cell("No"), cell("Yes"), cell("Yes")],
    [{ text: "Google Glass scene analysis", options: { fontSize: 15 } }, cell("No"), cell("No"), cell("No"), cell("Yes")],
    [{ text: "AIris (2024)", options: { fontSize: 15 } }, cell("No"), cell("No"), cell("No"), cell("Yes")],
    [{ text: "Raspberry Pi reading aids", options: { fontSize: 15 } }, cell("No"), cell("No"), cell("Partly"), cell("Partly")],
    [{ text: "SARA (ours)", options: { fontSize: 15, bold: true, color: HEX.dk2 } }, cell("Yes", true), cell("Yes", true), cell("Yes", true), cell("Yes", true)],
  ];
  s.addTable(rows, { x: 0.6, y: 1.5, w: 8.2, colW: [3.0, 1.3, 1.3, 1.3, 1.3], rowH: 0.62, border: { type: "solid", pt: 0.75, color: "C9D6D0" }, fill: { color: HEX.lt1 }, fontFace: "Calibri" });
  card(s, 9.2, 1.5, 3.5, 3.72, "gap card");
  s.addText("The gap", body("", { x: 9.45, y: 1.7, w: 3.0, h: 0.5, fontSize: 20, bold: true, color: C.text2 }));
  s.addText("No device combines Bangla, Taka and offline use. Counterfeit studies never check whether test notes are copies of prints seen in training.",
    body("", { x: 9.45, y: 2.25, w: 3.0, h: 2.8, fontSize: 16 }));
  s.addText("Online: the online description mode needs the internet; the other four modes do not.", body("", { x: 0.6, y: 5.55, w: 8.2, h: 0.5, fontSize: 12, color: C.accent5 }));
  s.addNotes("Commercial glasses such as OrCam work offline but not in Bangla and not with Taka. Glass-based research systems send pictures to a server. Raspberry Pi reading aids are cheap but mostly English and use online speech. The second gap is in evaluation, which we come back to in the counterfeit part.");

  // ======================================================== 6. architecture
  pres.addSection({ title: "System" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "System" });
  s.addText("How the glass works", { placeholder: "title" });
  const box = (x, y, w, h, t, fill, txt, name) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: name });
    s.addText(t, { x, y, w, h, fontSize: 15, bold: true, color: txt, align: "center", valign: "middle", margin: 4, isTextBox: true });
  };
  // a line may not have a negative height: an upward arrow is drawn downward and flipped
  const arrow = (x1, y1, x2, y2, n) => s.addShape(pres.shapes.LINE, {
    x: x1, y: Math.min(y1, y2), w: x2 - x1, h: Math.max(Math.abs(y2 - y1), 0.001), flipV: y2 < y1,
    line: { color: HEX.accent5, width: 2, endArrowType: "triangle" }, objectName: n });
  badge(s, "cam", 0.75, 1.75, 0.85, HEX.accent4, "camera");
  s.addText("Camera\n640×480", body("", { x: 0.4, y: 2.68, w: 1.6, h: 0.7, fontSize: 14, align: "center" }));
  badge(s, "touch", 0.75, 4.25, 0.85, HEX.accent4, "buttons");
  s.addText("5 buttons", body("", { x: 0.4, y: 5.18, w: 1.6, h: 0.4, fontSize: 14, align: "center" }));
  box(2.5, 2.4, 2.4, 2.0, "Raspberry Pi 5\nmode manager", HEX.dk2, HEX.lt1, "pi box");
  arrow(1.8, 2.2, 2.5, 3.0, "arrow camera");
  arrow(1.8, 4.7, 2.5, 3.8, "arrow buttons");
  const mb = [["Text: EasyOCR + main-block rule", HEX.lt2], ["Objects: YOLOv8", HEX.lt2], ["Currency: YOLOv8s + safe check", HEX.lt2], ["Emotion: YuNet + EfficientNet", HEX.lt2], ["Online: vision model (optional)", "E6E9E8"]];
  mb.forEach(([t, f], i) => {
    const y = 1.45 + i * 0.95;
    box(5.6, y, 3.9, 0.75, t, f, HEX.dk2, "mode box " + (i + 1));
    arrow(4.9, 3.4, 5.6, y + 0.37, "arrow mode " + (i + 1));
    arrow(9.5, y + 0.37, 10.25, 3.4, "arrow out " + (i + 1));
  });
  badge(s, "voice", 10.4, 2.65, 1.0, HEX.accent1, "speech");
  s.addText("Bangla / English speech\n+ vibration", body("", { x: 10.0, y: 3.75, w: 2.8, h: 0.8, fontSize: 15, align: "center", bold: true, color: C.text2 }));
  s.addText("All models run on the Pi's CPU. Every event is written to a small CSV field log for later study.",
    body("", { x: 0.6, y: 6.35, w: 12.1, h: 0.4, fontSize: 14, color: C.accent5 }));
  s.addNotes("The camera frame and the button presses go to a mode manager on the Raspberry Pi. The active mode runs its model, and the result is spoken in Bangla or English, with a vibration for a detected note. The heavy work runs only when the user presses ACTION, except the object scan.");

  // ======================================================== 7. prototype
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "System" });
  s.addText("From design to a first prototype", { placeholder: "title" });
  photo(s, "proto_side.jpg", 0.6, 1.45, 3.0, 3.9, "Prototype side");
  photo(s, "proto_top.jpg", 3.8, 1.45, 3.0, 3.9, "Prototype top");
  photo(s, "worn_100.jpg", 7.0, 1.45, 3.0, 3.9, "Prototype worn with note");
  s.addText([
    { text: "Camera above the bridge of a 3D-printed frame", options: { bullet: true, breakLine: true } },
    { text: "Raspberry Pi 5 with fan tied to the right temple", options: { bullet: true, breakLine: true } },
    { text: "Ribbon cable loops from camera to board", options: { bullet: true, breakLine: true } },
    { text: "Next: board, buttons and battery in a pocket enclosure", options: { bullet: true } },
  ], body("", { x: 10.3, y: 1.5, w: 2.5, h: 3.9, fontSize: 15, paraSpaceAfter: 10 }));
  s.addText("The first prototype is heavy on one side and the fan sits by the ear. It is enough to test the software on a person.",
    body("", { x: 0.6, y: 5.6, w: 9.4, h: 0.8, fontSize: 15, italic: true, color: C.accent1 }));
  s.addNotes("This is the first prototype. It was the quickest way to get a working device on a person's head. It also shows why the design calls for a separate pocket enclosure: the board on one temple makes it heavy on one side.");

  // ======================================================== 8. Taka detection
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Taka detection: 22,333 training images, no hand labels", { placeholder: "title" });
  s.addImage({ path: FIG + "fig_compositing.jpg", x: 0.6, y: 1.5, w: 7.6, h: 7.6 * 466 / 1600, objectName: "Compositing pipeline" });
  s.addText("5,073 note photos are pasted onto real COCO pictures with random angle, scale, light and blur. The program writes the box labels itself.",
    body("", { x: 0.6, y: 3.9, w: 7.6, h: 1.2, fontSize: 16 }));
  const stats = [["0.995", "mAP@0.5 on 2,282 held-out composites", HEX.accent1], ["91.5%", "correct on 1,536 photos taken by others", HEX.accent1], ["0.992", "mAP@0.5 on 566 pictures with new backgrounds", HEX.accent1]];
  stats.forEach(([v, l, col], i) => {
    const y = 1.45 + i * 1.75;
    card(s, 8.7, y, 4.0, 1.55, "stat card " + (i + 1));
    s.addText(v, { x: 8.9, y: y + 0.1, w: 3.6, h: 0.8, fontSize: 40, bold: true, color: col, margin: 0, isTextBox: true, fontFace: "Cambria" });
    s.addText(l, body("", { x: 8.9, y: y + 0.9, w: 3.6, h: 0.55, fontSize: 14 }));
  });
  s.addNotes("Public Taka images are clean close-ups, so a model trained on them fails on a live camera. We pasted the notes onto real photographs and let the program write the labels. The detector scores 0.995 mAP on held-out composites and names 91.5 percent of independent photographs correctly.");

  // ======================================================== 9. live webcam
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Live test with real notes", { placeholder: "title" });
  const live = ["live_50.jpg", "live_200.jpg", "live_100.jpg", "live_10.jpg", "live_1000_500.jpg", "live_three.jpg"];
  live.forEach((f, i) => photo(s, f, 0.6 + (i % 3) * 2.85, 1.45 + Math.floor(i / 3) * 2.6, 2.7, 2.45, "Live " + f));
  s.addText([
    { text: "10, 50, 100, 200, 500 and 1,000 Taka named at 0.63–0.89 confidence", options: { bullet: true, breakLine: true } },
    { text: "Works on worn and folded notes and with two or three notes at once", options: { bullet: true, breakLine: true } },
    { text: "Boxes are loose and can overlap when notes are close", options: { bullet: true } },
  ], body("", { x: 9.3, y: 1.5, w: 3.4, h: 3.6, fontSize: 16, paraSpaceAfter: 12 }));
  s.addText("Informal sessions with a laptop webcam; no error rate was counted.", body("", { x: 9.3, y: 5.3, w: 3.4, h: 0.8, fontSize: 13, italic: true, color: C.accent5 }));
  s.addNotes("Here the detector runs live on a laptop webcam with our own notes, including worn and folded ones. These are illustrations, not a measurement: we did not count an error rate in these sessions.");

  // ======================================================== 10. view count
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Counterfeit check: one view is enough with prefix training", { placeholder: "title" });
  s.addChart(pres.charts.LINE, [
    { name: "Fixed six-view training", labels: ["1", "2", "3", "4", "5", "6"], values: [71.2, 73.1, 85.7, 90.5, 90.5, 98.1] },
    { name: "Prefix training (ours)", labels: ["1", "2", "3", "4", "5", "6"], values: [98.2, 98.4, 98.6, 98.6, 98.7, 98.7] },
  ], {
    x: 0.6, y: 1.45, w: 7.8, h: 5.1, chartColors: [HEX.accent2, HEX.accent1], lineSize: 3, lineDataSymbolSize: 9,
    showTitle: true, title: "Test accuracy (%) by number of views, mean of 3 seeds", titleFontSize: 14, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
    showLegend: true, legendPos: "b", legendFontSize: 13, legendFontFace: "+mn-lt",
    valAxisMinVal: 60, valAxisMaxVal: 100, valAxisLabelFontSize: 12, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    catAxisLabelColor: HEX.accent5, valAxisLabelColor: HEX.accent5, valGridLine: { color: "DDE5E1", size: 0.5 }, catGridLine: { style: "none" },
    showCatAxisTitle: true, catAxisTitle: "Views captured", catAxisTitleFontSize: 12, catAxisTitleColor: HEX.accent5,
  });
  s.addText("71.2% → 98.2%", { x: 8.9, y: 1.6, w: 3.9, h: 0.9, fontSize: 32, bold: true, color: C.accent1, margin: 0, isTextBox: true, fontFace: "Cambria" });
  s.addText("accuracy with a single view, JaalTaka, 3 seeds", body("", { x: 8.9, y: 2.5, w: 3.9, h: 0.5, fontSize: 14 }));
  s.addText([
    { text: "A blind user cannot promise six views", options: { bullet: true, breakLine: true } },
    { text: "Training on random first-k views removes the collapse", options: { bullet: true, breakLine: true } },
    { text: "No loss at six views; stable across seeds", options: { bullet: true } },
  ], body("", { x: 8.9, y: 3.3, w: 3.9, h: 2.6, fontSize: 16, paraSpaceAfter: 10 }));
  s.addNotes("A model trained only with all six views of a note collapses when it sees one view: 71 percent. Training on random prefixes of the views keeps it at about 98 percent from one to six views. The controlled experiment shows prefix sampling alone is responsible.");

  // ======================================================== 11. unseen prints
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Unseen counterfeit prints: a stricter test", { placeholder: "title" });
  card(s, 0.6, 1.45, 4.6, 4.9, "serial card");
  s.addText("279 of 322", { x: 0.85, y: 1.65, w: 4.1, h: 0.8, fontSize: 40, bold: true, color: C.accent2, margin: 0, isTextBox: true, fontFace: "Cambria" });
  s.addText("readable 500 Taka counterfeits carry the same serial number", body("", { x: 0.85, y: 2.45, w: 4.1, h: 0.7, fontSize: 15 }));
  s.addText("The usual split puts copies of one print in training and test. A serial lookup alone scores 90.2%. We built a split where every test counterfeit is a new print.",
    body("", { x: 0.85, y: 3.35, w: 4.1, h: 2.8, fontSize: 16 }));
  s.addChart(pres.charts.BAR, [
    { name: "Accuracy at 6 views (%)", labels: ["Frozen ResNet-50", "Prefix network", "Fine-tuned ResNet-50", "Prefix + watermark"], values: [83.8, 89.3, 94.4, 95.0] },
  ], {
    x: 5.6, y: 1.45, w: 7.1, h: 4.9, barDir: "bar", chartColors: [HEX.accent5, HEX.accent4, HEX.accent6, HEX.accent1],
    showTitle: true, title: "Accuracy on unseen prints, 6 views (%), mean of 3 seeds", titleFontSize: 14, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
    showValue: true, dataLabelFormatCode: "0.0", dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
    valAxisMinVal: 70, valAxisMaxVal: 100, valAxisHidden: true, catAxisLabelFontSize: 13, catAxisLabelColor: HEX.dk1, catAxisLabelFontFace: "+mn-lt",
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
  });
  s.addText("Adding the watermark raises accuracy to 95%, the same on all three seeds, level with a fully fine-tuned ResNet-50.",
    body("", { x: 0.6, y: 6.45, w: 12.1, h: 0.45, fontSize: 14, italic: true, color: C.accent1 }));
  s.addNotes("We found that most counterfeit notes in JaalTaka are copies of a few prints, so the usual test measures recognition of known reprints. On truly unseen prints our network drops from about 98 to about 89 percent. Adding the banknote watermark, located by a small learned model, brings it to 95 percent, the same as a properly fine-tuned ResNet-50.");

  // ======================================================== 11b. DMWR watermark method
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("DMWR: a tiny, physics-based watermark checker", { placeholder: "title" });
  s.addImage({ path: FIG + "fig_dmwr_steps.png", x: 0.6, y: 1.35, w: 7.6, h: 7.6 * 960 / 1800, objectName: "DMWR steps" });
  card(s, 0.6, 5.55, 7.6, 1.3, "equation card");
  s.addText([
    { text: "1. Remove the lamp:  Q = I / (G∗I)", options: { breakLine: true } },
    { text: "2. Genuine portrait of the note value:  T = median of Q over genuine training notes", options: { breakLine: true } },
    { text: "3. Residual R = Q − T  →  two small CNN streams + 4 summary numbers  →  \"watermark clear?\"" },
  ], body("", { x: 0.8, y: 5.65, w: 7.3, h: 1.1, fontSize: 14, color: C.text2 }));
  const dm = [["91.5%", "on unseen counterfeit prints (3 seeds, ±0.3)", HEX.accent1],
              ["47,818", "parameters, 0.21 MB: 46× smaller than MobileNetV2", HEX.accent1],
              ["p ≥ 0.375", "no significant difference from MobileNetV2 (92.9%)", HEX.accent1]];
  dm.forEach(([v, l, col], i) => {
    const y = 1.35 + i * 1.85;
    card(s, 8.7, y, 4.0, 1.65, "dmwr card " + (i + 1));
    s.addText(v, { x: 8.9, y: y + 0.1, w: 3.6, h: 0.8, fontSize: 36, bold: true, color: col, margin: 0, isTextBox: true, fontFace: "Cambria" });
    s.addText(l, body("", { x: 8.9, y: y + 0.9, w: 3.6, h: 0.65, fontSize: 14 }));
  });
  s.addNotes("This is our new watermark method, the denomination-matched watermark residual. A back-lit photo is the lamp times the paper's transmittance, so dividing by a blurred copy removes the lamp. We build a genuine portrait of each note value from genuine training notes only, subtract it, and classify what is left with two very small networks and four summary numbers. With under fifty thousand parameters it reaches 91.5 percent on counterfeit prints it has never seen, statistically level with a MobileNetV2 that is 46 times larger. In the counterfeit example, the printed portrait stands out in the residual.");

  // ======================================================== 12. safety
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Safe before decisive: the glass never says \"counterfeit\"", { placeholder: "title" });
  s.addText("0 / 88", { x: 0.6, y: 1.6, w: 5.0, h: 1.5, fontSize: 80, bold: true, color: C.accent1, margin: 0, isTextBox: true, fontFace: "Cambria" });
  s.addText("test counterfeits passed as \"likely genuine\"", body("", { x: 0.6, y: 3.1, w: 5.0, h: 0.5, fontSize: 18 }));
  s.addText("Threshold = highest score of any validation counterfeit, so a new counterfeit passes with probability 1/(n+1) = 1.1%.",
    body("", { x: 0.6, y: 3.9, w: 5.0, h: 1.4, fontSize: 15, color: C.accent5 }));
  box(6.3, 1.6, 2.6, 1.0, "Score above threshold?", HEX.dk2, HEX.lt1, "decision box");
  box(9.6, 1.1, 3.1, 0.9, "\"Likely genuine\"", HEX.accent1, HEX.lt1, "yes box");
  box(9.6, 2.3, 3.1, 0.9, "\"Check by hand\"", HEX.accent3, HEX.dk1, "no box");
  arrow(8.9, 2.0, 9.6, 1.55, "arrow yes");
  arrow(8.9, 2.2, 9.6, 2.75, "arrow no");
  s.addText("yes", body("", { x: 9.0, y: 1.35, w: 0.6, h: 0.3, fontSize: 12, color: C.accent5 }));
  s.addText("no", body("", { x: 9.0, y: 2.55, w: 0.6, h: 0.3, fontSize: 12, color: C.accent5 }));
  card(s, 6.3, 3.75, 6.4, 2.6, "cost card");
  badge(s, "shield", 6.55, 4.05, 0.8, HEX.accent1, "cost");
  s.addText("When it is not sure", body("", { x: 7.55, y: 4.0, w: 4.9, h: 0.45, fontSize: 20, bold: true, color: C.text2 }));
  s.addText("57.5% of genuine notes are confirmed from close-up views. Otherwise the glass asks for a quick check by hand, and the guided watermark check gives a second opinion.",
    body("", { x: 7.55, y: 4.5, w: 4.9, h: 1.7, fontSize: 15 }));
  s.addNotes("Because a blind user cannot verify the answer, the glass never says a note is counterfeit. It says likely genuine only above a threshold set from validation data, otherwise it asks for a check by hand. No test counterfeit passed. When the glass is not sure, it asks for a check by hand.");

  // ======================================================== 13. OCR
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Bangla reading: error cut by three quarters", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [
    { name: "Before", labels: ["Overall", "Bangla", "English", "Signs in photos"], values: [9.3, 11.9, 6.6, 10.7] },
    { name: "Pipeline v2", labels: ["Overall", "Bangla", "English", "Signs in photos"], values: [2.4, 3.2, 1.6, 2.2] },
  ], {
    x: 0.6, y: 1.45, w: 7.4, h: 5.0, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent5, HEX.accent1],
    showTitle: true, title: "Character error rate (%) on the held-out test set", titleFontSize: 14, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
    showValue: true, dataLabelFormatCode: "0.0", dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, catAxisLabelFontSize: 13, catAxisLabelColor: HEX.dk1, catAxisLabelFontFace: "+mn-lt",
    showLegend: true, legendPos: "b", legendFontSize: 13, legendFontFace: "+mn-lt",
  });
  s.addText("What helped", body("", { x: 8.5, y: 1.5, w: 4.2, h: 0.45, fontSize: 20, bold: true, color: C.accent1 }));
  s.addText([
    { text: "Reading only the main block of text", options: { bullet: true, breakLine: true } },
    { text: "Tuned detector settings, CLAHE", options: { bullet: true, breakLine: true } },
    { text: "Simple clean-up rules", options: { bullet: true } },
  ], body("", { x: 8.5, y: 2.0, w: 4.2, h: 1.5, fontSize: 15, paraSpaceAfter: 6 }));
  s.addText("What did not", body("", { x: 8.5, y: 3.7, w: 4.2, h: 0.45, fontSize: 20, bold: true, color: C.accent2 }));
  s.addText([
    { text: "Binarisation, multi-scale reading", options: { bullet: true, breakLine: true } },
    { text: "Tesseract, TrOCR, PaddleOCR", options: { bullet: true, breakLine: true } },
    { text: "Dictionaries and language models", options: { bullet: true } },
  ], body("", { x: 8.5, y: 4.2, w: 4.2, h: 1.5, fontSize: 15, paraSpaceAfter: 6 }));
  s.addText("Benchmark with correctly shaped Bangla; every choice made on validation data.", body("", { x: 8.5, y: 5.85, w: 4.2, h: 0.6, fontSize: 13, italic: true, color: C.accent5 }));
  s.addNotes("We built a test set with correctly shaped Bangla and separate phrases and fonts for validation and test. The biggest gain came from reading only the main block of text and ignoring background text. Overall error fell from 9.3 to 2.4 percent, Bangla from 11.9 to 3.2.");

  // ======================================================== 14. Pi 5
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Running on the Raspberry Pi 5", { placeholder: "title" });
  const pi = [
    ["money", "327 ms", "currency check, median", HEX.accent1],
    ["shield", "50 ms", "watermark check", HEX.accent1],
    ["obj", "207 ms", "object scan", HEX.accent1],
    ["text", "18.7 s", "text reading: too slow", HEX.accent2],
    ["thermo", "77.1 °C", "peak, no throttling in 30 min", HEX.accent4],
    ["chip", "1.48 GB", "peak memory", HEX.accent4],
  ];
  pi.forEach(([k, v, l, col], i) => {
    const colN = i % 3, row = Math.floor(i / 3);
    const x = 0.6 + colN * 4.1, y = 1.5 + row * 2.45;
    card(s, x, y, 3.9, 2.2, "pi card " + (i + 1));
    badge(s, k, x + 0.25, y + 0.3, 0.8, col, "pi " + (i + 1));
    s.addText(v, { x: x + 1.25, y: y + 0.25, w: 2.5, h: 0.9, fontSize: 36, bold: true, color: col, margin: 0, isTextBox: true, fontFace: "Cambria" });
    s.addText(l, body("", { x: x + 0.25, y: y + 1.35, w: 3.4, h: 0.6, fontSize: 15 }));
  });
  s.addNotes("Measured on a Raspberry Pi 5 with 8 GB. The currency check takes about a third of a second, and with the spoken answer under three seconds. It ran 30 minutes with no thermal throttling. Text reading takes about 19 seconds, which is our clearest speed problem.");

  // ======================================================== 15. testing with users
  pres.addSection({ title: "Testing" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Testing" });
  s.addText("Testing the worn glass", { placeholder: "title" });
  photo(s, "worn_200.jpg", 0.6, 1.45, 3.2, 4.8, "Worn with 200 Taka");
  photo(s, "worn_front.jpg", 4.0, 1.45, 3.2, 4.8, "Worn front");
  badge(s, "warn", 7.7, 1.5, 0.75, HEX.accent2, "field test");
  s.addText("First field test", body("", { x: 8.65, y: 1.55, w: 4.0, h: 0.45, fontSize: 20, bold: true, color: C.text2 }));
  s.addText("With the glass camera, currency was unreliable and Bangla text was not read. We suspect the 640×480 fixed-focus picture; the laptop webcam works.",
    body("", { x: 8.65, y: 2.05, w: 4.05, h: 1.8, fontSize: 15 }));
  badge(s, "group", 7.7, 4.1, 0.75, HEX.accent1, "trials");
  s.addText("Trials and survey", body("", { x: 8.65, y: 4.15, w: 4.0, h: 0.45, fontSize: 20, bold: true, color: C.text2 }));
  s.addText("Sessions with a visually impaired participant have started. A Bangla voice questionnaire records answers; results will follow.",
    body("", { x: 8.65, y: 4.65, w: 4.05, h: 1.6, fontSize: 15 }));
  s.addNotes("The laboratory numbers do not yet describe the worn glass. In the first field test with the glass camera, currency recognition was unreliable and Bangla text was not read. We suspect the low-resolution fixed-focus camera, because the same software works with a laptop webcam. User trials have started and we will add the survey results.");

  // ======================================================== 17. next steps
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
  s.addText("Ongoing work and next steps", { placeholder: "title" });
  const next = [
    ["Tune the glass camera", "Higher resolution or autofocus; test set taken through the glass"],
    ["Complete the user study", "Trials started; 8–12 participants: task time, accuracy, SUS and NASA-TLX"],
    ["Collect more prints", "About 100 unseen counterfeit prints with a bank or the police"],
    ["Faster reading", "Line-by-line page reading; lighter text detector"],
    ["Pocket enclosure", "Move board, buttons and battery off the frame; measure battery"],
  ];
  next.forEach(([h, t], i) => {
    const x = 0.6 + i * 2.5;
    numBadge(s, i + 1, x + 0.85, 1.7, 0.75, HEX.accent1);
    if (i < 4) s.addShape(pres.shapes.LINE, { x: x + 1.7, y: 2.07, w: 1.55, h: 0, line: { color: "C9D6D0", width: 2 }, objectName: "step line " + (i + 1) });
    s.addText(h, body("", { x, y: 2.7, w: 2.3, h: 0.8, fontSize: 18, bold: true, color: C.text2, align: "center" }));
    s.addText(t, body("", { x, y: 3.55, w: 2.3, h: 1.8, fontSize: 15, align: "center" }));
  });
  s.addNotes("Our next steps: tune the glass camera, complete the user study that has already started, collect more counterfeit prints, make page reading faster, and build a lighter frame with a pocket enclosure.");

  // ======================================================== 18. conclusion
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Conclusion" });
  s.addText("Thank you", { placeholder: "title" });
  s.addText("Questions are welcome", { placeholder: "body" });
  s.addText([
    { text: "An offline glass that speaks Bangla, reads text, names objects and recognises Taka on a Raspberry Pi 5", options: { bullet: true, breakLine: true } },
    { text: "New methods: prefix training, a safe verdict, and the DMWR watermark checker", options: { bullet: true, breakLine: true } },
    { text: "Next: tune the glass camera and complete the user trials", options: { bullet: true } },
  ], { x: 0.7, y: 4.5, w: 7.6, h: 2.0, fontSize: 16, color: C.background1, margin: 0, isTextBox: true, paraSpaceAfter: 8 });
  photo(s, "worn_200.jpg", 8.75, 0.7, 4.0, 6.1, "Closing photo");
  s.addNotes("To sum up: we built an assistive glass that speaks Bangla, that runs on a Raspberry Pi 5, tested every part on held-out data, and added a new, very small watermark checker. Thank you. We are happy to take questions.");

  fs.mkdirSync(OUT_DIR, { recursive: true });
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
