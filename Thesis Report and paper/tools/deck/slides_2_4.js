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

  // ================================================================ 3. problem statement: flowchart
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
  s.addText("Problem Statement", { placeholder: "title" });
  card(s, 0.6, 1.75, 1.95, 2.65, "User node", "FFFFFF");
  s.addImage({ data: I.blind, x: 1.13, y: 1.95, w: 0.9, h: 0.9, objectName: "User icon" });
  s.addText(hb("Visually impaired user", "in Bangladesh"), { x: 0.65, y: 2.95, w: 1.85, h: 1.3, fontSize: 15, color: HEX.dk1, align: "center", valign: "top", isTextBox: true, margin: 0, objectName: "User label" });
  const needs = [
    ["Money", "Which note was handed back? Is it genuine?", "Asks the shopkeeper"],
    ["Reading", "What does a medicine label, sign or letter say?", "Asks family or a pharmacist"],
    ["Surroundings", "What or who is ahead?", "Asks a stranger"],
  ];
  needs.forEach(([h, t, who], i) => {
    const y = 1.45 + i * 1.15;
    node(s, hb(h, t, 16), 3.0, y, 3.0, 0.95, HEX.accent1, "FFFFFF", h + " need", 13);
    node(s, who, 6.45, y, 2.35, 0.95, "FFFFFF", HEX.dk1, h + " helper", 15);
    arrow(s, 2.55, 3.075, 3.0, y + 0.475, h + " arrow 1");
    arrow(s, 6.0, y + 0.475, 6.45, y + 0.475, h + " arrow 2");
    arrow(s, 8.8, y + 0.475, 9.25, 3.075, h + " arrow 3");
  });
  s.addText([
    { text: "An answer they cannot verify", options: { bold: true, fontSize: 20, breakLine: true, paraSpaceAfter: 10 } },
    { text: "A wrong note or a fake note can go unnoticed", options: { bullet: { indent: 14 }, breakLine: true, paraSpaceAfter: 6 } },
    { text: "A label or letter can be misread", options: { bullet: { indent: 14 }, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Every task waits for someone else", options: { bullet: { indent: 14 } } },
  ], { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x: 9.25, y: 1.45, w: 3.45, h: 3.25, fill: { color: HEX.dk2 }, line: { color: HEX.dk2 },
    color: "FFFFFF", fontSize: 15, valign: "middle", isTextBox: true, margin: 0.15, objectName: "Outcome" });
  s.addText("Existing aids do not close this gap", { x: 0.6, y: 4.92, w: 8.0, h: 0.36, fontFace: "Arial", fontSize: 17, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "Aids heading" });
  const aids = [
    ["OrCam MyEye " + cite("orcam"), "Works offline, but Bangla is not among its listed languages; no Taka notes"],
    ["Research glasses " + cite("glass", "airis"), "Google Glass based systems and AIris send pictures to a server"],
    ["Pi reading aids " + cite("rpiaid"), "Low cost, but mostly English-only"],
    ["Counterfeit-note studies " + cite("uddin", "pathak", "uvink", "jaal", "jaal26"), "Not brought to a wearable a blind user can operate"],
  ];
  aids.forEach(([h, t], i) => node(s, hb(h, t, 14), 0.6 + i * 2.3, 5.35, 2.15, 1.2, "FFFFFF", HEX.dk1, "Aid " + (i + 1), 12));
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 9.8, y: 5.7, w: 0.45, h: 0.5, fill: { color: HEX.accent5 }, line: { color: HEX.accent5 }, objectName: "Aids to gap" });
  node(s, hb("The gap", "No reviewed device is offline, speaks Bangla and names Taka notes", 16), 10.35, 5.35, 2.35, 1.2, HEX.accent2, "FFFFFF", "Gap", 13);
  notes(s, "This flowchart shows the problem. A visually impaired user has three everyday questions: money, reading and the surroundings. Each one is answered by another person, the shopkeeper, a family member or pharmacist, or a stranger, so the user ends up with an answer they cannot verify. Existing aids do not close this gap. OrCam MyEye works offline but does not list Bangla among its languages and does not name Taka notes. The research glasses send pictures to a server. Low-cost Raspberry Pi reading aids are mostly English-only. Studies on Bangladeshi counterfeit notes, and the JaalTaka benchmark, have not been brought to a wearable. So no reviewed device is offline, speaks Bangla and names Taka notes.");

  // ================================================================ 4. proposed solution: flowchart
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
  s.addText("Proposed Solution: SARA", { placeholder: "title" });
  s.addImage({ path: AS + "v2_action_stack.jpg", x: 0.6, y: 1.45, w: 2.6, h: 2.6 * 1816 / 1600, objectName: "Prototype in a staged exchange" });
  s.addText("Our prototype in a staged shop exchange: receiving a note, holding it up to the camera", { x: 0.6, y: 4.5, w: 2.6, h: 0.9, fontSize: 12, italic: true, color: C.accent5, isTextBox: true, margin: 0, valign: "top", objectName: "Prototype caption" });
  const inputs = [
    ["Ask", "Button press or voice: \"Glass, check money\" " + cite("vosk")],
    ["Capture", "The glass camera takes a picture"],
    ["Process offline", "On the Raspberry Pi 5, no internet " + cite("rpi5")],
  ];
  inputs.forEach(([h, t], i) => {
    const x = 3.55 + i * 3.2;
    node(s, hb(h, t, 16), x, 1.45, 2.75, 1.0, HEX.accent1, "FFFFFF", "Input " + (i + 1), 13);
    if (i < 2) arrow(s, x + 2.75, 1.95, x + 3.2, 1.95, "Input arrow " + (i + 1));
  });
  const flowModes = [
    ["Taka recognition", "Names the note " + cite("yolo")],
    ["Authenticity check", "Watermark (DMWR) and UV security fibres"],
    ["Text reading", "Bangla and English " + cite("craft", "easyocr")],
    ["Objects and people", "Names what is ahead " + cite("yolo", "coco")],
  ];
  const mx = (i) => 3.55 + i * 2.38;
  arrow(s, 11.325, 2.45, 11.325, 2.7, "Pi to modes", false);
  arrow(s, mx(0) + 1.0, 2.7, mx(3) + 1.0, 2.7, "Mode bus", false);
  flowModes.forEach(([h, t], i) => {
    node(s, hb(h, t, 15), mx(i), 2.95, 2.0, 1.45, "FFFFFF", HEX.dk1, h, 13);
    arrow(s, mx(i) + 1.0, 2.7, mx(i) + 1.0, 2.95, h + " arrow in");
    arrow(s, mx(i) + 1.0, 4.4, mx(i) + 1.0, 4.7, h + " arrow out");
  });
  node(s, hb("Safe verdict", "\"Likely genuine\" or \"check by hand\"; it never says \"counterfeit\"", 15), mx(0), 4.7, 4.38, 0.95, HEX.accent3, "FFFFFF", "Safe verdict", 13);
  node(s, hb("Spoken answer", "Bangla or English speech " + cite("piper") + " and a short vibration", 15), mx(2), 4.7, 4.38, 0.95, HEX.dk2, "FFFFFF", "Spoken answer", 13);
  arrow(s, mx(0) + 4.38, 5.175, mx(2), 5.175, "Verdict to answer");
  arrow(s, mx(2) + 2.19, 5.65, mx(2) + 2.19, 5.85, "Answer to user");
  s.addText([{ text: "পাঁচশত টাকার নোট", options: { fontFace: BN, fontSize: 22, bold: true, color: HEX.accent1 } },
    { text: "   \"500 Taka note\": the user knows without asking anyone", options: { fontSize: 15, color: HEX.dk1 } }],
    { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, fill: { color: HEX.lt2 }, line: { color: HEX.accent6 }, align: "center", valign: "middle", x: mx(0), y: 5.85, w: 9.14, h: 0.7, margin: 8, isTextBox: true, objectName: "Spoken example" });
  s.addText([{ text: "Offline: ", options: { bold: true, color: HEX.accent1 } }, { text: "text, objects, Taka, emotion.  " }, { text: "Optional online: ", options: { bold: true, color: HEX.accent5 } }, { text: "scene description only." }],
    { x: 0.6, y: 6.65, w: 12.1, h: 0.3, fontSize: 13, color: C.text1, align: "center", isTextBox: true, margin: 0, objectName: "Offline note" });
  notes(s, "This flowchart is the same moment with SARA. The user presses a button or says glass, check money. The camera takes a picture, and the Raspberry Pi 5 processes it on the device without the internet. Depending on the mode, it recognises the Taka note, checks the watermark and the UV security fibres, reads Bangla or English text, or names objects and people ahead. For money, a safe verdict policy says likely genuine or check by hand, and never says counterfeit. The answer is spoken in Bangla or English with a short vibration, for example five hundred Taka note, so the user knows without asking anyone.");

