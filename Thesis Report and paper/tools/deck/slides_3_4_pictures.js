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

