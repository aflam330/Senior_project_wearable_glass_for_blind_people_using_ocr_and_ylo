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

