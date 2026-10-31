  // ================================================================ 17b. object detection
  pres.addSection({ title: "Objects and emotion" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Objects and emotion" });
  s.addText("Object Detection", { placeholder: "title" });
  circleIcon(s, "cube", 0.6, 1.45, 0.8, HEX.accent1, "Objects");
  s.addText("How it works", { x: 1.6, y: 1.6, w: 4.5, h: 0.5, fontFace: "Arial", fontSize: 20, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "Objects heading" });
  s.addText([
    { text: "Every 1.5 s a camera frame goes to YOLOv8s " + cite("yolo", "coco") + ": 80 everyday object types", options: { bullet: true, breakLine: true } },
    { text: "Only new objects are spoken; the same object is not repeated within 4 s", options: { bullet: true, breakLine: true } },
    { text: "ACTION announces at once", options: { bullet: true, breakLine: true } },
    { text: "Animals need 65% confidence: in dim rooms, clutter was called \"cat\" or \"dog\"", options: { bullet: true } },
  ], { x: 0.6, y: 2.4, w: 6.0, h: 2.3, fontSize: 16, color: C.text1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 8, objectName: "Objects list" });
  bubble(s, "সামনে আছে: মানুষ, মোবাইল ফোন", 0.6, 4.85, 3.5, 0.75, HEX.accent1, "FFFFFF", "Objects example", BN);
  s.addText("spoken by the glass in a test session", { x: 0.6, y: 5.7, w: 3.5, h: 0.3, fontSize: 12, italic: true, color: C.accent5, isTextBox: true, margin: 0, objectName: "Objects example note" });
  stat(s, 4.45, 4.75, 2.3, "462 ms", "one scan on the Raspberry Pi 5", "Objects Pi");
  s.addChart(pres.charts.BAR, [{ name: "mAP@0.5", labels: ["YOLOv8n", "YOLOv8s (used)", "YOLOv8m"], values: [0.453, 0.551, 0.607] }],
    { x: 7.0, y: 1.45, w: 5.7, h: 4.3, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent5, HEX.accent1, HEX.accent5], showValue: true,
      dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0, valAxisMaxVal: 0.7, valAxisLabelFormatCode: "0.0", showLegend: false,
      showTitle: true, title: "mAP@0.5 on all 5,000 COCO validation photos", ...chartFrame, objectName: "Objects chart" });
  s.addText("YOLOv8s balances accuracy and speed on the Pi. Through the glass's own path it scores 0.521.",
    { x: 7.0, y: 5.85, w: 5.7, h: 0.6, fontSize: 13, italic: true, color: C.accent2, isTextBox: true, margin: 0, valign: "top", objectName: "Objects note" });
  sources(s, `Detector ${cite("yolo")} Ultralytics YOLOv8, trained on ${cite("coco")} COCO. Results: savior_glass/results/object_coco_val2017_full.json, object_shrink_vs_full.json; Pi time from benchmark_report/.`);
  notes(s, "The object mode looks at a camera frame every one and a half seconds and runs YOLOv8s, trained on the 80 everyday object types of COCO. It speaks only objects it has not just named, and does not repeat the same object within four seconds; the ACTION button announces at once. Animals need a higher confidence, because in dim rooms clutter was called a cat or a dog. On all 5,000 COCO validation photos YOLOv8s scores 0.551 mAP at 0.5, between the smaller YOLOv8n and the larger YOLOv8m; through the glass's own path, with the camera frame as it is, it scores 0.521. One scan takes 462 milliseconds on the Raspberry Pi 5.");

  // ================================================================ 17c. facial expression
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Objects and emotion" });
  s.addText("Facial Expression Recognition", { placeholder: "title" });
  circleIcon(s, "smile", 0.6, 1.45, 0.8, HEX.accent1, "Emotion");
  s.addText("How it works", { x: 1.6, y: 1.6, w: 4.5, h: 0.5, fontFace: "Arial", fontSize: 20, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "Emotion heading" });
  s.addText([
    { text: "Finds the largest face in the picture", options: { bullet: true, breakLine: true } },
    { text: "EfficientNet-V2-S trained on RAF-DB " + cite("rafdb") + " (12,271 faces), blended 3 : 1 with FER+", options: { bullet: true, breakLine: true } },
    { text: "Speaks one of 7 expressions in Bangla", options: { bullet: true } },
  ], { x: 0.6, y: 2.4, w: 6.0, h: 1.9, fontSize: 16, color: C.text1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 8, objectName: "Emotion list" });
  bubble(s, "সামনের মানুষটিকে খুশি দেখাচ্ছে", 0.6, 3.85, 3.7, 0.75, HEX.accent1, "FFFFFF", "Emotion example", BN);
  s.addText("\"The person in front looks happy\"", { x: 0.6, y: 4.7, w: 3.7, h: 0.3, fontSize: 12, italic: true, color: C.accent5, isTextBox: true, margin: 0, objectName: "Emotion example note" });
  stat(s, 0.6, 5.1, 3.0, "86.5%", "correct on the RAF-DB test (3,068 faces)", "Emotion accuracy");
  stat(s, 3.9, 5.1, 2.9, "536 ms", "one answer on the Pi 5, face in view", "Emotion Pi");
  s.addChart(pres.charts.BAR, [{ name: "Correct", labels: ["Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust", "Neutral"],
    values: [92.5, 83.9, 80.9, 87.2, 55.4, 58.8, 89.0] }],
    { x: 7.0, y: 1.45, w: 5.7, h: 4.3, barDir: "col", barGapWidthPct: 45, chartColors: [HEX.accent1], showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: "0\"%\"", valAxisMinVal: 0, valAxisMaxVal: 100, valAxisLabelFormatCode: "0\"%\"", showLegend: false,
      showTitle: true, title: "Correct per expression, RAF-DB official test", ...chartFrame, objectName: "Emotion chart" });
  s.addText("Fear and disgust are hardest (fewest examples). Measured for the network; the blend on the glass was not tested separately.",
    { x: 7.0, y: 5.85, w: 5.7, h: 0.6, fontSize: 13, italic: true, color: C.accent2, isTextBox: true, margin: 0, valign: "top", objectName: "Emotion note" });
  sources(s, `Data ${cite("rafdb")} RAF-DB, official train / test split. Results: savior_glass/results/emotion_rafdb.json; Pi time from benchmark_report/.`);
  notes(s, "The emotion mode finds the largest face in the picture and classifies its expression with an EfficientNet-V2-S network trained on 12,271 RAF-DB faces, blended three to one with the FER+ model. It then says, for example, the person in front looks happy. On the 3,068 faces of the official RAF-DB test set the network is right 86.5 percent of the time, with a macro F1 of 0.795. Happy, neutral and surprise are recognised best; fear and disgust are hardest, because they have the fewest examples. The blend that the glass uses was not tested separately. One answer takes 536 milliseconds on the Raspberry Pi 5 with a face in view.");

