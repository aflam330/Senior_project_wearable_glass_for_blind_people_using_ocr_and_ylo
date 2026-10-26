"""build3.js: Pi heat headline; done / in progress / limitation on slide 21; future-work roadmap; conclusion without the infographic."""
from pathlib import Path
p = Path(__file__).parent / "build3.js"
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


# 20: heat as an engineering finding
rep('''  s.addText("Heat", { x: 10.25, y: 1.85, w: 2.2, h: 0.5, fontFace: "Cambria", fontSize: 20, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "Heat heading" });
  s.addText([
    { text: "Board idled at 71 °C; peak 88.7 °C, throttled in 76% of samples", options: { bullet: true, breakLine: true } },
    { text: "Thermal governor slows background scans when hot", options: { bullet: true, breakLine: true } },
    { text: "Text reading is the slowest task", options: { bullet: true } },
  ], { x: 9.25, y: 2.75, w: 3.25, h: 3.2, fontSize: 14, color: "E4EFEA", isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 8, objectName: "Heat list" });''',
    '''  s.addText("Key finding: heat", { x: 10.2, y: 1.85, w: 2.4, h: 0.5, fontFace: "Cambria", fontSize: 19, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "Heat heading" });
  s.addText("88.7 °C", { x: 9.25, y: 2.65, w: 3.3, h: 0.8, fontFace: "Cambria", fontSize: 44, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "Heat peak" });
  s.addText("peak temperature; the Pi throttled in 76% of samples (idle 71 °C)", { x: 9.25, y: 3.45, w: 3.25, h: 0.9, fontSize: 15, color: "E4EFEA", isTextBox: true, margin: 0, valign: "top", objectName: "Heat label" });
  s.addText("Response: a thermal governor slows background scans when hot; better cooling is future work.",
    { x: 9.25, y: 4.55, w: 3.25, h: 1.3, fontSize: 14, italic: true, color: "CFE2DA", isTextBox: true, margin: 0, valign: "top", objectName: "Heat response" });''')

# 21: done / in progress / limitation
rep('''  card(s, 8.45, 1.45, 4.25, 2.25, "Field test card");
  s.addText("Limitation: the glass camera", { x: 8.7, y: 1.6, w: 3.8, h: 0.45, fontFace: "Cambria", fontSize: 19, bold: true, color: C.accent2, isTextBox: true, margin: 0, objectName: "Field heading" });
  s.addText("In the first field test, currency was unreliable and Bangla text was not read (640×480, fixed focus).",
    { x: 8.7, y: 2.1, w: 3.8, h: 1.5, fontSize: 16, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "Field text" });
  card(s, 8.45, 3.9, 4.25, 2.25, "Trials card");
  s.addText("User trials: in progress", { x: 8.7, y: 4.05, w: 3.8, h: 0.45, fontFace: "Cambria", fontSize: 19, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "Trials heading" });
  s.addText("Sessions with a visually impaired participant have started. No user study is complete yet.",
    { x: 8.7, y: 4.55, w: 3.8, h: 1.5, fontSize: 16, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "Trials text" });''',
    '''  const status = [
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
  });''')
rep('''  s.addText("Prototype version 2 on team members: fit, weight and button reach are checked before user trials.",
    { x: 0.6, y: 5.85, w: 7.5, h: 0.7, fontSize: 13,''', '''  s.addText("Prototype version 2 on team members: fit, weight and button reach are checked before user trials.",
    { x: 0.6, y: 5.85, w: 7.5, h: 0.7, fontSize: 14,''')
rep('''  s.addText("Testing the worn glass", { placeholder: "title" });''', '''  s.addText("Testing the worn glass: where the project stands", { placeholder: "title" });''')

# 22: future work
rep('''  s.addText("Limitations and next steps", { placeholder: "title" });''', '''  s.addText("Future work: the roadmap", { placeholder: "title" });''')

# 23: conclusion without the infographic
rep('''  s.addImage({ path: AS + "infographic_sara.jpg", x: 0.6, y: 1.4, w: 7.8, h: 5.2, objectName: "SARA before and after" });
  s.addText("Illustration", { x: 0.6, y: 6.62, w: 7.8, h: 0.3, fontSize: 10, italic: true, color: C.accent5, isTextBox: true, margin: 0, objectName: "Illustration note" });''',
    '''  photo(s, AS + "v2_collage.jpg", 0.6, 1.4, 4.1, 5.2, "Second prototype");''')
rep('''  concl.forEach(([k, h, t], i) => {
    const y = 1.4 + i * 1.78;
    card(s, 8.75, y, 3.95, 1.6, h + " card");
    circleIcon(s, k, 8.95, y + 0.2, 0.62, i === 2 ? HEX.accent3 : HEX.accent1, h);
    s.addText(h, { x: 9.72, y: y + 0.22, w: 2.85, h: 0.5, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: h + " heading" });
    s.addText(t, { x: 8.95, y: y + 0.88, w: 3.6, h: 0.68, fontSize: 13, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: h + " text" });
  });''',
    '''  concl.forEach(([k, h, t], i) => {
    const y = 1.4 + i * 1.78;
    card(s, 5.1, y, 7.6, 1.6, h + " card");
    circleIcon(s, k, 5.35, y + 0.4, 0.8, i === 2 ? HEX.accent3 : HEX.accent1, h);
    s.addText(h, { x: 6.4, y: y + 0.2, w: 6.1, h: 0.5, fontFace: "Cambria", fontSize: 22, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: h + " heading" });
    s.addText(t, { x: 6.4, y: y + 0.75, w: 6.1, h: 0.75, fontSize: 17, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: h + " text" });
  });''')
rep('''"Conclusion: from challenges to independence"''', '''"Conclusion"''')
p.write_text(s, encoding="utf-8")
print("closing ok")
