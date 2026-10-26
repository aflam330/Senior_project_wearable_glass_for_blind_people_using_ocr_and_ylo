"""build3.js: slides 10-11 separate synthetic and real-photo results; one large takeaway each."""
from pathlib import Path
p = Path(__file__).parent / "build3.js"
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


TAG = '''  const tag = (s, text, x, y, w, fill, color, name) => s.addText(text, { shape: pres.shapes.ROUNDED_RECTANGLE, x, y, w, h: 0.36, rectRadius: 0.06,
    fill: { color: fill }, line: { color: fill }, color, fontSize: 12, bold: true, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: name });
  const notes = (s, t) => s.addNotes(t);'''
rep("  const notes = (s, t) => s.addNotes(t);", TAG)

# slide 10
rep('''  card(s, 9.2, 1.5, 3.5, 4.9, "Stats card");
  stat(s, 9.5, 1.75, 3.0, "0.995", "mAP@0.5 on 2,282 held-out composites", "mAP");
  stat(s, 9.5, 3.3, 3.0, "94.1%", "right on 1,536 photos taken by other people", "Bangla Money");
  stat(s, 9.5, 4.85, 3.0, "0", "hand-drawn boxes needed", "No labels");''',
    '''  card(s, 9.2, 1.5, 3.5, 2.3, "Synthetic card");
  tag(s, "SYNTHETIC COMPOSITES", 9.45, 1.7, 2.6, HEX.accent6, HEX.dk2, "Synthetic tag");
  stat(s, 9.45, 2.2, 3.0, "0.995", "mAP@0.5 on 2,282 held-out composites", "mAP");
  card(s, 9.2, 4.0, 3.5, 2.4, "Real card", C.text2);
  tag(s, "REAL PHOTOS", 9.45, 4.2, 1.7, HEX.accent4, HEX.dk1, "Real tag");
  s.addText("94.1%", { x: 9.45, y: 4.7, w: 3.0, h: 0.75, fontFace: "Cambria", fontSize: 40, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "Real value" });
  s.addText("right note value on 1,536 photos taken by other people", { x: 9.45, y: 5.45, w: 3.0, h: 0.8, fontSize: 14, color: "E4EFEA", isTextBox: true, margin: 0, valign: "top", objectName: "Real label" });''')
rep('''and names 94.1 percent of independent photographs correctly.");''',
    '''and, the number that matters for users, names 94.1 percent of 1,536 real photographs taken by other people correctly. The two numbers measure different things: the first is on synthetic composites, the second on real photos.");''')

# slide 11
rep('''  s.addImage({ path: REP + "fig58_handheld_closeups.png", x: 0.6, y: 1.45, w: 7.0, h: 2.8, objectName: "Synthetic close-ups" });
  s.addText([{ text: "6,000 synthetic close-ups: folded, blurred, under fingers", options: { fontSize: 19, breakLine: true } },
    { text: "Chosen on validation; the test set was read once", options: { fontSize: 14, color: HEX.accent5 } }],
    { x: 0.6, y: 4.4, w: 7.0, h: 1.0, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "Close-up text" });''',
    '''  tag(s, "TRAINING: 6,000 SYNTHETIC CLOSE-UPS", 0.6, 1.45, 3.9, HEX.accent6, HEX.dk2, "Training tag");
  s.addImage({ path: REP + "fig58_handheld_closeups.png", x: 0.6, y: 1.9, w: 5.5, h: 2.2, objectName: "Synthetic close-ups" });
  tag(s, "TEST: REAL HAND-HELD PHOTOS", 0.6, 4.3, 3.3, HEX.accent4, HEX.dk1, "Test tag");
  s.addText("11.8% → 47.5%", { x: 0.6, y: 4.75, w: 7.0, h: 0.85, fontFace: "Cambria", fontSize: 44, bold: true, color: C.accent1, isTextBox: true, margin: 0, objectName: "Hand-held takeaway" });''')
rep('''  s.addText("Trade-off: with 3 notes in view, the total is right 97.0% of the time (was 99.2%).",
    { x: 0.6, y: 5.6, w: 7.0, h: 0.8, fontSize: 14,''',
    '''  s.addText("Right note value on the laptop. Trade-off: with 3 notes in view, the total is right 97.0% (was 99.2%).",
    { x: 0.6, y: 5.65, w: 7.0, h: 0.8, fontSize: 14,''')
rep('title: "Right note value, NSTU hand-held test (1,127 photos)"', 'title: "Right note value, real hand-held photos (NSTU, 1,127)"')
p.write_text(s, encoding="utf-8")
print("results ok")
