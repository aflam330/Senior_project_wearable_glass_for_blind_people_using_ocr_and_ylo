"""First polish pass on build2.js after visual QA (25 October 2026)."""
from pathlib import Path
p = Path(__file__).parent / "build2.js"
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


# references in order of first citation; ascending numbers inside brackets
order = ["who", "orcam", "glass", "airis", "craft", "easyocr", "yolo", "coco", "rafdb", "piper", "espeak", "banglataka", "nstu", "jaal",
         "mnv2", "mlt", "vosk", "esc50"]
rep('const NUM = Object.fromEntries(REFS.map((r, i) => [r[0], i + 1]));',
    'const ORDER = ' + str(order).replace("'", '"') + ';\n'
    'REFS.sort((a, b) => ORDER.indexOf(a[0]) - ORDER.indexOf(b[0]));\n'
    'const NUM = Object.fromEntries(REFS.map((r, i) => [r[0], i + 1]));')
rep('const cite = (...keys) => "[" + keys.map((k) => NUM[k]).join(", ") + "]";',
    'const cite = (...keys) => "[" + keys.map((k) => NUM[k]).sort((a, b) => a - b).join(", ") + "]";')
# titles left-aligned
rep('fontFace: "Cambria", fontSize: 30, bold: true, color: C.text2, valign: "middle", margin: 0 }, text: "" } },',
    'fontFace: "Cambria", fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },')
# slide 2 title
rep('s.addText("A normal day: three questions a blind person cannot answer alone", { placeholder: "title" });',
    's.addText("Three questions a blind person cannot answer alone", { placeholder: "title" });')
# slide 9: four photos across the full width
rep('''  photo(s, AS + "test_10.jpg", 0.6, 1.45, 2.6, 4.62, "Prototype version 1");
  s.addText("Version 1: board and fan on one temple", { x: 0.6, y: 6.12, w: 2.6, h: 0.45, fontSize: 12, color: C.accent5, isTextBox: true, margin: 0, objectName: "V1 caption" });''',
    '''  photo(s, AS + "test_10.jpg", 0.6, 1.45, 2.84, 4.62, "Prototype version 1");
  s.addText("Version 1: board and fan on one temple", { x: 0.6, y: 6.12, w: 2.84, h: 0.45, fontSize: 12, color: C.accent5, isTextBox: true, margin: 0, objectName: "V1 caption" });''')
rep('''    photo(s, AS + f, 3.55 + i * 2.75, 1.45, 2.6, 4.62, cap);
    s.addText(cap, { x: 3.55 + i * 2.75, y: 6.12, w: 2.6, h: 0.45, fontSize: 12, color: C.accent5, isTextBox: true, margin: 0, objectName: "V2 caption " + (i + 1) });''',
    '''    photo(s, AS + f, 3.69 + i * 3.09, 1.45, 2.84, 4.62, cap);
    s.addText(cap, { x: 3.69 + i * 3.09, y: 6.12, w: 2.84, h: 0.45, fontSize: 12, color: C.accent5, isTextBox: true, margin: 0, objectName: "V2 caption " + (i + 1) });''')
rep('s.addText("Version 2", { shape: pres.shapes.ROUNDED_RECTANGLE, x: 3.7, y: 1.6,',
    's.addText("Version 2", { shape: pres.shapes.ROUNDED_RECTANGLE, x: 3.84, y: 1.6,')
# slide 12 caption
rep('s.addText("Informal sessions: laptop webcam (top) and the first glass prototype (bottom); no error rate was counted.",',
    's.addText("Informal sessions with a laptop webcam (top row, bottom right) and the first glass prototype (bottom left, centre); no error rate was counted.",')
# slide 16 card
rep('card(s, 10.0, 1.45, 2.7, 3.3, "Safety stat");', 'card(s, 10.0, 1.45, 2.7, 1.95, "Safety stat");')
# links slide: show each URL under its name
rep('''    s.addText(items.map(([t, u], i) => ({ text: t, options: { hyperlink: { url: u, tooltip: u }, color: HEX.accent1, breakLine: i < items.length - 1 } })),
      { x: x + 0.25, y: 2.35, w: 3.4, h: 4.0, fontSize: 14, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 9, objectName: h + " links" });''',
    '''    const runs = [];
    items.forEach(([t, u], i) => {
      runs.push({ text: t, options: { hyperlink: { url: u, tooltip: u }, color: HEX.accent1, bold: true, fontSize: 13, breakLine: true } });
      runs.push({ text: u.replace(/^https?:\\/\\//, ""), options: { color: HEX.accent5, fontSize: 10, breakLine: i < items.length - 1, paraSpaceAfter: 7 } });
    });
    s.addText(runs, { x: x + 0.25, y: 2.35, w: 3.4, h: 4.05, isTextBox: true, margin: 0, valign: "top", fit: "none", objectName: h + " links" });''')
p.write_text(s, encoding="utf-8")
print("polish 1 ok")
