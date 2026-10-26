"""build3.js: architecture slide as a horizontal pipeline (inputs -> Pi 5 -> AI modules -> audio)."""
from pathlib import Path
p = Path(__file__).parent / "build3.js"
s = p.read_text(encoding="utf-8")
a = s.index('  circleIcon(s, "cam", 0.8, 2.0, 1.0, HEX.accent1, "Camera");')
b = s.index('  sources(s, `Sources: ${cite("craft")} Baek et al. 2019;')
NEW = r'''  const stages = [
    ["cam", HEX.accent1, "Camera, buttons, voice", ["Camera 640×480", "5 buttons on the frame", "Voice: \"glass, …\""]],
    ["chip", HEX.dk2, "Raspberry Pi 5", ["Mode manager", "All models on the CPU", "No internet needed"]],
    ["cube", HEX.accent3, "AI modules", ["Text: CRAFT + EasyOCR " + cite("craft", "easyocr"), "Objects: YOLOv8s, COCO " + cite("yolo", "coco"),
      "Taka: YOLOv8s + safe checks", "Emotion: YuNet + EfficientNet " + cite("rafdb"), "Online: cloud model (optional)"]],
    ["speak", HEX.accent1, "Audio + vibration", ["Bangla or English speech " + cite("piper", "espeak"), "Vibration for a note"]],
  ];
  stages.forEach(([k, col, h, lines], i) => {
    const x = 0.6 + i * 3.15;
    card(s, x, 1.45, 2.75, 4.95, h + " card");
    circleIcon(s, k, x + 0.78, 1.7, 1.2, col, h);
    s.addText(h, { x: x + 0.1, y: 3.05, w: 2.55, h: 0.8, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text2, align: "center", valign: "top", isTextBox: true, margin: 0, objectName: h + " heading" });
    s.addText(lines.map((t, j) => ({ text: t, options: { breakLine: j < lines.length - 1 } })),
      { x: x + 0.15, y: 3.9, w: 2.45, h: 2.35, fontSize: i === 2 ? 12 : 14, color: C.text1, align: "center", valign: "top", paraSpaceAfter: 6, isTextBox: true, margin: 0, objectName: h + " details" });
    if (i < 3) s.addShape(pres.shapes.CHEVRON, { x: x + 2.83, y: 2.05, w: 0.28, h: 0.55, fill: { color: HEX.accent6 }, line: { color: HEX.accent6 }, objectName: "Arrow " + (i + 1) });
  });
'''
s = s[:a] + NEW + s[b:]
p.write_text(s, encoding="utf-8")
print("arch2 ok")
