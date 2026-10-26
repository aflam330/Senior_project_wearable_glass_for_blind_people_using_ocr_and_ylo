"""build3.js: slides 13-18 - one large takeaway each, test split and verdict meanings, simulated UV result labelled."""
from pathlib import Path
p = Path(__file__).parent / "build3.js"
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


# 13: test split
rep('''  s.addText("Training on random first views: one view is enough.",
    { x: 8.9, y: 4.6, w: 3.5, h: 1.5, fontSize: 18,''',
    '''  s.addText("Held-out JaalTaka test notes, never used for training or for any choice", { x: 8.9, y: 5.55, w: 3.5, h: 0.7, fontSize: 12, italic: true, color: "CFE2DA", isTextBox: true, margin: 0, valign: "top", objectName: "Split note" });
  s.addText("Training on random first views: one view is enough.",
    { x: 8.9, y: 4.5, w: 3.5, h: 1.0, fontSize: 18,''')

# 14: the main result large
rep('''  stat(s, 8.6, 1.6, 4.1, "279 of 322", "readable 500 Taka counterfeits carry the same serial number", "Serial");
  s.addText([{ text: "Our split: every test counterfeit is a new print.", options: { breakLine: true } },
    { text: "The watermark check lifts accuracy to 95.0%.", options: { bold: true, color: HEX.accent1 } }],
    { x: 8.6, y: 3.3, w: 4.1, h: 2.7, fontSize: 19, color: C.text1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 12, objectName: "Unseen text" });''',
    '''  card(s, 8.6, 1.45, 4.1, 2.55, "Main result", C.text2);
  s.addText("95.0%", { x: 8.9, y: 1.65, w: 3.5, h: 1.0, fontFace: "Cambria", fontSize: 54, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, objectName: "Unseen value" });
  s.addText("on counterfeit prints never seen in training, with the watermark check", { x: 8.9, y: 2.7, w: 3.5, h: 1.1, fontSize: 16, color: "E4EFEA", isTextBox: true, margin: 0, valign: "top", objectName: "Unseen label" });
  s.addText([{ text: "Why a stricter split: ", options: { bold: true } }, { text: "279 of 322 readable 500 Taka counterfeits share one serial number, so the usual split tests copies of known prints." }],
    { x: 8.6, y: 4.25, w: 4.1, h: 2.1, fontSize: 15, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "Unseen text" });''')

# 15: two numbers only
rep('''  stat(s, 8.9, 1.7, 3.6, "91.5%", "on unseen counterfeit prints (3 seeds, ±0.3)", "DMWR acc");
  stat(s, 8.9, 3.2, 3.6, "47,818", "parameters (0.21 MB): 46× smaller than MobileNetV2 " + cite("mnv2"), "DMWR params");
  stat(s, 8.9, 4.7, 3.6, "p ≥ 0.375", "no significant difference from MobileNetV2 (92.9%)", "DMWR p");''',
    '''  s.addText("91.5%", { x: 8.9, y: 1.7, w: 3.6, h: 1.0, fontFace: "Cambria", fontSize: 54, bold: true, color: C.accent1, isTextBox: true, margin: 0, objectName: "DMWR value" });
  s.addText("on unseen counterfeit prints (3 seeds, ±0.3)", { x: 8.9, y: 2.7, w: 3.6, h: 0.7, fontSize: 16, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "DMWR value label" });
  stat(s, 8.9, 3.65, 3.6, "46× smaller", "than MobileNetV2 " + cite("mnv2") + " (47,818 parameters, 0.21 MB), and as accurate: 92.9%, p ≥ 0.375", "DMWR size");''')

# 16: meanings, screening aid
rep('''  s.addText("The threshold is the highest score of any validation counterfeit. 57.5% of genuine notes are confirmed; for the rest, the glass asks for a check by hand.",
    { x: 0.6, y: 5.1, w: 12.1, h: 1.2, fontSize: 18, color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: "Safety text" });''',
    '''  s.addText([{ text: "\\"Likely genuine\\": ", options: { bold: true, color: HEX.accent1 } }, { text: "the score is higher than every counterfeit seen in validation (57.5% of genuine notes).", options: { breakLine: true } },
    { text: "\\"Check by hand\\": ", options: { bold: true, color: "B07D00" } }, { text: "the glass is not sure. It never means \\"counterfeit\\"." }],
    { x: 0.6, y: 4.75, w: 12.1, h: 1.0, fontSize: 18, color: C.text1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 6, objectName: "Safety text" });
  s.addText("A screening aid, not a guaranteed counterfeit detector", { shape: pres.shapes.ROUNDED_RECTANGLE, x: 0.6, y: 5.9, w: 12.1, h: 0.5, rectRadius: 0.08,
    fill: { color: "FFF4D6" }, line: { color: HEX.accent4 }, fontSize: 16, bold: true, color: HEX.dk1, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "Screening banner" });''')

# 17: simulated result labelled
rep('''  stat(s, 4.6, 5.1, 3.6, "88.6%", "of simulated fibre-free printed sheets rejected", "UV sheets");''',
    '''  stat(s, 4.6, 5.1, 3.6, "88.6%", "of SIMULATED fibre-free sheets rejected (not real counterfeits)", "UV sheets", C.accent5);''')
rep('''    { text: "Must still be validated on real counterfeit notes", options: { fontSize: 15 } }],''',
    '''    { text: "The rejection result is simulated; it must be validated on real counterfeit notes", options: { fontSize: 14 } }],''')

# 18: what the evaluation is
rep('''  s.addText("Character error rate, lower is better", { x: 0.6, y: 6.32, w: 8.0, h: 0.3, fontSize: 12, italic: true,''',
    '''  s.addText("Character error rate, lower is better. Measured on evaluation datasets, not yet through the glass.", { x: 0.6, y: 6.32, w: 8.0, h: 0.3, fontSize: 12, bold: true, italic: true,''')
p.write_text(s, encoding="utf-8")
print("results2 ok")
