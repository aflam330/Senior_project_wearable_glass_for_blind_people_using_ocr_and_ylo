"""build3.js: larger architecture slide."""
from pathlib import Path
p = Path(__file__).parent / "build3.js"
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('''(all on the CPU)", { shape: pres.shapes.ROUNDED_RECTANGLE, x: 3.0, y: 2.9, w: 2.5, h: 1.7, rectRadius: 0.1,
    fill: { color: HEX.dk2 }, line: { color: HEX.dk2 }, color: "FFFFFF", fontSize: 16,''',
    '''(all on the CPU)", { shape: pres.shapes.ROUNDED_RECTANGLE, x: 2.95, y: 2.75, w: 2.65, h: 2.0, rectRadius: 0.1,
    fill: { color: HEX.dk2 }, line: { color: HEX.dk2 }, color: "FFFFFF", fontSize: 20,''')
rep('''x: 6.3, y, w: 4.2, h: 0.82, rectRadius: 0.08, fill: { color: i === 4 ? "F3F5F4" : HEX.lt2 }, line: { color: i === 4 ? "F3F5F4" : HEX.lt2 },
        fontSize: 13,''', '''x: 6.3, y, w: 4.3, h: 0.88, rectRadius: 0.08, fill: { color: i === 4 ? "F3F5F4" : HEX.lt2 }, line: { color: i === 4 ? "F3F5F4" : HEX.lt2 },
        fontSize: 15,''')
rep('connect(s, 5.5, 3.75, 6.3, y + 0.41, h + " link", true);', 'connect(s, 5.6, 3.75, 6.3, y + 0.44, h + " link", true);')
rep('''circleIcon(s, "speak", 11.2, 3.0, 1.0, HEX.accent1, "Speech");''', '''circleIcon(s, "speak", 11.1, 2.85, 1.35, HEX.accent1, "Speech");''')
rep('''{ x: 10.75, y: 4.1, w: 1.95, h: 0.95, fontSize: 13,''', '''{ x: 10.7, y: 4.35, w: 2.1, h: 1.1, fontSize: 15,''')
rep('connect(s, 10.5, 3.5, 11.2, 3.5, "Output link", true);', 'connect(s, 10.6, 3.52, 11.1, 3.52, "Output link", true);')
rep('connect(s, x, y, 3.0, 3.75, "Input " + (i + 1), true));', 'connect(s, x, y, 2.95, 3.75, "Input " + (i + 1), true));')
for a in ['{ x: 0.45, y: 3.05, w: 1.7, h: 0.35, fontSize: 13', '{ x: 0.45, y: 4.8, w: 1.7, h: 0.35, fontSize: 13', '{ x: 0.35, y: 6.22, w: 1.9, h: 0.35, fontSize: 13']:
    rep(a, a.replace("fontSize: 13", "fontSize: 15"))
p.write_text(s, encoding="utf-8")
print("arch ok")
