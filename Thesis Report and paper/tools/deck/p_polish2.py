"""Deck: UV lamp named correctly; four Bangladeshi counterfeit papers cited (26 October 2026)."""
from pathlib import Path
p = Path(__file__).parent / "build2.js"
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


rep('''  ["esc50", "Piczak, K. J. ESC: dataset for environmental sound classification. ACM Multimedia, 2015.", "https://github.com/karolpiczak/ESC-50"],
];''', '''  ["esc50", "Piczak, K. J. ESC: dataset for environmental sound classification. ACM Multimedia, 2015.", "https://github.com/karolpiczak/ESC-50"],
  ["jaal26", "Piyas, B. R. C. et al. JaalTaka: a benchmark dataset for detecting counterfeit Bangladeshi banknotes. Data in Brief 64, 112377, 2026.", "https://doi.org/10.1016/j.dib.2025.112377"],
  ["uddin", "Uddin, M. S., Das, P. P., Roney, M. S. A. Image-based approach for the detection of counterfeit banknotes of Bangladesh. ICIEV, 2016.", "https://doi.org/10.1109/ICIEV.2016.7760162"],
  ["pathak", "Pathak, A. et al. Enhanced counterfeit detection of Bangladesh currency through convolutional neural networks. IJIRCST 12(2), 2024.", "https://doi.org/10.55524/ijircst.2024.12.2.2"],
  ["uvink", "Chowdhury, A. A. et al. Evaluation of deep learning models on UV ink: a fake money detection scheme with RPN. ICCA (ACM), 2022.", "https://doi.org/10.1145/3542954.3542965"],
];''')
rep('"jaal", "mnv2", "mlt", "vosk", "esc50"];', '"jaal", "jaal26", "uddin", "pathak", "mnv2", "uvink", "mlt", "vosk", "esc50"];')
# slide 9 caption
rep('["1_20.jpg", "Front: camera and LED on the bridge"]', '["1_20.jpg", "Front: camera and UV lamp (blue light)"]')
# slide 13: the JaalTaka article as well as the dataset
rep('sources(s, `Data: ${cite("jaal")} Piyas et al., JaalTaka benchmark, Mendeley Data 2025 (1,390 notes, six views each).`);',
    'sources(s, `Data: ${cite("jaal", "jaal26")} Piyas et al., JaalTaka dataset and article, 2025-2026 (1,390 notes, six views each; counterfeits seized by the Rapid Action Battalion).`);')
# slide 14: earlier Taka counterfeit work
rep('sources(s, `Data: ${cite("jaal")} JaalTaka; serial-number audit and print-disjoint split are ours.`);',
    'sources(s, `Data: ${cite("jaal")} JaalTaka; serial-number audit and print-disjoint split are ours. Earlier Taka counterfeit work: ${cite("uddin")} SVM on security features; ${cite("pathak")} CNNs; ${cite("uvink")} UV flecks + RPN.`);')
# slide 17: the lamp is on the glass; prior UV work
rep('s.addText("UV fibre check: the security fibres light up", { placeholder: "title" });',
    's.addText("UV fibre check: the glass carries its own UV lamp", { placeholder: "title" });')
rep('''  s.addText("A prototype, off by default: no real counterfeit has been photographed under the lamp yet.",''',
    '''  sources(s, `Prior work on Taka under UV light: ${cite("uvink")} Chowdhury et al., ICCA 2022 (96.3% with an RPN on UV images). Fibre check (ENFR) and photographs are ours.`);
  s.addText("A prototype, off by default: no real counterfeit has been photographed under the lamp yet.",''')
p.write_text(s, encoding="utf-8")
print("polish 2 ok")
