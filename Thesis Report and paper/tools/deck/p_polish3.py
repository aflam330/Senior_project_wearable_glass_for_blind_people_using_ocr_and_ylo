"""Deck: UV lamp 360-395 nm; Mutanen et al. 2003 cited for 365 nm inspection and non-glowing banknote paper."""
from pathlib import Path
p = Path(__file__).parent / "build2.js"
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


rep('''  ["uvink", "Chowdhury, A. A. et al. Evaluation of deep learning models on UV ink: a fake money detection scheme with RPN. ICCA (ACM), 2022.", "https://doi.org/10.1145/3542954.3542965"],
];''', '''  ["uvink", "Chowdhury, A. A. et al. Evaluation of deep learning models on UV ink: a fake money detection scheme with RPN. ICCA (ACM), 2022.", "https://doi.org/10.1145/3542954.3542965"],
  ["mutanen", "Mutanen, J., Jaaskelainen, T., Parkkinen, J. P. S. Luminescent security properties of banknotes. IS&T PICS Conference, 2003, pp. 421-424.", "https://www.imaging.org/common/uploaded%20files/pdfs/Papers/2003/PICS-0-287/8558.pdf"],
];''')
rep('"pathak", "mnv2", "uvink", "mlt", "vosk", "esc50"];', '"pathak", "mnv2", "uvink", "mutanen", "mlt", "vosk", "esc50"];')
rep('''  sources(s, `Prior work on Taka under UV light: ${cite("uvink")} Chowdhury et al., ICCA 2022 (96.3% with an RPN on UV images). Fibre check (ENFR) and photographs are ours.`);''',
    '''  sources(s, `The glass's lamp emits 360-395 nm, covering the 365 nm used for banknote checks; banknote paper stays dark while its fibres glow ${cite("mutanen")}. Prior Taka UV work: ${cite("uvink")} (96.3%). Fibre check and photos are ours.`);''')
p.write_text(s, encoding="utf-8")
print("polish 3 ok")
