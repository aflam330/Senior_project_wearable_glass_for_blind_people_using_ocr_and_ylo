"""Thesis: UV bands (WHO 2002 guide, read 26 October 2026: UVA 315-400, UVB 280-315, UVC 100-280 nm) and why the
360-395 nm lamp looks violet-blue."""
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "thesis report" / "My report"
BIB = r"""
@book{who2002uvi,
  author    = {{World Health Organization}},
  title     = {Global Solar {UV} Index: A Practical Guide},
  publisher = {World Health Organization},
  address   = {Geneva},
  year      = {2002},
  isbn      = {92-4-159007-6},
  url       = {https://www.who.int/publications/i/item/9241590076}
}
"""

bib = D / "ref.bib"
b = bib.read_text(encoding="utf-8")
if "who2002uvi" not in b:
    bib.write_text(b.rstrip("\n") + "\n" + BIB, encoding="utf-8")

p = D / "methodology.tex"
s = p.read_text(encoding="utf-8")
a = r"""The lamp emits between 360 and 395~nm, a band that includes the 365~nm commonly used for banknote inspection \cite{mutanen2003luminescent}."""
n = r"""Ultraviolet light is commonly divided into three bands: UV-A (315--400~nm), UV-B (280--315~nm) and UV-C (100--280~nm) \cite{who2002uvi}. Banknote inspection relies on long-wave UV-A, the least energetic of the three, and the most widely used fluorescent security features are excited at 365~nm \cite{mutanen2003luminescent}. The lamp on the glass emits between 360 and 395~nm, which lies entirely within UV-A and covers this 365~nm line. The upper end of its band sits just below the 400~nm edge of visible light, so part of its output is seen as the violet-blue glow in the photographs. The method below turns this to advantage: the visible part of the lamp's light serves as the reference against which the fluorescence of the fibres is measured."""
assert s.count(a) == 1
s = s.replace(a, n)
assert not any(ord(c) < 32 and c not in "\n\t\r" for c in s)
p.write_text(s, encoding="utf-8")
print("thesis uv bands ok")
