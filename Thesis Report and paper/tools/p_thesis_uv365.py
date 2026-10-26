"""Thesis: sources for 365 nm UV inspection and for non-fluorescent banknote paper; the glass's lamp is 360-395 nm.
Sources read in full (27 October 2026): Mutanen et al., IS&T PICS 2003, pp. 421-424; ECB quick guide 2010;
US patent 6,918,482 B2 (Giesecke & Devrient)."""
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "thesis report" / "My report"
BIB = r"""
@inproceedings{mutanen2003luminescent,
  author    = {Mutanen, Jarkko and Jaaskelainen, Timo and Parkkinen, Jussi P. S.},
  title     = {Luminescent Security Properties of Banknotes},
  booktitle = {IS\&T's 2003 PICS Conference},
  pages     = {421--424},
  publisher = {Society for Imaging Science and Technology},
  year      = {2003}
}

@misc{ecb2010quickguide,
  author       = {{European Central Bank}},
  title        = {Quick Guide to the Security Features of Euro Banknotes},
  year         = {2010},
  howpublished = {ISBN 978-92-899-0147-5},
  url          = {https://www.ecb.europa.eu/euro/pdf/material/Quick_Guide_EN_Specimen.pdf}
}

@misc{thierauf2005patent,
  author       = {Thierauf, Klaus},
  title        = {Device and Method for Verifying the Authenticity of Banknotes},
  year         = {2005},
  howpublished = {US Patent 6,918,482 B2, Giesecke \& Devrient GmbH},
  url          = {https://patents.google.com/patent/US6918482}
}
"""


def patch(name, pairs):
    p = D / name
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert s.count(a) == 1, (name, a[:80], s.count(a))
        s = s.replace(a, b)
    assert not any(ord(c) < 32 and c not in "\n\t\r" for c in s), name
    p.write_text(s, encoding="utf-8")


bib = D / "ref.bib"
b = bib.read_text(encoding="utf-8")
if "mutanen2003luminescent" not in b:
    bib.write_text(b.rstrip("\n") + "\n" + BIB, encoding="utf-8")

patch("LiteratureReview.tex", [
    (r"""Under a UV lamp, typically emitting near 365~nm, genuine notes show security fibres and inks that fluoresce in particular colours, while counterfeits printed on ordinary paper either lack these marks or glow evenly because such paper usually contains optical brighteners.""",
     r"""The most widely used fluorescent security features are excited by long-wave UV light at 365~nm \cite{mutanen2003luminescent}. Under such a lamp, banknote paper itself does not glow, while fibres embedded in it light up in colour \cite{ecb2010quickguide,mutanen2003luminescent}. Most commercial papers, by contrast, contain optical brighteners that glow strongly bluish under UV, so a counterfeit printed on ordinary paper either lacks the fibres or glows evenly \cite{mutanen2003luminescent,thierauf2005patent}. The check is not infallible: a genuine note washed with detergent can take up brighteners and glow as well \cite{mutanen2003luminescent}."""),
])
patch("methodology.tex", [
    (r"""As an additional, optional check, the UV lamp beside the camera (the blue light in Figure~\ref{fig:prototype2}) is switched on after the note has been named, and the fibres inside the note box are counted.""",
     r"""As an additional, optional check, the UV lamp beside the camera (the blue light in Figure~\ref{fig:prototype2}) is switched on after the note has been named, and the fibres inside the note box are counted. The lamp emits between 360 and 395~nm, a band that includes the 365~nm commonly used for banknote inspection \cite{mutanen2003luminescent}."""),
    (r"""3D-printed frame with the camera housing and a small ultraviolet lamp (blue light) at the front, held by a head strap""",
     r"""3D-printed frame with the camera housing and a small ultraviolet lamp (360--395~nm, blue light) at the front, held by a head strap"""),
])
print("thesis uv365 ok")
