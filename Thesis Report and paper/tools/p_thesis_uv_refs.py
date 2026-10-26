"""Thesis: the blue light is the UV lamp; four Bangladeshi counterfeit-detection papers cited (26 October 2026).
Bibliographic data checked against Crossref (DOIs) and the IJIRCST article page."""
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "thesis report" / "My report"

BIB = r"""
@inproceedings{chowdhury2022uvink,
  author    = {Chowdhury, Anjir Ahmed and Das, Argho and Karmaker, Debajyoti and Hoque, Khadija Kubra Shahjalal},
  title     = {Evaluation of Deep Learning Models on {UV} Ink: A Fake Money Detection Scheme with {RPN}},
  booktitle = {Proceedings of the 2nd International Conference on Computing Advancements (ICCA 2022)},
  pages     = {67--74},
  publisher = {ACM},
  year      = {2022},
  doi       = {10.1145/3542954.3542965}
}

@article{piyas2026jaaltaka,
  author  = {Piyas, Bibhas Roy Chowdhury and Tisha, Fatama Jannat and Rahman, Sadia and Islam, Shahrin and Preenon, Bijoy Roy Chowdhury},
  title   = {{JaalTaka}: A Benchmark Dataset for Detecting Counterfeit {Bangladeshi} Banknotes},
  journal = {Data in Brief},
  volume  = {64},
  pages   = {112377},
  year    = {2026},
  doi     = {10.1016/j.dib.2025.112377}
}

@article{pathak2024counterfeit,
  author  = {Pathak, Abhijit and Chakraborty, Arnab and Rahaman, Minhajur and Rafa, Taiyaba Shadaka and Nayema, Ummay},
  title   = {Enhanced Counterfeit Detection of {Bangladesh} Currency through Convolutional Neural Networks: A Deep Learning Approach},
  journal = {International Journal of Innovative Research in Computer Science and Technology (IJIRCST)},
  volume  = {12},
  number  = {2},
  pages   = {10--20},
  year    = {2024},
  doi     = {10.55524/ijircst.2024.12.2.2}
}

@inproceedings{uddin2016counterfeit,
  author    = {Uddin, Mohammad Shorif and Das, Pronaya Prosun and Roney, Md. Shamim Ahmed},
  title     = {Image-Based Approach for the Detection of Counterfeit Banknotes of {Bangladesh}},
  booktitle = {5th International Conference on Informatics, Electronics and Vision (ICIEV)},
  pages     = {1067--1072},
  publisher = {IEEE},
  year      = {2016},
  doi       = {10.1109/ICIEV.2016.7760162}
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
if "chowdhury2022uvink" not in b:
    bib.write_text(b.rstrip("\n") + "\n" + BIB, encoding="utf-8")

patch("LiteratureReview.tex", [
    (r"""Counterfeit detection is much less studied in assistive systems. Bank equipment uses ultraviolet light, magnetic ink and infrared sensors, which are not available on a glass. With a visible-light camera, the security thread, hologram strip, watermark and micro-printing must be inspected from close views, and one view is often not enough. The JaalTaka dataset provides six close-up views of genuine and counterfeit 500 and 1,000 Taka notes \cite{jaaltaka}. This motivates the multi-view authenticator in this work.""",
     r"""Counterfeit detection is much less studied in assistive systems. Bank equipment checks ultraviolet (UV) features, magnetic ink and the response of the note to infrared light. Under a UV lamp, typically emitting near 365~nm, genuine notes show security fibres and inks that fluoresce in particular colours, while counterfeits printed on ordinary paper either lack these marks or glow evenly because such paper usually contains optical brighteners. For Bangladeshi notes, Uddin et al.\ extracted the watermark, micro-printing and hologram from note images and classified them with a support vector machine \cite{uddin2016counterfeit}; Pathak et al.\ trained convolutional networks on photographs of genuine and counterfeit Taka \cite{pathak2024counterfeit}; and Chowdhury et al.\ used the randomly placed colour flecks that appear only under UV light, reaching 96.3\% accuracy with a region proposal network followed by CNN classification \cite{chowdhury2022uvink}. With a visible-light camera, the security thread, hologram strip, watermark and micro-printing must be inspected from close views, and one view is often not enough. The JaalTaka dataset, built from genuine notes supplied by government banks and counterfeits seized by the Rapid Action Battalion, provides six close-up views of genuine and counterfeit 500 and 1,000 Taka notes \cite{jaaltaka,piyas2026jaaltaka}. This motivates the multi-view authenticator in this work, and the UV fibre check that a small UV lamp on the glass makes possible."""),
])

patch("methodology.tex", [
    (r"""A genuine note therefore \emph{has} UV marks; a note without them is the suspicious one. As an additional, optional check, a UV lamp beside the camera is switched on after the note has been named, and the fibres inside the note box are counted.""",
     r"""A genuine note therefore \emph{has} UV marks; a note without them is the suspicious one. Chowdhury et al.\ used the same marks, photographed under UV light, to detect counterfeit Taka with deep networks \cite{chowdhury2022uvink}. As an additional, optional check, the UV lamp beside the camera (the blue light in Figure~\ref{fig:prototype2}) is switched on after the note has been named, and the fibres inside the note box are counted."""),
    (r"""The camera now sits in a small closed housing on the front of the frame, with a blue LED beside it.""",
     r"""The camera now sits in a small closed housing on the front of the frame, with a small ultraviolet lamp beside it that lights the note for the UV fibre check (Section~\ref{sec:uv_method}); it is the blue light in the photographs."""),
    (r"""\caption{Second prototype of SARA worn by a team member: the camera housing and LED at the front of the frame (top),""",
     r"""\caption{Second prototype of SARA worn by a team member: the camera housing and the UV lamp (blue light) at the front of the frame (top),"""),
    (r"""& Spectacle frame & 3D-printed frame with the camera housing and a blue LED at the front, held by a head strap & Holds the camera where the user looks, so the glass is worn like ordinary spectacles \\ \hline""",
     r"""& Spectacle frame and UV lamp & 3D-printed frame with the camera housing and a small ultraviolet lamp (blue light) at the front, held by a head strap & Holds the camera where the user looks; the UV lamp lights the note for the UV fibre check \\ \hline"""),
])

patch("implementation.tex", [
    (r"""In the second prototype (Figure~\ref{fig:prototype2}) the camera sits in a closed housing with an LED beside it,""",
     r"""In the second prototype (Figure~\ref{fig:prototype2}) the camera sits in a closed housing with a UV lamp beside it for the fibre check,"""),
])
print("thesis uv + refs ok")
