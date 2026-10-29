# Thesis update for 29 October 2026: the third Pi run and the fixes from the field log (sec_pi_third_run.tex, folded
# into resultandchallenges.tex and removed after this ran),
# plus the sentences elsewhere that those results change. Run once; each change is checked.
import pathlib

d = pathlib.Path(__file__).resolve().parent.parent / "thesis report" / "My report"
sec = (d / "sec_pi_third_run.tex").read_text(encoding="utf-8")
sec = sec.replace(r"(Section~\ref{sec:currency_results})",
                  r"(Section~\ref{sec:handheld}; \texttt{results/closeup\_ft/handheld\_int8.json})")


def edit(name, pairs):
    p = d / name
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert s.count(a) == 1, (name, a[:60])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")


edit("resultandchallenges.tex", [
    ("\\section{Corrections to Earlier Figures}", sec + "\\section{Corrections to Earlier Figures}"),
    ("Its speed on the Pi has not been measured yet.",
     "Its speed on the Pi was measured on 29 October: 2.5--3.0~s per pass (Section~\\ref{sec:pi5_third})."),
    ("the guided watermark check and a larger set of whole-note photographs are the way to confirm more genuine notes.",
     "the guided watermark check and a larger set of whole-note photographs are the way to confirm more genuine notes. "
     "With the C270 webcam all six guided watermark checks timed out, so that step is off until it is calibrated on "
     "photographs from this camera (Section~\\ref{sec:pi5_third})."),
    ("but has not been tested on a real counterfeit note, on real printed paper under the lamp, or through the glass camera.",
     "but has not been tested on a real counterfeit note, on real printed paper under the lamp, or through the glass camera. "
     "Currency mode now answers with it when the picture is taken under the lamp (Section~\\ref{sec:pi5_third})."),
])
edit("conclusion.tex", [
    ("ran for 30~minutes without throttling.",
     "ran for 30~minutes without throttling. In a third run on 29 October text reading was about three times faster "
     "(a 3-line sign in 5.4~s), but the 30-minute run reached 87.0~$^\\circ$C with the soft temperature limit active "
     "in 55\\% of the samples; the board's surface read 57.8--62.3~$^\\circ$C on a thermometer in normal use. The field "
     "log of that day led to the same Taka detector on the Pi as on the laptop, a spoken UV answer under the lamp and "
     "reading the sharpest frame of a press."),
])
edit("abstract.tex", [
    ("and without thermal throttling.",
     "and without thermal throttling; in a third run text reading took 5.4~s for a 3-line sign, three times faster, "
     "while continuous load drove the processor to its soft temperature limit."),
])
print("ok")
