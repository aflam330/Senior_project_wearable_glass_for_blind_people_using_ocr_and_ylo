# Thesis, slides and evidence note: the full-precision Taka detector was timed on the Pi after the change
# (benchmark_report/2026-10-29/raw/currency_current_code.json; app session benchmark_report/raw/field/20261029_213948_memo).
import pathlib

root = pathlib.Path(__file__).resolve().parents[2]
rep = root / "Thesis Report and paper" / "thesis report" / "My report"


def edit(p, pairs):
    s = p.read_bytes().decode("utf-8")
    for a, b in pairs:
        assert s.count(a) == 1, (p.name, a[:60])
        s = s.replace(a, b)
    p.write_bytes(s.encode("utf-8"))


edit(rep / "resultandchallenges.tex", [
    ("By the timings above it costs about 0.3~s more per check than the 8-bit copy; it was not yet timed on the Pi in the application.",
     "Timed on the Pi after the change (100 calls, cool board): 532~ms for the whole currency job against 422~ms with "
     "the 8-bit copy, the detector alone 483~ms against 339~ms. In a 6-minute session with the application afterwards "
     "a note was named in 5 of 8 presses (2 of 10 in the afternoon sessions), 0.81~s from press to speech."),
    ("None of these changes has yet been measured on the glass itself:",
     "Apart from that currency timing and session, none of these changes has yet been measured on the glass itself:"),
])
edit(root / "Thesis Report and paper" / "tools" / "deck" / "build4.js", [
    ("The Pi now runs the full-precision note detector: about 0.3 s more per check, not yet timed.",
     "Since then the Pi runs the full-precision note detector: 532 ms a check, a note named in 5 of 8 presses (was 2 of 10)."),
    ("Since that run the Pi uses the same full-precision note detector as the laptop, which costs about 0.3 seconds per check and has not been timed yet.",
     "Since that run the Pi uses the same full-precision note detector as the laptop. Timed afterwards it takes 532 milliseconds a check instead of 422, and in a session with the glass a note was named in 5 of 8 presses, against 2 of 10 in the afternoon."),
])
edit(root / "paper_evidence" / "PI_FIXES_2026-10-29.md", [
    ("| `best_handheld.pt` timed in the app on the Pi |",
     "| Timed after the change (`benchmark_report/2026-10-29/raw/currency_current_code.json`, 100 calls): 532 ms end to end (8-bit: 422 ms), detector 483 ms (8-bit: 339 ms). App session after it (`benchmark_report/raw/field/20261029_213948_memo`): note named in 5 of 8 presses (2 of 10 before), 0.81 s press to speech. |"),
])
print("ok")
