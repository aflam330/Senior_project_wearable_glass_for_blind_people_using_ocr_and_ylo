"""Voice-command evaluation of 15 October into the features and results PDFs."""
from pathlib import Path
H = Path(__file__).parent


def patch(name, pairs):
    p = H / name
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert s.count(a) == 1, (name, a[:70], s.count(a))
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")


patch("make_features.py", [
    ('''    ["Voice commands — new", "Say \\"read text\\", \\"check money\\", \\"objects\\", \\"repeat\\", \\"louder\\", \\"stop\\" instead of pressing buttons",
     "Offline recogniser (Vosk) limited to 39 command phrases; ignores its own voice", "Synthetic speech: 99.1% carried out, no wrong command (Section 5)"],''',
     '''    ["Voice commands — new", "Say \\"glass, read text\\", \\"glass, check money\\", \\"glass, stop\\" ... instead of pressing buttons",
     "Offline recogniser (Vosk) limited to 39 command phrases after the wake word; ignores its own voice",
     "12 unseen synthetic voices: 100% clean, 98.9% whispered, 82.5% in noise as loud as the voice; 13 false commands in 5,040 sentences"],'''),
    ('''    "while it speaks. Tested with three synthetic voices: 116 of 117 commands carried out with no noise or light noise, "
    "113 of 117 in heavy noise, never a wrong command; 1 false command in 90 ordinary English sentences in noise. "
    "Not yet tested with people or on the Pi. The command words are English: no small offline Bangla recogniser exists.",''',
     '''    "while it speaks. Every command starts with \\"glass\\", which keeps other people's talk from becoming commands. "
    "Chosen on 8 synthetic voices and measured once on 12 others (four of them Bangla voices speaking English): 100% "
    "clean, 99.8% soft, 98.9% whispered, 82.5% with street noise as loud as the voice, 54% with people talking nearby "
    "(3% before); 13 false commands in 5,040 ordinary sentences (212 before). A command is acted on 0.86 s after the "
    "speaker stops. Not yet tested with people or on the Pi. Bangla command words were tried offline (45-62% on "
    "unseen voices) and are not used.",'''),
])
patch("make_results.py", [
    ('''    ["Voice commands (new, 14 October)", "39 command phrases spoken by 3 synthetic voices (117 commands); not people",
     "Carried out: 99.1% clean, 99.1% at 20 dB noise, 96.6% at 10 dB noise; never a wrong command"],
    ["Voice commands, false commands", "90 ordinary English sentences and 10 Bangla sentences per noise level; 60 s of noise alone",
     "English: 0 / 1 / 1 of 90 (clean / 20 dB / 10 dB); Bangla: 0 / 0 / 2 of 10; noise alone: 0"],''',
     '''    ["Voice commands (wake word \\"glass\\")", "12 held-out synthetic voices, 468 commands per condition; not people",
     "Clean 100%; soft 99.8%; whispered 98.9%; street noise 10 / 5 / 0 / -5 / -10 dB: 98.5 / 93.8 / 82.5 / 57.7 / 39.5%; people talking 10 / 5 dB: 54.3 / 26.1%"],
    ["Voice commands, false commands", "360 ordinary sentences per condition, 14 conditions",
     "13 of 5,040 (212 without the wake word); median response 0.86 s after the speech ends"],
    ["Bangla voice commands (Whisper base, offline)", "4 unseen Bangla voices, 15 phrases each", "45% clean, 62% with noise; no wrong command; not deployed"],'''),
])
print("voice pdfs ok")
