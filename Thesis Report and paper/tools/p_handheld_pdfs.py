"""Hand-held detector results into the features and results PDFs (15 October 2026)."""
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
    ('''     "YOLOv8s trained on 22,333 automatically labelled pictures", "mAP@0.5 0.995 on held-out pictures; 91.5% on 1,536 photos by other people"],''',
     '''     "YOLOv8s trained on 22,333 automatically labelled pictures, fine-tuned with 6,000 synthetic hand-held close-ups",
     "mAP@0.5 0.995 on held-out pictures; 94.1% on 1,536 photos by other people; notes held in a hand 11.8% → 47.5% (Pi: 8.7% → 33.5%)"],'''),
])
patch("make_results.py", [
    ('''    ["Taka recognition", "1,536 photographs by other people (Bangla Money)", "91.5% correct note value"],
    ["Taka recognition", "1,144 close-ups of folded notes held in a hand (NSTU-BDTAKA)", "18.5% correct note value"],
    ["Several notes in one picture", "900 pictures of 1, 2 or 3 test-split notes on new backgrounds", "Count right 99.7%; spoken total right 99.2%"],''',
     '''    ["Taka recognition", "1,536 photographs by other people (Bangla Money)", "94.1% right, 17 wrong (hand-held detector, 0.75); earlier detector 89.6%, 50 wrong (91.5% at 0.25)"],
    ["Taka recognition, notes held in a hand", "1,127 NSTU-BDTAKA close-ups, one per photo", "47.5% right (earlier detector 11.8%); Pi 8-bit: 33.5% (earlier 8.7%)"],
    ["Unknown notes (one Taka, not a class)", "101 Bangla Money photos", "Announced as a value: 36.6% (earlier 22.8%); Pi 8.9% (earlier 5.0%)"],
    ["Several notes in one picture", "900 pictures of 1, 2 or 3 test-split notes on new backgrounds", "Spoken total right 97.0% (earlier detector 99.2%); never too many counted"],'''),
])
print("pdf handheld ok")
