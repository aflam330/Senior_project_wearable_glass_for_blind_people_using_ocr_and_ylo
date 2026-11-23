"""Checks of the multi-note count and total (CurrencyMode.distinct_notes, total_sentence). No model is loaded.

    python scripts/test_multi_note.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from modes.currency_mode import CurrencyMode  # noqa: E402


def hit(name, conf, box):
    return {"name": name, "conf": conf, "bbox": box}


D = CurrencyMode.distinct_notes
# confidences relative to the announce threshold of the detector in use (0.60 to 0.90 depending on the detector)
T = config.CURRENCY_ANNOUNCE_CONF
HI, MID = min(0.99, T + 0.08), min(0.98, T + 0.04)
# two separate notes, both sure
n = D([hit("500_taka", HI, (0, 0, 100, 50)), hit("100_taka", MID, (200, 0, 100, 50))])
assert [x["name"] for x in n] == ["500_taka", "100_taka"]
# the same note seen twice with two values: the surer one is kept
n = D([hit("1000_taka", MID, (0, 0, 100, 50)), hit("500_taka", HI, (5, 2, 100, 50))])
assert [x["name"] for x in n] == ["500_taka"]
# a note the detector is not sure about is not counted
n = D([hit("500_taka", HI, (0, 0, 100, 50)), hit("5_taka", T - 0.01, (200, 0, 100, 50))])
assert len(n) == 1
# notes that touch but overlap little are two notes
n = D([hit("50_taka", HI, (0, 0, 100, 50)), hit("50_taka", HI, (90, 0, 100, 50))])
assert len(n) == 2
assert D([]) == []

m = CurrencyMode.__new__(CurrencyMode)
s = m.total_sentence([hit("100_taka", 0.9, (0, 0, 1, 1)), hit("500_taka", 0.9, (0, 0, 1, 1)), hit("50_taka", 0.9, (0, 0, 1, 1))])
assert s.startswith("৩টি নোট") and "৬৫০" in s and s.index("পাঁচশত") < s.index("একশত") < s.index("পঞ্চাশ"), s
print("multi-note: all checks pass")
