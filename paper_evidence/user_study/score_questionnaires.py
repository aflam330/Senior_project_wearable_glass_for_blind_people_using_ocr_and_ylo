"""Score SUS and raw NASA-TLX from item-level answers.

Input CSV columns: participant_id, condition, sus_1..sus_10 (1-5), tlx_mental, tlx_physical,
tlx_temporal, tlx_performance, tlx_effort, tlx_frustration (0-100; performance 0 = perfect).
Output: the same rows with `sus` (0-100) and `nasa_tlx` (raw TLX, mean of six, 0-100) added,
ready to merge into participant_data.csv for analysis_script.py.

SUS (Brooke 1996): odd items contribute (x - 1), even items (5 - x); sum x 2.5.
Raw TLX (Hart 2006): unweighted mean of the six subscales.

Usage: python score_questionnaires.py items.csv scored.csv
"""
from __future__ import annotations

import csv
import sys

TLX = ("tlx_mental", "tlx_physical", "tlx_temporal", "tlx_performance", "tlx_effort", "tlx_frustration")


def sus_score(items: list[float]) -> float:
    if len(items) != 10 or any(not 1 <= x <= 5 for x in items):
        raise ValueError("SUS needs 10 answers in 1..5")
    return 2.5 * sum((x - 1) if i % 2 == 0 else (5 - x) for i, x in enumerate(items))


def raw_tlx(values: list[float]) -> float:
    if len(values) != 6 or any(not 0 <= v <= 100 for v in values):
        raise ValueError("NASA-TLX needs 6 subscales in 0..100")
    return sum(values) / 6.0


def main(src: str, dst: str) -> None:
    with open(src, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        try:
            r["sus"] = f"{sus_score([float(r[f'sus_{i}']) for i in range(1, 11)]):.1f}"
        except (KeyError, ValueError):
            r["sus"] = ""
        try:
            r["nasa_tlx"] = f"{raw_tlx([float(r[k]) for k in TLX]):.1f}"
        except (KeyError, ValueError):
            r["nasa_tlx"] = ""
    if not rows:
        print("no rows: NOT MEASURED")
        return
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"scored {len(rows)} rows -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:  # self-check with textbook values
        assert sus_score([3] * 10) == 50.0
        assert sus_score([5, 1] * 5) == 100.0
        assert sus_score([1, 5] * 5) == 0.0
        assert raw_tlx([50] * 6) == 50.0
        print("self-check passed")
