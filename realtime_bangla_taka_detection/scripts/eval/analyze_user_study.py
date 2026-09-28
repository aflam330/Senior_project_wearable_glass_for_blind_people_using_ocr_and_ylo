"""Paired comparison for a user study. Prints DATA_NOT_COLLECTED when no CSV exists."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("user_study_pairs.csv")
    if not path.is_file():
        print("DATA_NOT_COLLECTED")
        print("Expected columns: participant, baseline_time, roboeye_time")
        raise SystemExit(0)
    diffs = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            diffs.append(float(row["roboeye_time"]) - float(row["baseline_time"]))
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1)
    sd = math.sqrt(var)
    t = mean / (sd / math.sqrt(n)) if sd else 0.0
    half = 1.96 * sd / math.sqrt(n) if n else 0.0
    d = mean / sd if sd else 0.0
    print({"n": n, "mean_diff": mean, "t": t, "cohens_d": d, "ci95": [mean - half, mean + half]})


if __name__ == "__main__":
    main()
