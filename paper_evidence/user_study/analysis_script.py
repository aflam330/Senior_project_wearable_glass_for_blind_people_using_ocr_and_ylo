"""User-study analysis. Exits with NOT MEASURED if no real rows exist."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "participant_data_template.csv"
OUT = Path(__file__).resolve().parent / "statistical_report.json"


def main() -> None:
    rows = list(csv.DictReader(TEMPLATE.open(encoding="utf-8")))
    filled = [r for r in rows if (r.get("task_time_s") or "").strip()]
    if not filled:
        OUT.write_text(
            json.dumps(
                {
                    "status": "NOT MEASURED",
                    "reason": "participant_data_template.csv has no completed trials",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("NOT MEASURED: no participant data")
        sys.exit(0)
    print("would analyse", len(filled), "rows — implement after collection")


if __name__ == "__main__":
    main()
