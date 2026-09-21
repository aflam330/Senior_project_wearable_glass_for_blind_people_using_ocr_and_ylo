"""Fail if a claim_id has a numeric claim but no existing evidence file."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REGISTRY = ROOT.parent / "paper_evidence" / "claim_registry.csv"


def main() -> None:
    if not REGISTRY.is_file():
        print("FAIL missing", REGISTRY)
        sys.exit(1)
    rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8")))
    bad = 0
    for row in rows:
        status = (row.get("evidence_status") or "").strip().upper()
        path = (row.get("result_file") or "").strip()
        if status in {"NOT MEASURED", "PROTOCOL_ONLY"}:
            print("WARNING", row["claim_id"], status)
            continue
        if not path:
            print("FAIL", row["claim_id"], "no result_file")
            bad += 1
            continue
        full = ROOT.parent / path if not Path(path).is_absolute() else Path(path)
        if not full.is_file() and not (ROOT / path).is_file():
            # try relative to roboeye root
            alt = ROOT / path
            if not alt.is_file():
                print("FAIL", row["claim_id"], "missing", path)
                bad += 1
                continue
        print("PASS", row["claim_id"])
    if bad:
        sys.exit(1)
    print("claim registry ok", len(rows), "rows")


if __name__ == "__main__":
    main()
