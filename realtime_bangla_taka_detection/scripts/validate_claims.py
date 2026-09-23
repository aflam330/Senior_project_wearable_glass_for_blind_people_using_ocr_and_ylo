"""Fail if a numeric claim has no existing artifact. Reads CLAIM_REGISTRY.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
REGISTRY = WORKSPACE / "paper_evidence" / "CLAIM_REGISTRY.json"


def main() -> None:
    if not REGISTRY.is_file():
        print("FAIL missing", REGISTRY)
        sys.exit(1)
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    claims = data if isinstance(data, list) else data.get("claims", [])
    bad = 0
    for row in claims:
        status = str(row.get("status", "")).upper()
        cid = row.get("claim_id")
        if status in {"NOT_MEASURED", "PENDING", "REJECTED"}:
            print("INFO", cid, status)
            continue
        if status != "VERIFIED":
            print("FAIL", cid, "unknown status", status)
            bad += 1
            continue
        art = row.get("source_artifact") or ""
        if not art:
            print("FAIL", cid, "verified but no source_artifact")
            bad += 1
            continue
        path = Path(art)
        if not path.is_file():
            alt = WORKSPACE / art
            alt2 = ROOT / art
            if not alt.is_file() and not alt2.is_file():
                print("FAIL", cid, "missing artifact", art)
                bad += 1
                continue
        print("PASS", cid)
    if bad:
        print("CLAIM VALIDATION FAIL", bad)
        sys.exit(1)
    print("CLAIM VALIDATION PASS", len(claims), "claims")


if __name__ == "__main__":
    main()
