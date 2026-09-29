"""Fail if a numeric claim has no existing artifact. Reads CLAIM_REGISTRY.json.

A claim with "json_key" (a list of keys, or a dot path; list indices as integers) is also value-checked: the number at
that key in the artifact must equal "value". Claims without it are checked for existence only.
"""
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
    checked_values = 0
    for row in claims:
        status = str(row.get("status", "")).upper()
        cid = row.get("claim_id")
        info_prefixes = (
            "NOT_MEASURED",
            "PENDING",
            "REJECTED",
            "READY_FOR_DEVICE",
            "DATA_READY_FOR_COLLECTION",
            "PROTOCOL_READY",
        )
        if status in {"NOT_MEASURED", "PENDING", "REJECTED"} or status.startswith(info_prefixes):
            print("INFO", cid, status)
            continue
        if status not in {"VERIFIED", "VERIFIED_SIMULATED", "NOT_MET"}:
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
        key = row.get("json_key")
        if key:  # value check: the claimed number must equal the number stored in the artifact
            src = next(p for p in (path, WORKSPACE / art, ROOT / art) if p.is_file())
            try:
                node = json.loads(src.read_text(encoding="utf-8"))
                for part in (key if isinstance(key, list) else key.split(".")):
                    node = node[int(part)] if isinstance(node, list) else node[part]
                ok = abs(float(node) - float(row["value"])) <= 1e-9
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                print("FAIL", cid, "json_key", key, "unreadable:", exc)
                bad += 1
                continue
            if not ok:
                print("FAIL", cid, "value", row["value"], "!= artifact", node)
                bad += 1
                continue
            checked_values += 1
        print("PASS", cid)
    print("value-checked claims:", checked_values, "of", len(claims))
    if bad:
        print("CLAIM VALIDATION FAIL", bad)
        sys.exit(1)
    print("CLAIM VALIDATION PASS", len(claims), "claims")


if __name__ == "__main__":
    main()
