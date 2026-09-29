"""Attach json_key to existing claims so validate_claims.py can check their values, and report
claims whose value does not appear in their own artifact.

For each claim with a numeric value and a JSON source_artifact, every numeric leaf of the artifact is
compared with the value (|a - b| <= 1e-9). One match: json_key is set. Several: the first is used only
if all matches share the same final key name, else left unset. None: listed as NOT_FOUND for review;
the claim itself is not changed.

Output: CLAIM_REGISTRY.json updated in place; paper_evidence/CLAIM_VALUE_AUDIT.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
REG = WORK / "paper_evidence" / "CLAIM_REGISTRY.json"
OUT = WORK / "paper_evidence" / "CLAIM_VALUE_AUDIT.md"


def leaves(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, path + (str(k),))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaves(v, path + (str(i),))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path, float(node)


def resolve(art: str) -> Path | None:
    for p in (Path(art), WORK / art, ROOT / art):
        if p.is_file():
            return p
    return None


def main() -> None:
    data = json.loads(REG.read_text(encoding="utf-8"))
    claims = data["claims"] if isinstance(data, dict) else data
    rows = {"attached": [], "already": [], "not_found": [], "ambiguous": [], "not_json": [], "non_numeric": []}
    for c in claims:
        cid = c.get("claim_id")
        if c.get("json_key"):
            rows["already"].append(cid)
            continue
        try:
            val = float(c.get("value"))
        except (TypeError, ValueError):
            rows["non_numeric"].append(cid)
            continue
        src = resolve(c.get("source_artifact") or "")
        if src is None or src.suffix.lower() != ".json":
            rows["not_json"].append(cid)
            continue
        try:
            blob = json.loads(src.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rows["not_json"].append(cid)
            continue
        hits = [p for p, v in leaves(blob) if abs(v - val) <= 1e-9]
        if not hits:
            rows["not_found"].append((cid, c.get("value"), c.get("source_artifact")))
        elif len(hits) == 1 or len({h[-1] for h in hits}) == 1:
            c["json_key"] = list(hits[0])  # list: some keys contain dots
            rows["attached"].append((cid, c["json_key"]))
        else:
            rows["ambiguous"].append((cid, len(hits)))
    REG.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    L = ["# Claim value audit (2026-09-29)", "",
         "`validate_claims.py` used to check only that each claim's source file exists. It now also checks the value "
         "when a claim has `json_key`. This pass attached `json_key` where the claimed value occurs in the claim's own JSON "
         "artifact (`scripts/eval/attach_claim_keys.py`).", "",
         f"- value now checked: {len(rows['attached']) + len(rows['already'])} (newly attached {len(rows['attached'])})",
         f"- value not found in its artifact: {len(rows['not_found'])}",
         f"- value found at several unrelated keys (left unchecked): {len(rows['ambiguous'])}",
         f"- artifact not JSON (markdown, CSV, weights): {len(rows['not_json'])}",
         f"- value not numeric: {len(rows['non_numeric'])}", "",
         "## Value not found in its own artifact", "",
         "These are not proven wrong: the artifact may store the number in another form (a percentage, a count, a rounded value). "
         "Each needs a look before it is cited.", "", "| claim | value | artifact |", "|---|---|---|"]
    L += [f"| {cid} | {v} | `{a}` |" for cid, v, a in rows["not_found"]]
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print({k: len(v) for k, v in rows.items()})


if __name__ == "__main__":
    main()
