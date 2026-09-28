"""Compile project Python files and check claim artifacts exist. Does not train."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT.parent
OUT = WS / "paper_evidence" / "diagnosis_compile.json"
SKIP = {"__pycache__", ".git", "cache", "node_modules"}


def py_files() -> list[Path]:
    found = []
    for base in (ROOT, WS / "savior_glass", WS / "paper_evidence"):
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if any(part in SKIP for part in path.parts):
                continue
            found.append(path)
    return found


def main() -> None:
    errors = []
    n = 0
    for path in py_files():
        n += 1
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append({"file": str(path), "error": str(exc)[:500]})
    claims_path = WS / "paper_evidence" / "CLAIM_REGISTRY.json"
    missing = []
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    for claim in claims.get("claims", []):
        src = claim.get("source_artifact")
        if not src or src == "NOT_MEASURED":
            continue
        if not Path(src).is_file():
            missing.append({"id": claim.get("claim_id"), "artifact": src})
    payload = {"python_files": n, "syntax_errors": errors, "missing_claim_artifacts": missing}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"python_files": n, "syntax_errors": len(errors), "missing_claims": len(missing)}), flush=True)


if __name__ == "__main__":
    main()
