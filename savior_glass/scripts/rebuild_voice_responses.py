"""Rebuild study_data/voice/responses.csv from each P***/answers.json on disk.

study_data/ is gitignored, so Pi recordings are not on GitHub — copy them onto this
machine (scp/rsync), then run:

  python savior_glass/scripts/rebuild_voice_responses.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

VOICE = Path(__file__).resolve().parents[1] / "study_data" / "voice"
OUT = VOICE / "responses.csv"
COLS = [
    "id",
    "name",
    "start",
    "end",
    "minutes",
    "easy_to_use",
    "speech_clear",
    "helped_currency",
    "helped_reading",
    "trust",
    "would_use_daily",
    "answered_by_voice",
    "field_run",
    "source",
]
IMPORTED = VOICE / "imported" / "user_data.txt"   # rows without answers.json (see imported/README.txt); kept, marked "imported"
QKEYS = [
    "easy_to_use",
    "speech_clear",
    "helped_currency",
    "helped_reading",
    "trust",
    "would_use_daily",
]


def main() -> int:
    rows = []
    for folder in sorted(VOICE.glob("P*")):
        if not folder.is_dir():
            continue
        path = folder / "answers.json"
        if not path.is_file():
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        answers = rec.get("answers") or {}
        sources = rec.get("answered_by") or {}
        voice_n = sum(1 for v in sources.values() if v == "voice")
        rows.append(
            {
                "id": rec.get("id", folder.name),
                "name": rec.get("name", ""),
                "start": rec.get("start", ""),
                "end": rec.get("end", ""),
                "minutes": rec.get("minutes", ""),
                **{k: answers.get(k, "") for k in QKEYS},
                "answered_by_voice": voice_n,
                "field_run": rec.get("field_run", ""),
                "source": "app",
            }
        )
    if IMPORTED.is_file():
        with IMPORTED.open(encoding="utf-8-sig", newline="") as f:
            rows += [dict(zip(COLS, r), source="imported") for r in csv.reader(f) if r]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT} ({len(rows)} participants from answers.json)")
    for r in rows:
        print(f"  {r['id']}: name={r['name']!r} complete_fields={sum(1 for k in QKEYS if r.get(k))} field_run={r['field_run']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
