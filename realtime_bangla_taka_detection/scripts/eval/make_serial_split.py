"""Serial-disjoint JaalTaka split: no TEST counterfeit shares a serial print with a TRAIN counterfeit.

Rule (written before any model was trained on it):
  counterfeit notes are grouped by the first six digits of their OCR serial
  (results/jaal_whole/serial_audit.json); a note with no readable serial is its own group.
  The two largest counterfeit groups (the 3274658 and 184383x prints) go to TRAIN. The remaining
  counterfeit groups are shuffled (seed 42) and assigned whole to VAL or TEST, alternating by size,
  so each gets about half of the remaining counterfeit notes.
  Genuine notes (unique serials) are split at random, seed 42, 70 / 15 / 15.
Caveat: OCR can misread digits, so a few "unseen" groups may be misread copies of a train print.
Output: results/serial_split/{train,val,test}_note_ids.json, note_manifest.json, split_metadata.json
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from roboeye.camva.notes import SPLIT_DIR, load_splits  # noqa: E402

OUT = ROOT / "results" / "serial_split"


def main() -> None:
    _, records = load_splits(SPLIT_DIR)
    audit = {r["note_id"]: r for r in json.loads((ROOT / "results/jaal_whole/serial_audit.json").read_text(encoding="utf-8"))["rows"]}
    groups = defaultdict(list)
    genuine = []
    for nid, r in records.items():
        if int(r["label"]) == 1:
            genuine.append(nid)
            continue
        s = audit.get(nid, {}).get("serial")
        groups[s[:6] if s else f"unread:{nid}"].append(nid)
    order = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    train = [n for _, ns in order[:2] for n in ns]
    rest = order[2:]
    random.Random(42).shuffle(rest)
    rest.sort(key=lambda kv: -len(kv[1]))
    val, test = [], []
    for _, ns in rest:
        (val if len(val) <= len(test) else test).extend(ns)
    rng = random.Random(42)
    genuine.sort()
    rng.shuffle(genuine)
    n = len(genuine)
    g_tr, g_va = genuine[: int(0.70 * n)], genuine[int(0.70 * n): int(0.85 * n)]
    g_te = genuine[int(0.85 * n):]
    split = {"train": sorted(train + g_tr), "val": sorted(val + g_va), "test": sorted(test + g_te)}
    OUT.mkdir(parents=True, exist_ok=True)
    for k, v in split.items():
        (OUT / f"{k}_note_ids.json").write_text(json.dumps(v, indent=0), encoding="utf-8")
    shutil.copy(SPLIT_DIR / "note_manifest.json", OUT / "note_manifest.json")
    meta = {"rule": __doc__.split("Output")[0].strip(),
            "counts": {k: {"notes": len(v), "counterfeit": sum(int(records[x]["label"]) == 0 for x in v)} for k, v in split.items()},
            "train_counterfeit_groups": [g for g, _ in order[:2]],
            "counterfeit_groups_total": len(groups)}
    (OUT / "split_metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta["counts"]), meta["train_counterfeit_groups"])


if __name__ == "__main__":
    main()
