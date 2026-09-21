"""Physical-note identity, manifests, and leakage-free splits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..authenticity import _view_paths, list_jaaltaka_notes
from ..config import JAALTAKA_DIR, ROOT

CAMVA_ROOT = ROOT / "results" / "camva"
SPLIT_DIR = CAMVA_ROOT / "splits"
SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def note_id_from_dir(note_dir: Path, label: int) -> str:
    kind = "genuine" if int(label) == 1 else "counterfeit"
    return f"{kind}:{note_dir.name}"


def file_sha1(path: Path, nbytes: int = 1_048_576) -> str:
    h = hashlib.sha1()
    with path.open("rb") as handle:
        chunk = handle.read(nbytes)
        h.update(chunk)
    return h.hexdigest()


def note_records() -> list[dict]:
    records = []
    for note_dir, label in list_jaaltaka_notes():
        views = _view_paths(note_dir)
        nid = note_id_from_dir(note_dir, label)
        records.append(
            {
                "note_id": nid,
                "label": int(label),
                "class": "genuine" if label == 1 else "counterfeit",
                "note_dir": str(note_dir),
                "n_views": len(views),
                "view_paths": [str(p) for p in views],
                "view_ids": [p.stem for p in views],
                "source": "JaalTaka",
                "camera_id": "unknown",
                "session_id": "unknown",
            }
        )
    return records


def build_and_save_splits(
    seed: int = SEED,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    out_dir: Path | None = None,
) -> dict:
    out_dir = out_dir or SPLIT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    records = note_records()
    if len(records) < 20:
        raise RuntimeError(f"JaalTaka notes too few: {len(records)} under {JAALTAKA_DIR}")

    by_label: dict[int, list[dict]] = {0: [], 1: []}
    for rec in records:
        by_label[rec["label"]].append(rec)

    rng = np.random.default_rng(seed)
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    rec_by_id = {r["note_id"]: r for r in records}

    for label, items in by_label.items():
        order = np.arange(len(items))
        rng.shuffle(order)
        n = len(items)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n_test + n_val >= n:
            n_test = max(1, n // 5)
            n_val = max(1, n // 5)
        test_items = [items[int(i)] for i in order[:n_test]]
        val_items = [items[int(i)] for i in order[n_test : n_test + n_val]]
        train_items = [items[int(i)] for i in order[n_test + n_val :]]
        splits["test"].extend(r["note_id"] for r in test_items)
        splits["val"].extend(r["note_id"] for r in val_items)
        splits["train"].extend(r["note_id"] for r in train_items)
        _ = label

    for key in splits:
        splits[key] = sorted(splits[key])

    assert_disjoint_splits(splits["train"], splits["val"], splits["test"])

    def stats(ids: list[str]) -> dict:
        labs = [rec_by_id[i]["label"] for i in ids]
        n_img = sum(rec_by_id[i]["n_views"] for i in ids)
        views = [rec_by_id[i]["n_views"] for i in ids]
        return {
            "n_notes": len(ids),
            "n_genuine_notes": int(sum(1 for y in labs if y == 1)),
            "n_counterfeit_notes": int(sum(1 for y in labs if y == 0)),
            "n_images": int(n_img),
            "views_per_note": dict(Counter(views)),
            "class_distribution": {"genuine": int(sum(labs)), "counterfeit": int(len(labs) - sum(labs))},
        }

    meta = {
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(JAALTAKA_DIR),
        "independent_unit": "physical_note_id",
        "note_id_format": "genuine|counterfeit:<folder_name>",
        "n_unique_genuine_notes": len(by_label[1]),
        "n_unique_counterfeit_notes": len(by_label[0]),
        "n_notes_total": len(records),
        "n_images_total": int(sum(r["n_views"] for r in records)),
        "views_per_note_all": dict(Counter(r["n_views"] for r in records)),
        "train": stats(splits["train"]),
        "val": stats(splits["val"]),
        "test": stats(splits["test"]),
        "leakage_check": {
            "train_val": 0,
            "train_test": 0,
            "val_test": 0,
        },
    }
    (out_dir / "train_note_ids.json").write_text(json.dumps(splits["train"], indent=2), encoding="utf-8")
    (out_dir / "val_note_ids.json").write_text(json.dumps(splits["val"], indent=2), encoding="utf-8")
    (out_dir / "test_note_ids.json").write_text(json.dumps(splits["test"], indent=2), encoding="utf-8")
    (out_dir / "split_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "note_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("unique genuine notes", meta["n_unique_genuine_notes"])
    print("unique counterfeit notes", meta["n_unique_counterfeit_notes"])
    print("images", meta["n_images_total"])
    print("views per note", meta["views_per_note_all"])
    for name in ("train", "val", "test"):
        print(name, json.dumps(meta[name]))
    return {"splits": splits, "records": rec_by_id, "meta": meta}


def assert_disjoint_splits(train: list[str], val: list[str], test: list[str]) -> None:
    st, sv, se = set(train), set(val), set(test)
    assert st.isdisjoint(sv), f"train/val leak: {st & sv}"
    assert st.isdisjoint(se), f"train/test leak: {st & se}"
    assert sv.isdisjoint(se), f"val/test leak: {sv & se}"
    assert len(train) == len(st) and len(val) == len(sv) and len(test) == len(se)


def load_splits(split_dir: Path | None = None) -> tuple[dict[str, list[str]], dict[str, dict]]:
    split_dir = split_dir or SPLIT_DIR
    train = json.loads((split_dir / "train_note_ids.json").read_text(encoding="utf-8"))
    val = json.loads((split_dir / "val_note_ids.json").read_text(encoding="utf-8"))
    test = json.loads((split_dir / "test_note_ids.json").read_text(encoding="utf-8"))
    assert_disjoint_splits(train, val, test)
    records = {r["note_id"]: r for r in json.loads((split_dir / "note_manifest.json").read_text(encoding="utf-8"))}
    return {"train": train, "val": val, "test": test}, records
