"""Capture and registry tool for a print-disjoint counterfeit dataset.

Every photo carries note id, print id, currency, denomination, label, camera and lighting, plus
blur and exposure measured on save. A print is the physical printing a note came from: counterfeit
notes that share a serial (or plate) share a print id; each genuine note is its own print unless
the collector groups them. The split is decided per PRINT, written to splits.json before any model is
trained, and never changed: the tool refuses a photo whose print is in another split, a note that
moves to a second print, and a byte-identical duplicate.

Commands (all take --root, the dataset folder):
  plan    --prints prints.csv [--seed 42 --val 0.15 --test 0.30]
          Assign every print in the CSV (print_id,currency,denomination,label) to train/val/test,
          stratified by currency x denomination x label. Only unassigned prints are placed; existing
          assignments are never changed. The file is hashed and the hash printed for the paper.
  add     --image PATH | --camera-index N   --note-id --print-id --currency --denomination
          --label genuine|counterfeit --camera phone|glass --lighting room|backlight
          [--wm-corners x0,y0,x1,y1,x2,y2,x3,y3] [--notes TEXT]
          Copy the photo in (or grab one frame from a camera), measure quality, append to manifest.jsonl.
  check   Re-verify every rule over the whole manifest and print counts per split. Exit 1 on a violation.
  export  Write split lists (<split>.txt of image paths) and datasheet_counts.json for a release.

Rules are also importable (Registry) so the glass app or a notebook can use the same checks.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "val", "test")
CAMERAS = ("phone", "glass")
LIGHTING = ("room", "backlight")
LABELS = ("genuine", "counterfeit")
# Quality flags only; photos are kept, so the flags can be studied. Values are fixed here, before collection.
BLUR_MIN_LAPVAR = 60.0
DARK_MEAN = 45.0
CLIP_FRAC = 0.25
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")  # ids become folder and file names


class RuleError(ValueError):
    pass


def quality(bgr: np.ndarray) -> dict:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = 800 / max(g.shape)
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    mean = float(g.mean())
    dark = float((g < 10).mean())
    bright = float((g > 245).mean())
    flags = [f for f, bad in (("blurry", lap < BLUR_MIN_LAPVAR), ("dark", mean < DARK_MEAN),
                              ("underexposed", dark > CLIP_FRAC), ("overexposed", bright > CLIP_FRAC)) if bad]
    return {"lap_var": round(lap, 2), "mean": round(mean, 2), "frac_black": round(dark, 4),
            "frac_white": round(bright, 4), "width": int(bgr.shape[1]), "height": int(bgr.shape[0]), "flags": flags}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Registry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.splits_path = self.root / "splits.json"
        self.manifest_path = self.root / "manifest.jsonl"
        self.splits = json.loads(self.splits_path.read_text(encoding="utf-8")) if self.splits_path.exists() else {"prints": {}, "history": []}
        self.rows = [json.loads(line) for line in self.manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()] \
            if self.manifest_path.exists() else []

    # ---- split plan ----
    def plan(self, prints: list[dict], seed: int = 42, val: float = 0.15, test: float = 0.30) -> dict:
        new = [p for p in prints if p["print_id"] not in self.splits["prints"]]
        strata = defaultdict(list)
        for p in new:
            strata[(p["currency"], str(p["denomination"]), p["label"])].append(p)
        rng = random.Random(seed)
        placed = Counter()
        for key in sorted(strata):
            group = sorted(strata[key], key=lambda p: p["print_id"])
            rng.shuffle(group)
            n = len(group)
            n_te = round(n * test)
            n_va = round(n * val)
            for i, p in enumerate(group):
                s = "test" if i < n_te else "val" if i < n_te + n_va else "train"
                self.splits["prints"][p["print_id"]] = {"split": s, "currency": p["currency"],
                                                        "denomination": str(p["denomination"]), "label": p["label"]}
                placed[s] += 1
        self.splits["history"].append({"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "seed": seed,
                                       "val": val, "test": test, "placed": dict(placed)})
        self.splits_path.write_text(json.dumps(self.splits, indent=1, sort_keys=True), encoding="utf-8")
        return {"placed": dict(placed), "sha256": sha256(self.splits_path)}

    def split_of(self, print_id: str) -> str | None:
        e = self.splits["prints"].get(print_id)
        return e["split"] if e else None

    # ---- adding photos ----
    def validate(self, rec: dict, digest: str | None = None) -> None:
        for k in ("note_id", "print_id", "currency", "denomination"):
            if not SAFE_ID.match(str(rec.get(k, ""))):
                raise RuleError(f"{k}={rec.get(k)!r}: use letters, digits, '.', '_' or '-' only")
        for k, allowed in (("camera", CAMERAS), ("lighting", LIGHTING), ("label", LABELS)):
            if rec[k] not in allowed:
                raise RuleError(f"{k}={rec[k]!r} not in {allowed}")
        entry = self.splits["prints"].get(rec["print_id"])
        if entry is None:
            raise RuleError(f"print {rec['print_id']!r} has no split; run `plan` with it first (split is fixed before training)")
        for k in ("currency", "denomination", "label"):
            if str(entry[k]) != str(rec[k]):
                raise RuleError(f"print {rec['print_id']!r} is registered as {k}={entry[k]!r}, photo says {rec[k]!r}")
        for r in self.rows:
            if r["note_id"] == rec["note_id"] and r["print_id"] != rec["print_id"]:
                raise RuleError(f"note {rec['note_id']!r} already belongs to print {r['print_id']!r}")
            if digest and r["sha256"] == digest:
                raise RuleError(f"identical file already stored as {r['file']}")

    def add(self, image: Path | None, meta: dict, frame: np.ndarray | None = None) -> dict:
        rec = {k: str(v) for k, v in meta.items() if v is not None}
        if frame is not None:
            tmp = self.root / "_incoming.jpg"
            cv2.imwrite(str(tmp), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image = tmp
        digest = sha256(image)
        self.validate(rec, digest)
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise RuleError(f"cannot read image {image}")
        split = self.split_of(rec["print_id"])
        idx = sum(r["note_id"] == rec["note_id"] for r in self.rows) + 1
        rel = Path("images") / split / rec["currency"] / rec["print_id"] / f"{rec['note_id']}_{rec['camera']}_{rec['lighting']}_{idx:02d}{image.suffix.lower()}"
        (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (shutil.move if frame is not None else shutil.copy2)(str(image), str(self.root / rel))
        rec.update({"file": rel.as_posix(), "split": split, "sha256": digest, "quality": quality(bgr),
                    "added_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.rows.append(rec)
        return rec

    # ---- whole-dataset check ----
    def check(self) -> tuple[list[str], dict]:
        errors = []
        print_split, note_print, hashes = defaultdict(set), {}, {}
        for r in self.rows:
            print_split[r["print_id"]].add(r["split"])
            if self.split_of(r["print_id"]) != r["split"]:
                errors.append(f"{r['file']}: stored under {r['split']}, plan says {self.split_of(r['print_id'])}")
            if note_print.setdefault(r["note_id"], r["print_id"]) != r["print_id"]:
                errors.append(f"note {r['note_id']} in prints {note_print[r['note_id']]} and {r['print_id']}")
            if r["sha256"] in hashes:
                errors.append(f"duplicate file {r['file']} = {hashes[r['sha256']]}")
            hashes[r["sha256"]] = r["file"]
            if not (self.root / r["file"]).exists():
                errors.append(f"missing file {r['file']}")
        errors += [f"print {p} in splits {sorted(s)}" for p, s in print_split.items() if len(s) > 1]
        counts = {}
        for s in SPLITS:
            rs = [r for r in self.rows if r["split"] == s]
            counts[s] = {"photos": len(rs), "notes": len({r["note_id"] for r in rs}),
                         "prints": dict(Counter(f"{self.splits['prints'][p]['currency']}/{self.splits['prints'][p]['label']}"
                                                for p in {r["print_id"] for r in rs})),
                         "camera_x_lighting": dict(Counter(f"{r['camera']}/{r['lighting']}" for r in rs)),
                         "quality_flags": dict(Counter(f for r in rs for f in r["quality"]["flags"])),
                         "with_watermark_corners": sum("wm_corners" in r for r in rs)}
        counts["planned_prints"] = dict(Counter(v["split"] for v in self.splits["prints"].values()))
        return errors, counts

    def export(self) -> dict:
        errors, counts = self.check()
        if errors:
            raise RuleError("fix these before a release:\n" + "\n".join(errors))
        for s in SPLITS:
            (self.root / f"{s}.txt").write_text("\n".join(r["file"] for r in self.rows if r["split"] == s) + "\n", encoding="utf-8")
        counts["splits_sha256"] = sha256(self.splits_path)
        (self.root / "datasheet_counts.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
        return counts


def _grab(index: int) -> np.ndarray:
    cap = cv2.VideoCapture(index)
    frame = None
    for _ in range(10):  # let exposure settle
        ok, f = cap.read()
        frame = f if ok else frame
    cap.release()
    if frame is None:
        raise RuleError(f"camera {index} gave no frame")
    return frame


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("plan", "add", "check", "export"))
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--prints", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.30)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--camera-index", type=int)
    for k in ("note-id", "print-id", "currency", "denomination", "label", "camera", "lighting", "wm-corners", "notes"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args(argv)
    reg = Registry(a.root)
    try:
        if a.command == "plan":
            with open(a.prints, newline="", encoding="utf-8") as f:
                print(json.dumps(reg.plan(list(csv.DictReader(f)), a.seed, a.val, a.test), indent=1))
        elif a.command == "add":
            meta = {k: getattr(a, k.replace("-", "_")) for k in ("note-id", "print-id", "currency", "denomination", "label", "camera", "lighting")}
            missing = [k for k, v in meta.items() if not v]
            if missing:
                raise RuleError(f"missing --{', --'.join(missing)}")
            meta = {k.replace("-", "_"): v for k, v in meta.items()}
            if a.wm_corners:
                c = [float(v) for v in a.wm_corners.split(",")]
                if len(c) != 8 or not all(0 <= v <= 1 for v in c):
                    raise RuleError("--wm-corners needs 8 fractions: TL, TR, BR, BL as x,y")
                meta["wm_corners"] = json.dumps(c)
            meta["notes"] = a.notes
            frame = _grab(a.camera_index) if a.camera_index is not None else None
            if frame is None and not a.image:
                raise RuleError("give --image or --camera-index")
            rec = reg.add(a.image, meta, frame)
            print(json.dumps(rec, indent=1, ensure_ascii=False))
            if rec["quality"]["flags"]:
                print("warning: quality flags", rec["quality"]["flags"], "- retake if this was not intended", file=sys.stderr)
        elif a.command == "check":
            errors, counts = reg.check()
            print(json.dumps(counts, indent=1))
            for e in errors:
                print("VIOLATION", e, file=sys.stderr)
            return 1 if errors else 0
        else:
            print(json.dumps(reg.export(), indent=1))
    except RuleError as exc:
        print("REFUSED:", exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
