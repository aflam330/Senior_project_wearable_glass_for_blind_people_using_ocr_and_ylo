"""Count files in the comparison-dataset folder. Does not train."""
from __future__ import annotations

import json
from pathlib import Path

WORK = Path(__file__).resolve().parents[3]
ROOT = WORK / "data set for comparison"
OUT = WORK / "paper_evidence" / "comparison_dataset_manifest.json"
IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def count_tree(path: Path) -> dict:
    n_img = 0
    n_txt = 0
    n_other = 0
    classes = {}
    for dirpath, dirnames, filenames in __import__("os").walk(path):
        dirnames[:] = [d for d in dirnames if d not in {".vs", "__pycache__"}]
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext in IMAGE:
                n_img += 1
            elif ext == ".txt":
                n_txt += 1
            else:
                n_other += 1
    # class folders = immediate subdirs that contain files, searched one or two levels
    return {"images": n_img, "txt": n_txt, "other": n_other}


def class_counts(path: Path, depth: int = 2) -> list:
    rows = []
    if not path.is_dir():
        return rows
    for child in sorted(path.iterdir()):
        if not child.is_dir() or child.name in {".vs", "__MACOSX"}:
            continue
        n_img = 0
        n_txt = 0
        for dirpath, dirnames, filenames in __import__("os").walk(child):
            dirnames[:] = [d for d in dirnames if d not in {".vs"}]
            for name in filenames:
                ext = Path(name).suffix.lower()
                if ext in IMAGE:
                    n_img += 1
                elif ext == ".txt":
                    n_txt += 1
        rows.append({"name": child.name, "images": n_img, "txt": n_txt})
    return rows


def main() -> None:
    sets = []
    for child in sorted(ROOT.iterdir()):
        if not child.is_dir():
            continue
        stats = count_tree(child)
        stats["name"] = child.name
        stats["path"] = str(child)
        # one level down if there is a single wrapper folder
        kids = [p for p in child.iterdir() if p.is_dir() and p.name not in {".vs"}]
        focus = kids[0] if len(kids) == 1 else child
        stats["focus"] = str(focus)
        stats["top"] = class_counts(focus)
        sets.append(stats)
        print(child.name, stats["images"], stats["txt"], "classes", len(stats["top"]))
    OUT.write_text(json.dumps({"sets": sets}, indent=2), encoding="utf-8")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
