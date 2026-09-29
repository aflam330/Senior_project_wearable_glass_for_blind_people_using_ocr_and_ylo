"""Look for extra Bangladeshi Taka folders and write a manifest. Does not download foreign currency."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
OUT = WORK / "paper_evidence" / "external_bdt_manifest.json"
CANDIDATES = [
    WORK / "external_data" / "noteshieldbd",
    WORK / "external_data" / "bdt_counterfeit_mendeley",
    WORK / "external_data" / "nstu_bdtaka",
    WORK / "external_data" / "modelnet40",
]
SOURCES = [
    {
        "name": "NoteShieldBD",
        "currency": "BDT",
        "task": "authenticity",
        "url": "https://www.kaggle.com/datasets/bibhaschowdhury/bangladeshi-banknote-authentication-dataset",
        "listed_size": "20.4 GB on Kaggle",
        "download": "not_started",
        "reason": "No Kaggle token at %USERPROFILE%\\.kaggle\\kaggle.json",
    },
    {
        "name": "Bangladeshi Counterfeit Currency Image Dataset",
        "currency": "BDT",
        "task": "authenticity",
        "url": "https://data.mendeley.com/datasets/gzzz5nrvbn/1",
        "doi": "10.17632/gzzz5nrvbn.1",
        "listed_size": "1286 images; Kaggle mirror listed 3.18 GB",
        "download": "api_http_400",
        "reason": "GET https://data.mendeley.com/public-api/datasets/gzzz5nrvbn/files?folder_id=root returned {\"error\": 400}",
    },
    {
        "name": "NSTU-BDTAKA",
        "currency": "BDT",
        "task": "denomination_and_boxes",
        "url": "https://data.mendeley.com/datasets/w4y6h723xg/1",
        "doi": "10.17632/w4y6h723xg.1",
        "download": "not_started",
        "reason": "Public labels are denomination and boxes, not genuine versus counterfeit",
    },
    {
        "name": "ModelNet40",
        "currency": "none",
        "task": "multi_view_objects",
        "url": "http://modelnet.cs.princeton.edu/ModelNet40.zip",
        "download": "not_started",
        "reason": "Not a currency set. A VCDS run on it is a separate training job and was not started.",
    },
]


def count_images(path: Path) -> int:
    if not path.is_dir():
        return 0
    n = 0
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        n += sum(1 for _ in path.rglob(ext))
    return n


def main() -> None:
    rows = []
    for path in CANDIDATES:
        rows.append({"path": str(path), "present": path.is_dir(), "images": count_images(path)})
    payload = {"sources": SOURCES, "local": rows, "foreign_currency_downloaded": False}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", OUT)
    for row in rows:
        print(row["present"], row["images"], row["path"])


if __name__ == "__main__":
    main()
