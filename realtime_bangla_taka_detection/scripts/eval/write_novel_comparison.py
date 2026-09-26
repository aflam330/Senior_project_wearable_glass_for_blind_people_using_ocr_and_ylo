"""Write the novel-algorithm comparison from saved test metrics only."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALGOS = ["ogpd", "vcie", "apc", "sfaq", "igcr", "ugf", "ndal", "sfpl", "cvs", "mtpt"]
OUTS = [
    ROOT.parent / "paper_evidence" / "NOVEL_ALGORITHMS_COMPARISON.md",
    ROOT / "paper_evidence" / "NOVEL_ALGORITHMS_COMPARISON.md",
]


def main() -> None:
    lines = [
        "# Novel algorithms, seed 42",
        "",
        "Every number below is copied from `results/novel/<algo>/seed42/test/test_metrics.json`.",
        "Missing files are NOT_MEASURED. Nothing here was typed in by hand.",
        "",
        "| algorithm | 1-view | 2-view | 3-view | 4-view | 5-view | 6-view | 6-view F1 | 6-view ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    found = 0
    for algo in ALGOS:
        path = ROOT / "results" / "novel" / algo / "seed42" / "test" / "test_metrics.json"
        if not path.is_file():
            lines.append(f"| {algo} | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        by_k = {int(row["k"]): row for row in payload["views"]}
        cells = []
        for k in range(1, 7):
            row = by_k.get(k)
            cells.append("NOT_MEASURED" if row is None else f"{row['accuracy']:.4f}")
        last = by_k.get(6)
        f1 = "NOT_MEASURED" if last is None else f"{last['macro_f1']:.4f}"
        ece = "NOT_MEASURED" if last is None else f"{last['ece']:.4f}"
        lines.append(f"| {algo} | " + " | ".join(cells) + f" | {f1} | {ece} |")
        found += 1
    lines.append("")
    lines.append(f"Algorithms with a test file: {found} of {len(ALGOS)}.")
    lines.append("")
    text = "\n".join(lines)
    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
