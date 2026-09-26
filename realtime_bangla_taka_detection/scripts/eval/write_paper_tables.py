"""LaTeX tables from JSON artifacts. Missing cells are NOT_MEASURED."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT.parent / "paper_evidence"
TABLES = PAPER / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def _load(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _acc_views(path: Path) -> dict[int, float] | None:
    payload = _load(path)
    if not payload:
        return None
    rows = payload.get("views") or payload.get("proposed")
    if not rows:
        return None
    return {int(r["k"]): float(r["accuracy"]) for r in rows}


def _fmt(v) -> str:
    if v is None:
        return "NOT\\_MEASURED"
    return f"{v:.4f}"


def _write(name: str, body: str) -> None:
    (TABLES / name).write_text(body, encoding="utf-8")


def _baseline() -> dict[int, float]:
    out = {}
    for k in range(1, 7):
        payload = _load(ROOT / "results" / "qduig" / "eval" / "seed42" / "baseline" / f"baseline_{k}view" / "test_metrics.json")
        if payload:
            out[k] = float(payload["accuracy"])
    return out


def _novel(algo: str, seed: int, v2: bool) -> dict[int, float] | None:
    root = ROOT / "results" / ("novel_v2" if v2 else "novel") / algo / f"seed{seed}" / "test" / "test_metrics.json"
    if seed == 42 and not root.is_file():
        root = ROOT / "results" / "novel" / algo / "seed42" / "test" / "test_metrics.json"
    return _acc_views(root)


def _prmvt(seed: int) -> dict[int, float] | None:
    return _acc_views(ROOT / "results" / "qduig" / "prefix_ft" / f"seed{seed}" / "test_views" / "views_1_to_6.json")


SERIES = [
    ("Baseline", "base", False),
    ("PRMVT", "prmvt", False),
    ("NDAL", "ndal", True),
    ("PRAVT", "pravt", True),
    ("VAT", "vat", True),
    ("UGF", "ugf", False),
    ("CVS", "cvs", True),
    ("APC", "apc", True),
    ("MTPT", "mtpt", True),
    ("CRIS", "cris", True),
    ("MAVT", "mavt", True),
    ("SFAQ", "sfaq", False),
    ("IGCR", "igcr", False),
    ("SAVS", "savs", True),
    ("VCIE", "vcie", True),
    ("OGPD", "ogpd", False),
    ("SFPL", "sfpl", True),
]


def _seed42(name: str, key: str, v2: bool) -> dict[int, float] | None:
    if key == "base":
        return _baseline() or None
    if key == "prmvt":
        return _prmvt(42)
    return _novel(key, 42, v2)


def table_dataset() -> None:
    meta = _load(ROOT / "results" / "camva" / "splits" / "split_metadata.json")
    if not meta:
        _write("table1_dataset.tex", "% NOT_MEASURED\n")
        return
    rows = [
        ("Genuine notes", meta["n_unique_genuine_notes"]),
        ("Counterfeit notes", meta["n_unique_counterfeit_notes"]),
        ("Notes", meta["n_notes_total"]),
        ("Images", meta["n_images_total"]),
        ("Train notes", meta["train"]["n_notes"]),
        ("Validation notes", meta["val"]["n_notes"]),
        ("Test notes", meta["test"]["n_notes"]),
    ]
    body = ["\\begin{tabular}{lr}", "\\hline", "Quantity & Count \\\\", "\\hline"]
    for label, value in rows:
        body.append(f"{label} & {value} \\\\")
    body += ["\\hline", "\\end{tabular}", ""]
    _write("table1_dataset.tex", "\n".join(body))


def table_all_views() -> None:
    lines = ["\\begin{tabular}{lrrrrrr}", "\\hline", "Method & 1 & 2 & 3 & 4 & 5 & 6 \\\\", "\\hline"]
    for name, key, v2 in SERIES:
        acc = _seed42(name, key, v2)
        cells = " & ".join(_fmt(None if not acc else acc.get(k)) for k in range(1, 7))
        lines.append(f"{name} & {cells} \\\\")
    lines += ["\\hline", "\\end{tabular}", ""]
    _write("table3_views.tex", "\n".join(lines))


def table_multiseed() -> None:
    lines = ["\\begin{tabular}{lrr}", "\\hline", "Method & 1-view mean $\\pm$ sd & 6-view mean $\\pm$ sd \\\\", "\\hline"]
    targets = [("PRMVT", "prmvt", False), ("NDAL", "ndal", True), ("PRAVT", "pravt", True), ("VAT", "vat", True), ("UGF", "ugf", False), ("CVS", "cvs", True)]
    for name, key, v2 in targets:
        chunks = []
        for k in (1, 6):
            vals = []
            for seed in (42, 43, 44):
                acc = _prmvt(seed) if key == "prmvt" else _novel(key, seed, v2)
                if acc and k in acc:
                    vals.append(acc[k])
            if len(vals) < 2:
                chunks.append("NOT\\_MEASURED")
            else:
                import statistics
                chunks.append(f"{statistics.mean(vals):.4f} $\\pm$ {statistics.stdev(vals):.4f}")
        lines.append(f"{name} & {chunks[0]} & {chunks[1]} \\\\")
    lines += ["\\hline", "\\end{tabular}", ""]
    _write("table2_multiseed_top.tex", "\n".join(lines))


def table_ablation() -> None:
    path = PAPER / "ABLATION_RESULTS.md"
    lines = ["\\begin{tabular}{lrrrrrr}", "\\hline", "Config & 1 & 2 & 3 & 4 & 5 & 6 \\\\", "\\hline"]
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.startswith("| ") or raw.startswith("| config") or raw.startswith("|---"):
                continue
            cells = [c.strip() for c in raw.strip("|").split("|")]
            if len(cells) < 7:
                continue
            name = cells[0].replace("_", "\\_")
            nums = " & ".join(cells[1:7])
            lines.append(f"{name} & {nums} \\\\")
    if len(lines) == 4:
        lines.append("NOT\\_MEASURED & & & & & & \\\\")
    lines += ["\\hline", "\\end{tabular}", ""]
    _write("table4_ablation.tex", "\n".join(lines))


def table_calibration() -> None:
    payload = _load(ROOT / "results" / "calibration" / "suite_seed42.json")
    lines = ["\\begin{tabular}{llrrrr}", "\\hline", "Model & Method & ECE & Adaptive ECE & Brier & NLL \\\\", "\\hline"]
    if not payload:
        lines.append("NOT\\_MEASURED & & & & & \\\\")
    else:
        for model, blob in payload.items():
            for method, row in blob["methods"].items():
                ece = row.get("ece_10", row.get("ece"))
                method_tex = method.replace("_", r"\_")
                lines.append(
                    f"{model} & {method_tex} & {_fmt(ece)} & {_fmt(row.get('adaptive_ece'))} & {_fmt(row.get('brier'))} & {_fmt(row.get('nll'))} \\\\"
                )
    lines += ["\\hline", "\\end{tabular}", ""]
    _write("table5_calibration.tex", "\n".join(lines))


def table_robust() -> None:
    payload = _load(ROOT / "results" / "robustness" / "top_seed42.json")
    lines = ["\\begin{tabular}{llrrr}", "\\hline", "Corruption & Severity & PRMVT & NDAL & Baseline \\\\", "\\hline"]
    if not payload:
        lines.append("NOT\\_MEASURED & & & & \\\\")
    else:
        seen = []
        for row in payload["rows"]:
            if int(row["k"]) != 6:
                continue
            lines.append(
                f"{row['corruption']} & {row['severity']} & {_fmt(row.get('prmvt'))} & {_fmt(row.get('ndal'))} & {_fmt(row.get('baseline'))} \\\\"
            )
            seen.append(row["corruption"])
        if not seen:
            lines.append("NOT\\_MEASURED & & & & \\\\")
    lines += ["\\hline", "\\end{tabular}", ""]
    _write("table6_robustness.tex", "\n".join(lines))
    sweep = _load(ROOT / "results" / "robustness" / "severity_seed42.json")
    sweep_lines = ["\\begin{tabular}{llrrr}", "\\hline", "Corruption & Severity & PRMVT & NDAL & Baseline \\\\", "\\hline"]
    if not sweep:
        sweep_lines.append("NOT\\_MEASURED & & & & \\\\")
    else:
        for row in sweep["rows"]:
            sweep_lines.append(
                f"{row['corruption']} & {row['severity']} & {_fmt(row.get('prmvt'))} & {_fmt(row.get('ndal'))} & {_fmt(row.get('baseline'))} \\\\"
            )
    sweep_lines += ["\\hline", "\\end{tabular}", ""]
    _write("table6b_severity.tex", "\n".join(sweep_lines))


def _paired_rows() -> list[list[str]]:
    path = PAPER / "STATISTICAL_ANALYSIS.md"
    if not path.is_file():
        return []
    rows = []
    started = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("| algorithm | views | n |"):
            started = True
            continue
        if not started or not raw.startswith("| ") or raw.startswith("|---"):
            continue
        cells = [c.strip() for c in raw.strip("|").split("|")]
        if len(cells) >= 7:
            rows.append(cells)
    return rows


def _paired_tex(picked: list[list[str]]) -> str:
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\hline",
        "Algorithm & Views & McNemar p & Bonferroni p & Bootstrap 95\\% CI & Wilcoxon p \\\\",
        "\\hline",
    ]
    if not picked:
        lines.append("NOT\\_MEASURED & & & & & \\\\")
    for cells in picked:
        algo = cells[0].replace("_", "\\_")
        lines.append(f"{algo} & {cells[1]} & {cells[3]} & {cells[4]} & {cells[5]} & {cells[6]} \\\\")
    lines += ["\\hline", "\\end{tabular}", ""]
    return "\n".join(lines)


def table_stats() -> None:
    rows = _paired_rows()
    _write("table7_paired_6view.tex", _paired_tex([row for row in rows if row[1] == "6"]))
    _write("table8_significance.tex", _paired_tex(rows))


def table_negative() -> None:
    lines = ["\\begin{tabular}{lrr}", "\\hline", "Method & 1-view & 6-view \\\\", "\\hline"]
    for name, key, v2 in (("SFPL v2", "sfpl", True), ("OGPD v1", "ogpd", False), ("OGPD v2", "ogpd", True), ("VCIE v2", "vcie", True)):
        acc = _novel(key, 42, v2)
        lines.append(f"{name} & {_fmt(None if not acc else acc.get(1))} & {_fmt(None if not acc else acc.get(6))} \\\\")
    lines += ["\\hline", "\\end{tabular}", ""]
    _write("table9_negative.tex", "\n".join(lines))


def table_limits() -> None:
    _write("table10_limitations.tex", "\n".join([
        "\\begin{tabular}{ll}",
        "\\hline",
        "Item & Status \\\\",
        "\\hline",
        "Pi 5 latency, temperature, sustained run & NOT\\_MEASURED \\\\",
        "Mobile / TFLite & NOT\\_MEASURED \\\\",
        "User study & NOT\\_MEASURED \\\\",
        "PAC-Bayes penalty, Gaussian posterior, fixed grid & 57.5893 (vacuous) \\\\",
        "External SOTA on this split & NOT\\_MEASURED \\\\",
        "\\hline",
        "\\end{tabular}",
        "",
    ]))


def main() -> None:
    table_dataset()
    table_multiseed()
    table_all_views()
    table_ablation()
    table_calibration()
    table_robust()
    table_stats()
    table_negative()
    table_limits()
    print(TABLES)


if __name__ == "__main__":
    main()
