"""Build 11 publication tables from JSON/CSV artifacts only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from roboeye.qduig.artifacts import QDUIG_ROOT, save_json

TABLE_DIR = WORKSPACE / "paper_evidence" / "tables"
EVID = WORKSPACE / "paper_evidence"


def load(path: Path):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines) + "\n"


def fmt(x, nd=4):
    if x is None:
        return "NOT_MEASURED"
    if isinstance(x, str):
        return x
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def write(name: str, title: str, body: str, source: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / name).write_text(f"# {title}\n\nSource: `{source}`\n\n{body}\n", encoding="utf-8")


def main() -> None:
    split = load(ROOT / "results" / "camva" / "splits" / "split_metadata.json")
    if split:
        rows = [
            ["physical notes", str(split["n_notes_total"])],
            ["images", str(split["n_images_total"])],
            ["genuine notes", str(split["n_unique_genuine_notes"])],
            ["counterfeit notes", str(split["n_unique_counterfeit_notes"])],
            ["views/note", str(split["views_per_note_all"])],
            ["train notes", str(split["train"]["n_notes"])],
            ["val notes", str(split["val"]["n_notes"])],
            ["test notes", str(split["test"]["n_notes"])],
            ["train genuine/counterfeit", f"{split['train']['n_genuine_notes']}/{split['train']['n_counterfeit_notes']}"],
            ["leakage train∩val∩test", str(split["leakage_check"])],
        ]
        write("table01_dataset.md", "Table 1. Dataset statistics", md_table(["item", "value"], rows), "results/camva/splits/split_metadata.json")

    b6 = load(QDUIG_ROOT / "eval" / "seed42" / "baseline" / "baseline_6view" / "test_metrics.json")
    p6 = load(QDUIG_ROOT / "eval" / "seed42" / "proposed_6view" / "test_metrics.json")
    keys = ["accuracy", "precision", "recall", "macro_f1", "balanced_accuracy", "roc_auc", "pr_auc", "ece", "brier"]
    if b6 or p6:
        rows = [[k, fmt((b6 or {}).get(k)), fmt((p6 or {}).get(k))] for k in keys]
        write("table02_baseline_vs_proposed.md", "Table 2. CNN+ViT baseline vs proposed (6 views, seed 42)", md_table(["metric", "baseline", "proposed"], rows), "results/qduig/eval/seed42/")

    rows = []
    for k in range(1, 7):
        b = load(QDUIG_ROOT / "eval" / "seed42" / "baseline" / f"baseline_{k}view" / "test_metrics.json")
        p = load(QDUIG_ROOT / "eval" / "seed42" / "policies" / f"full_proposed_{k}view" / "test_metrics.json")
        rows.append([str(k), fmt((b or {}).get("accuracy")), fmt((p or {}).get("accuracy")), fmt((p or {}).get("average_views"))])
    write("table03_views.md", "Table 3. 1–6 view performance", md_table(["views", "baseline_acc", "proposed_acc", "proposed_avg_views"], rows), "results/qduig/eval/seed42/")

    abl_rows = []
    for name in ("baseline", "quality", "uncertainty", "diversity", "qd", "ud", "full", "no_cost", "no_calibration", "no_redundancy", "no_infogain"):
        m = load(QDUIG_ROOT / "ablations" / name / "seed42" / "eval" / "test_metrics.json")
        abl_rows.append([name, fmt((m or {}).get("accuracy")), fmt((m or {}).get("macro_f1"))])
    write("table04_ablation.md", "Table 4. Ablation study", md_table(["config", "accuracy", "macro_f1"], abl_rows), "results/qduig/ablations/")

    rob = load(QDUIG_ROOT / "eval" / "seed42" / "robustness.json")
    if rob:
        rrows = [[r["corruption"], fmt(r.get("baseline", {}).get("accuracy") if isinstance(r.get("baseline"), dict) else None), fmt(r.get("proposed", {}).get("accuracy") if isinstance(r.get("proposed"), dict) else None), fmt(r.get("baseline_degradation")), fmt(r.get("proposed_degradation"))] for r in rob]
        write("table05_robustness.md", "Table 5. Robustness", md_table(["corruption", "baseline_acc", "proposed_acc", "Δ baseline", "Δ proposed"], rrows), "results/qduig/eval/seed42/robustness.json")
    else:
        write("table05_robustness.md", "Table 5. Robustness", "NOT_MEASURED\n", "missing")

    gen = load(QDUIG_ROOT / "eval" / "seed42" / "generalization.json")
    write("table06_generalization.md", "Table 6. Cross-camera/session", json.dumps(gen or {"status": "NOT_MEASURED"}, indent=2), "results/qduig/eval/seed42/generalization.json")

    cal = load(QDUIG_ROOT / "eval" / "seed42" / "calibration.json")
    if cal and "test" in cal:
        crows = [[m, fmt(v.get("ece")), fmt(v.get("adaptive_ece")), fmt(v.get("brier")), fmt(v.get("nll"))] for m, v in cal["test"].items()]
        write("table07_calibration.md", "Table 7. Calibration", md_table(["method", "ECE", "adaptive ECE", "Brier", "NLL"], crows), "results/qduig/eval/seed42/calibration.json")
    else:
        write("table07_calibration.md", "Table 7. Calibration", "NOT_MEASURED\n", "missing")

    edge = load(QDUIG_ROOT / "edge" / "edge.json")
    if edge:
        erows = [
            ["host", str(edge.get("host"))],
            ["device", str(edge.get("device"))],
            ["pi5", str(edge.get("pi5_metrics"))],
            ["energy", str(edge.get("energy"))],
            ["e2e median ms", fmt((edge.get("breakdown_ms") or {}).get("end_to_end", {}).get("median_ms"))],
            ["e2e p95 ms", fmt((edge.get("breakdown_ms") or {}).get("end_to_end", {}).get("p95_ms"))],
            ["model bytes", str(edge.get("model_size_bytes"))],
        ]
        write("table08_edge.md", "Table 8. Edge deployment", md_table(["item", "value"], erows), "results/qduig/edge/edge.json")
    else:
        write("table08_edge.md", "Table 8. Edge deployment", "NOT_MEASURED\n", "missing")

    emo = load(WORKSPACE / "savior_glass" / "results" / "emotion_rafdb.json")
    if emo:
        write("table09_emotion.md", "Table 9. Emotion recognition (RAF-DB, preserved)", md_table(["metric", "value"], [["n", str(emo.get("n_test"))], ["accuracy", fmt(emo.get("accuracy"))], ["macro_f1", fmt(emo.get("macro_f1"))]]), "savior_glass/results/emotion_rafdb.json")
    else:
        write("table09_emotion.md", "Table 9. Emotion", "NOT_MEASURED\n", "missing")

    ocr = load(WORKSPACE / "savior_glass" / "results" / "ocr_cer.json")
    wild = load(EVID / "detection" / "wild_note_metrics.json")
    e2e_rows = [
        ["OCR CER", fmt((ocr or {}).get("cer"))],
        ["wild detection", fmt((wild or {}).get("detection_recall") or (wild or {}).get("detection"))],
        ["wild top-1 denom", fmt((wild or {}).get("top1") or (wild or {}).get("top_1"))],
        ["auth 6-view proposed", fmt((p6 or {}).get("accuracy"))],
        ["human study", "NOT_MEASURED"],
    ]
    write("table10_e2e.md", "Table 10. End-to-end system (mixed sources, labeled)", md_table(["item", "value"], e2e_rows), "mixed artifacts")

    cont = load(QDUIG_ROOT / "continual" / "seed42" / "continual.json")
    fed = load(QDUIG_ROOT / "federated" / "seed42" / "federated.json")
    write("table11_continual_federated.md", "Table 11. Continual / simulated federated", json.dumps({"continual": cont or "NOT_MEASURED", "federated": fed or "NOT_MEASURED"}, indent=2, default=str), "results/qduig/continual|federated")
    print("tables written", TABLE_DIR)


if __name__ == "__main__":
    main()
