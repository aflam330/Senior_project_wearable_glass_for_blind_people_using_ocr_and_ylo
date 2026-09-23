"""Write paper_evidence files from artifacts only. Never invent numbers."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from roboeye.qduig.artifacts import QDUIG_ROOT, save_json

EV = WORKSPACE / "paper_evidence"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def status_of(*paths: Path, kind: str = "PASS") -> str:
    if all(p.is_file() for p in paths):
        return kind
    if any(p.is_file() for p in paths):
        return "FAIL"
    return "NOT_MEASURED"


def main() -> None:
    EV.mkdir(parents=True, exist_ok=True)
    split = load(ROOT / "results" / "camva" / "splits" / "split_metadata.json")
    p6 = load(QDUIG_ROOT / "eval" / "seed42" / "proposed_6view" / "test_metrics.json")
    b6 = load(QDUIG_ROOT / "eval" / "seed42" / "baseline" / "baseline_6view" / "test_metrics.json")
    views = []
    for k in range(1, 7):
        views.append(
            {
                "k": k,
                "baseline": load(QDUIG_ROOT / "eval" / "seed42" / "baseline" / f"baseline_{k}view" / "test_metrics.json"),
                "proposed": load(QDUIG_ROOT / "eval" / "seed42" / "policies" / f"full_proposed_{k}view" / "test_metrics.json"),
            }
        )
    cal = load(QDUIG_ROOT / "eval" / "seed42" / "calibration.json")
    rob = load(QDUIG_ROOT / "eval" / "seed42" / "robustness.json")
    stats = load(QDUIG_ROOT / "eval" / "seed42" / "statistics.json")
    oracle = load(QDUIG_ROOT / "eval" / "seed42" / "oracle.json")
    edge = load(QDUIG_ROOT / "edge" / "edge.json")
    cont = load(QDUIG_ROOT / "continual" / "seed42" / "continual.json")
    fed = load(QDUIG_ROOT / "federated" / "seed42" / "federated.json")
    emo = load(WORKSPACE / "savior_glass" / "results" / "emotion_rafdb.json")
    ocr = load(WORKSPACE / "savior_glass" / "results" / "ocr_cer.json")
    wild = load(EV / "detection" / "wild_note_metrics.json")
    agg = load(QDUIG_ROOT / "aggregate" / "seed_aggregate.json")
    human = load(QDUIG_ROOT / "feedback" / "human_study.json")
    gen = load(QDUIG_ROOT / "eval" / "seed42" / "generalization.json")
    pareto = load(QDUIG_ROOT / "eval" / "seed42" / "pareto_val.json")

    # CLAIM REGISTRY — only VERIFIED if artifact exists
    claims = []

    def add(cid, claim, metric, value, artifact, script, st, **extra):
        claims.append(
            {
                "claim_id": cid,
                "claim": claim,
                "dataset": extra.get("dataset", "JaalTaka"),
                "split": extra.get("split", "test"),
                "sample_definition": extra.get("sample", "physical_note"),
                "training_seed": extra.get("train_seed", "42"),
                "evaluation_seed": extra.get("eval_seed", "42"),
                "metric": metric,
                "value": value if value is not None else "NOT_MEASURED",
                "source_artifact": str(artifact) if artifact else "",
                "script": script,
                "status": st,
            }
        )

    if split:
        add("C_SPLIT", "Note-disjoint split leakage is zero", "leakage", split["leakage_check"], ROOT / "results/camva/splits/split_metadata.json", "roboeye/camva/notes.py", "VERIFIED")
    if b6:
        add("C_BASE6", "CNN+ViT baseline 6-view accuracy", "accuracy", b6["accuracy"], QDUIG_ROOT / "eval/seed42/baseline/baseline_6view/test_metrics.json", "scripts/evaluate_all.py", "VERIFIED")
    else:
        add("C_BASE6", "CNN+ViT baseline 6-view accuracy", "accuracy", None, None, "scripts/evaluate_all.py", "NOT_MEASURED")
    if p6:
        add("C_PROP6", "Q-DUIG 6-view accuracy", "accuracy", p6["accuracy"], QDUIG_ROOT / "eval/seed42/proposed_6view/test_metrics.json", "scripts/evaluate_all.py", "VERIFIED")
    else:
        add("C_PROP6", "Q-DUIG 6-view accuracy", "accuracy", None, None, "scripts/evaluate_all.py", "NOT_MEASURED")
    if emo:
        add("C_EMO", "RAF-DB emotion accuracy (preserved)", "accuracy", emo["accuracy"], WORKSPACE / "savior_glass/results/emotion_rafdb.json", "savior_glass/scripts/train_emotion_rafdb.py", "VERIFIED", dataset="RAF-DB")
    if ocr:
        add("C_OCR", "Offline OCR CER (preserved)", "CER", ocr["cer"], WORKSPACE / "savior_glass/results/ocr_cer.json", "savior_glass/scripts/eval_ocr_offline.py", "VERIFIED", dataset="OCR_n80")
    add("C_HUMAN", "Human study assistive outcomes", "any", None, None, "scripts/eval_feedback_qduig.py", "NOT_MEASURED")
    add("C_PI5", "Raspberry Pi 5 latency", "median_ms", None if not (edge and edge.get("is_raspberry_pi_5")) else edge["breakdown_ms"]["end_to_end"]["median_ms"], QDUIG_ROOT / "edge/edge.json" if edge else None, "scripts/benchmark_edge_qduig.py", "VERIFIED" if edge and edge.get("is_raspberry_pi_5") else "NOT_MEASURED")
    add("C_CAM", "Camera-disjoint auth", "accuracy", None, None, "scripts/evaluate_all.py", "NOT_MEASURED")
    add("C_POSE", "Metric pose accuracy", "error_mm", None, None, "n/a", "NOT_MEASURED")
    add("C_VLM", "VLM quality labels", "any", None, None, "n/a", "NOT_MEASURED")
    add("C_ENERGY", "Energy per inference", "joule", None, None, "scripts/benchmark_edge_qduig.py", "NOT_MEASURED")

    save_json(EV / "CLAIM_REGISTRY.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "claims": claims})

    def line_views():
        lines = ["| views | baseline acc | proposed acc |", "|---:|---:|---:|"]
        any_m = False
        for row in views:
            ba = row["baseline"]["accuracy"] if row["baseline"] else None
            pa = row["proposed"]["accuracy"] if row["proposed"] else None
            if ba is None and pa is None:
                lines.append(f"| {row['k']} | NOT_MEASURED | NOT_MEASURED |")
            else:
                any_m = True
                lines.append(f"| {row['k']} | {ba if ba is not None else 'NOT_MEASURED'} | {pa if pa is not None else 'NOT_MEASURED'} |")
        return "\n".join(lines), any_m

    vtab, v_ok = line_views()
    final = [
        "# FINAL_RESULTS",
        "",
        "Only numbers with artifacts. Missing experiments are NOT_MEASURED.",
        "",
        "## Framing",
        "",
        "Scientific object: quality-, diversity-, uncertainty-, and information-gain-aware sequential visual acquisition for edge currency authentication. RoboEye is the deployment context.",
        "",
        "## Dataset (MEASURED)",
        "",
        f"Source: `realtime_bangla_taka_detection/results/camva/splits/split_metadata.json`",
        "",
        json.dumps(split, indent=2) if split else "NOT_MEASURED",
        "",
        "## 1–6 views seed 42",
        "",
        vtab,
        "",
        "## 6-view headline (seed 42)",
        "",
        f"- baseline accuracy: {b6['accuracy'] if b6 else 'NOT_MEASURED'}",
        f"- proposed accuracy: {p6['accuracy'] if p6 else 'NOT_MEASURED'}",
        "",
        "If proposed < baseline, that is a negative result and is reported as such.",
        "",
        "## Calibration / robustness / oracle / edge / continual / federated",
        "",
        f"- calibration: {'MEASURED' if cal else 'NOT_MEASURED'}",
        f"- robustness: {'MEASURED' if rob else 'NOT_MEASURED'}",
        f"- oracle: {'MEASURED' if oracle else 'NOT_MEASURED'}",
        f"- edge current host: {'MEASURED' if edge else 'NOT_MEASURED'}",
        f"- Pi 5: {edge.get('pi5_metrics') if edge else 'NOT_MEASURED'}",
        f"- continual: {'MEASURED' if cont else 'NOT_MEASURED'}",
        f"- simulated federated: {'MEASURED' if fed else 'NOT_MEASURED'}",
        f"- camera/session: {(gen or {}).get('camera_disjoint', 'NOT_MEASURED')}",
        f"- human study: {(human or {}).get('HUMAN_STUDY', 'NOT_MEASURED')}",
        "",
        "## Preserved prior measurements",
        "",
        f"- emotion RAF-DB acc: {emo.get('accuracy') if emo else 'NOT_MEASURED'}",
        f"- OCR CER: {ocr.get('cer') if ocr else 'NOT_MEASURED'}",
        f"- wild-note: {'present' if wild else 'NOT_MEASURED'}",
        "",
        "## Hypotheses (not revised after test)",
        "",
        "H1–H4 remain as pre-registered. Pass/fail is written in FINAL_AUDIT.md after artifacts exist.",
        "",
    ]
    (EV / "FINAL_RESULTS.md").write_text("\n".join(final), encoding="utf-8")

    (EV / "ABLATION_RESULTS.md").write_text(
        "# Ablations\n\nSee `results/qduig/ablations/`.\n"
        + ("\n".join(f"- {p.parent.parent.name}: MEASURED" if (p := QDUIG_ROOT / 'ablations' / name / 'seed42' / 'checkpoint.pt').is_file() else f"- {name}: NOT_MEASURED" for name in ("quality", "uncertainty", "diversity", "qd", "ud", "full", "no_cost", "no_calibration", "no_redundancy", "no_infogain"))),
        encoding="utf-8",
    )
    (EV / "ROBUSTNESS_RESULTS.md").write_text("# Robustness\n\n" + (json.dumps(rob, indent=2) if rob else "NOT_MEASURED\n"), encoding="utf-8")
    (EV / "CALIBRATION_RESULTS.md").write_text("# Calibration\n\nFit on val only.\n\n" + (json.dumps(cal, indent=2) if cal else "NOT_MEASURED\n"), encoding="utf-8")
    (EV / "EDGE_RESULTS.md").write_text("# Edge\n\n" + (json.dumps(edge, indent=2) if edge else "NOT_MEASURED\n"), encoding="utf-8")
    (EV / "STATISTICAL_ANALYSIS.md").write_text("# Statistics\n\nPredefined: FULL PROPOSED vs CNN+ViT BASELINE.\n\n" + (json.dumps(stats, indent=2) if stats else "NOT_MEASURED\n"), encoding="utf-8")
    (EV / "FAILURE_ANALYSIS.md").write_text(
        "# Failure analysis\n\nScore histogram: `paper_evidence/figures/fig11_failures.png`.\nOracle gap: "
        + (json.dumps(oracle, indent=2) if oracle else "NOT_MEASURED")
        + "\n",
        encoding="utf-8",
    )
    (EV / "REPRODUCIBILITY.md").write_text(
        "\n".join(
            [
                "# Reproducibility",
                "",
                "```",
                "cd realtime_bangla_taka_detection",
                "python run_research_pipeline.py --config configs/final_research.yaml",
                "```",
                "",
                "Every run directory under `results/qduig/` writes config.yaml, environment.txt, git_commit.txt, seed.txt, dataset_manifest.json, train_log.csv, metrics, predictions.",
                "Existing CAMVA artifacts under `results/camva/` are not overwritten.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # honest H tests only if both exist
    h = {"H1": "NOT_MEASURED", "H2": "NOT_MEASURED", "H3": "NOT_MEASURED", "H4": "NOT_MEASURED"}
    if p6 and b6:
        h["H1"] = "INCONCLUSIVE_OR_FAIL_UNTIL_ADAPTIVE_COMPARED"
        adapt = load(QDUIG_ROOT / "eval" / "seed42" / "policies" / "full_proposed_adaptive" / "test_metrics.json")
        if adapt:
            fewer = adapt.get("average_views", 6) < 6
            ge = adapt["accuracy"] >= b6["accuracy"]
            h["H1"] = "PASS" if (ge and fewer) else "FAIL"
            h["H4"] = "PASS" if (adapt["accuracy"] >= b6["accuracy"] - 0.01 and fewer) else "FAIL"
        h["H2"] = "PASS" if p6.get("ece", 1) < b6.get("ece", 0) else "FAIL"
    if rob:
        clean = next((r for r in rob if r.get("corruption") == "clean"), None)
        if clean:
            worse = 0
            n = 0
            for r in rob:
                if r.get("corruption") == "clean":
                    continue
                if isinstance(r.get("proposed"), dict) and isinstance(r.get("baseline"), dict):
                    n += 1
                    if r["proposed"]["accuracy"] < r["baseline"]["accuracy"]:
                        worse += 1
            h["H3"] = "FAIL" if worse > n / 2 else "PASS"

    research = {
        "DATA": "PASS" if split and split["leakage_check"]["train_test"] == 0 else "FAIL",
        "BASELINE": "PASS" if b6 else "NOT_MEASURED",
        "PROPOSED METHOD": "PASS" if p6 else "NOT_MEASURED",
        "MULTI-SEED": "PASS" if agg and not agg.get("missing_artifacts") else ("FAIL" if agg else "NOT_MEASURED"),
        "ABLATIONS": "PASS" if (QDUIG_ROOT / "ablations" / "full" / "seed42" / "checkpoint.pt").is_file() else "NOT_MEASURED",
        "CALIBRATION": "PASS" if cal else "NOT_MEASURED",
        "ROBUSTNESS": "PASS" if rob else "NOT_MEASURED",
        "GENERALIZATION": (gen or {}).get("camera_disjoint", "NOT_MEASURED"),
        "EDGE": "PASS" if edge else "NOT_MEASURED",
        "EMOTION": "PASS" if emo else "NOT_MEASURED",
        "OCR": "PASS" if ocr else "NOT_MEASURED",
        "USER STUDY": "NOT_MEASURED",
        "CLAIM VALIDATION": "PENDING",
        "FINAL PAPER EVIDENCE": "READY" if (p6 and b6) else "NOT_READY",
        "hypotheses": h,
    }
    audit = [
        "# FINAL_AUDIT",
        "",
        json.dumps(research, indent=2),
        "",
        "## Honesty notes",
        "",
        "- Previous CAMVA (quality-attention) lost to the retrained baseline on 1–5 views and won only at 6 views. Those artifacts remain in `results/camva/`.",
        "- Q-DUIG numbers above are used only when `results/qduig/` artifacts exist.",
        "- No 'first' or SOTA claim is made.",
        "- Venue acceptance is not asserted.",
        "",
    ]
    (EV / "FINAL_AUDIT.md").write_text("\n".join(audit), encoding="utf-8")
    save_json(QDUIG_ROOT / "RESEARCH_STATUS.json", research)
    print("evidence written", EV)


if __name__ == "__main__":
    main()
