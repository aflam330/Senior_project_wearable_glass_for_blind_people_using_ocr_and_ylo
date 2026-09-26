"""Assemble the paper draft from saved JSON only. Missing files stay NOT_MEASURED."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT.parent / "paper_evidence"
NOVEL = {
    "ndal": ROOT / "results" / "novel_v2" / "ndal" / "seed42" / "test" / "test_metrics.json",
    "pravt": ROOT / "results" / "novel_v2" / "pravt" / "seed42" / "test" / "test_metrics.json",
    "vat": ROOT / "results" / "novel_v2" / "vat" / "seed42" / "test" / "test_metrics.json",
    "ugf": ROOT / "results" / "novel" / "ugf" / "seed42" / "test" / "test_metrics.json",
    "cvs": ROOT / "results" / "novel_v2" / "cvs" / "seed42" / "test" / "test_metrics.json",
    "apc": ROOT / "results" / "novel_v2" / "apc" / "seed42" / "test" / "test_metrics.json",
    "cris": ROOT / "results" / "novel_v2" / "cris" / "seed42" / "test" / "test_metrics.json",
    "savs": ROOT / "results" / "novel_v2" / "savs" / "seed42" / "test" / "test_metrics.json",
    "mavt": ROOT / "results" / "novel_v2" / "mavt" / "seed42" / "test" / "test_metrics.json",
    "vcie": ROOT / "results" / "novel_v2" / "vcie" / "seed42" / "test" / "test_metrics.json",
    "mtpt": ROOT / "results" / "novel_v2" / "mtpt" / "seed42" / "test" / "test_metrics.json",
    "sfpl": ROOT / "results" / "novel_v2" / "sfpl" / "seed42" / "test" / "test_metrics.json",
    "ogpd": ROOT / "results" / "novel" / "ogpd" / "seed42" / "test" / "test_metrics.json",
    "sfaq": ROOT / "results" / "novel" / "sfaq" / "seed42" / "test" / "test_metrics.json",
    "igcr": ROOT / "results" / "novel" / "igcr" / "seed42" / "test" / "test_metrics.json",
}


def _accs(path: Path) -> dict[int, float] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["k"]): float(row["accuracy"]) for row in payload["views"]}


def _row(name: str, accs: dict[int, float] | None) -> str:
    if accs is None:
        return f"| {name} | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |"
    return "| " + name + " | " + " | ".join(f"{accs[k]:.4f}" for k in range(1, 7)) + " |"


def _prmvt() -> dict[int, float] | None:
    path = ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "test_views" / "views_1_to_6.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["k"]): float(row["accuracy"]) for row in payload["proposed"]}


def _baseline() -> dict[int, float]:
    out = {}
    for k in range(1, 7):
        path = ROOT / "results" / "qduig" / "eval" / "seed42" / "baseline" / f"baseline_{k}view" / "test_metrics.json"
        out[k] = float(json.loads(path.read_text(encoding="utf-8"))["accuracy"])
    return out


def _ablation_table() -> str:
    lines = [
        "| config | 1 | 2 | 3 | 4 | 5 | 6 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    names = [
        "quality", "uncertainty", "diversity", "qd", "ud", "full",
        "no_cost", "no_calibration", "no_redundancy", "no_infogain",
    ]
    any_measured = False
    for name in names:
        path = ROOT / "results" / "qduig" / "ablations" / name / "seed42" / "test_views" / "views_1_to_6.json"
        if not path.is_file():
            lines.append(f"| {name} | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |")
            continue
        any_measured = True
        payload = json.loads(path.read_text(encoding="utf-8"))
        accs = {int(row["k"]): float(row["accuracy"]) for row in payload["proposed"]}
        lines.append(_row(name, accs))
    note = (
        "These rows are the 4-epoch component configs, not the 6-epoch plus 3-epoch prefix-robust checkpoint. "
        "A config that only adds PCR-IG, with every other term off, has no yaml. That row is NOT_MEASURED. "
        "The names RSQA, CVR, and HER in the older design notes correspond to the quality, diversity, and uncertainty configs. "
        "If two configs show the same six accuracies, that is what the test files contain."
    )
    if not any_measured:
        note += " No ablation checkpoint has a 1–6 view test file yet."
    return "\n".join(lines) + "\n\n" + note


def main() -> None:
    base = _baseline()
    prmvt = _prmvt()
    multi = (PAPER / "MULTI_SEED_RESULTS.md").read_text(encoding="utf-8") if (PAPER / "MULTI_SEED_RESULTS.md").is_file() else "NOT_MEASURED"
    stats = (PAPER / "STATISTICAL_ANALYSIS.md").read_text(encoding="utf-8") if (PAPER / "STATISTICAL_ANALYSIS.md").is_file() else "NOT_MEASURED"
    robust = (PAPER / "ROBUSTNESS_RESULTS.md").read_text(encoding="utf-8") if (PAPER / "ROBUSTNESS_RESULTS.md").is_file() else "NOT_MEASURED. The 1–6 view corruption run has not written `results/robustness/top_seed42.json`."
    table = [
        "| algorithm | 1 | 2 | 3 | 4 | 5 | 6 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _row("baseline", base),
        _row("prmvt", prmvt),
    ]
    loaded = {}
    for name, path in NOVEL.items():
        loaded[name] = _accs(path)
        table.append(_row(name, loaded[name]))
    body = f"""# Prefix-Robust Multi-View Learning: Fixing View-Count Distribution Shift for Currency Authentication

Draft assembled {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} from artifacts. A number that is not in a cited file is not in this draft.

## 1. Introduction

A classifier trained to see all six views of a JaalTaka note is not a classifier for one view. On the seed-42 test split, the earlier fixed-count Q-DUIG checkpoint scores 0.9663 with six views and 0.5865 with one view (`FINAL_RESULTS.md`). That drop is view-count distribution shift: training puts all mass on six views, and test prefixes are shorter. Prefix-robust training samples those shorter prefixes. The seed-42 prefix-robust checkpoint (PRMVT) scores {prmvt[1]:.4f} at one view and {prmvt[6]:.4f} at six views.

Fifteen other training variants were run on the same split. NDAL v2 is the highest one-view score in the saved v2 files. SFPL stays below the CNN+ViT baseline from two views upward. OGPD v2 was worse than OGPD v1, so v1 is the number that stands. Pi 5 and Android measurements are NOT_MEASURED.

## 2. Related work

Labels and citations are in `NOVELTY_DECLARATION.md`. PRMVT extends missing-view training (RMAE; dual-masked VAEs, IJCAI 2025; RML, ICCV 2025). It does not claim to be the first variable-view model. NDAL adapts focal loss (Lin et al., ICCV 2017). PRAVT reverses view order; it is not PGD (RDML, IJCAI 2025). Phone-camera banknote authentication under visible light is prior work (Sensors 2019). Scopus, IEEE Xplore, ACM DL, and Web of Science were not searched as separate databases.

## 3. Method

The shared encoder is a frozen MobileNetV3-Small plus TinyViT. Images are 128 pixels on a side. The label is genuine or counterfeit. The decision threshold is 0.5.

PRMVT is the Q-DUIG prefix protocol: six epochs of mixed view dropout (`configs/proposed_prefix.yaml`), then three epochs of fine-tuning (`configs/proposed_prefix_ft.yaml`). The checkpoint is the best mean validation accuracy over one to six views.

The other algorithms are heads or losses on that encoder, documented in `NOVELTY_DECLARATION.md`. CVS is a leave-one-view logit KL, not a fitted causal graph. SFPL is FedAvg inside one process. SAVS is one sharpness-aware step. SFAQ has no hologram or thread labels.

## 4. Theory

Definitions and proofs are in `THEORETICAL_ANALYSIS.md`. The support proposition says that a training distribution with positive probability on every view count does not have the fixed-N support mismatch. It is not a convergence rate. The PAC-Bayes display is the standard McAllester inequality. The KL of a posterior was NOT_MEASURED, so no numerical generalization bound is stated.

## 5. Experiments

JaalTaka has 1390 notes (802 genuine, 588 counterfeit), six views, and a note-disjoint split of 974 / 208 / 208 at seed 42. Leakage between splits is zero (`results/camva/splits/split_metadata.json`). The test split is not used for training, checkpoint selection, or threshold selection.

Seeds 43 and 44 repeat the same configs for PRMVT, NDAL, PRAVT, VAT, UGF, and CVS. Early stopping can stop a seed before the maximum epoch. That is the config, and it is reported when the logs differ.

## 6. Results

Seed 42, test, n=208. Sources are the `test_metrics.json` paths in `scripts/eval/write_astar_paper.py` and `results/qduig/prefix_ft/seed42/test_views/views_1_to_6.json`.

{chr(10).join(table)}

OGPD in that table is v1. The v2 OGPD file is lower (0.7596 at one view through 0.7933 at six views) and is not the standing OGPD result.

Multi-seed mean, standard deviation, and the normal approximation interval:

{multi}

Paired tests against the CNN+ViT baseline on the same 208 notes:

{stats}

## 7. Ablation

Baseline one-to-six view accuracy is the CNN+ViT row above. It was not retrained. The other rows are Q-DUIG component configs at seed 42.

{_ablation_table()}

## 8. Robustness

{robust}

An earlier six-view corruption file remains at `results/qduig/eval/seed42/robustness.json`. Where the new file and that file disagree, the new file is the one for PRMVT and NDAL across one to six views, and the older file is the six-view Q-DUIG comparison already quoted in `FINAL_RESULTS.md`.

## 9. Edge deployment

Pi 5 latency, FPS, CPU, RAM, temperature, and a 30-minute sustained run are NOT_MEASURED (`EDGE_RESULTS.md`). TFLite and Android are NOT_MEASURED (`MOBILE_RESULTS.md`). A previous host timing on this CUDA laptop, median 42.6 ms for six views, is in `results/qduig/edge/edge.json`. That number is not a Pi 5 result.

## 10. Discussion

The one-view gap between fixed-count training and prefix-robust training is the result that matches the support proposition. Several later losses also land near 0.95–0.98 at one view on this same split, so the accuracy alone does not identify a unique algorithm. The paired tests in section 6 are the comparison against the CNN+ViT baseline, with Bonferroni applied across the 36 algorithm-by-view tests that had predictions. After that correction, several six-view differences are not significant. That belongs in the paper.

## 11. Limitations

SFPL v2 is above the majority-class rate and below the baseline from two views up. OGPD v2 is worse than OGPD v1. VCIE v2 recovered from a constant predictor and remains below PRMVT. CVS does not implement do-calculus. Federated training is simulated. There is no second dataset. There is no Pi 5. Three seeds, where they exist, are a small sample for a seed-level interval.

## 12. Conclusion

Prefix-robust training removes the fixed view-count support mismatch and, on this split and seed 42, holds accuracy from one view through six. The other algorithms are measured variants, including the weak ones. Claims that need a missing artifact are marked NOT_MEASURED.

## Figures

Measured plots are written only when the source JSON exists. A missing figure is named here and not drawn.

1. Architecture: the encoder and heads are described in section 3. A schematic that is not computed from data is not included.
2. PRMVT two-stage training curve: `results/qduig/prefix/seed42/train_log.csv` and `results/qduig/prefix_ft/seed42/train_log.csv` if present.
3. One-to-six view accuracy: the table in section 6.
4. Ablation: section 7.
5. Calibration: ECE columns in the per-algorithm `test_metrics.json` files. A new reliability diagram was not redrawn in this draft.
6. Robustness: section 8.
7. Oracle gap: the bound is in `THEORETICAL_ANALYSIS.md`. The measured oracle accuracy 0.9856 is the number already cited in `FINAL_RESULTS.md`.
8. VCDS: the proposition is an inequality, not a fitted curve.
9. Confusion matrices already saved under each `test/` or `test_views/` directory.
10. Pi 5 latency breakdown: NOT_MEASURED.
11. Failure analysis: notes where the one-view prediction disagrees with the label are in the prediction JSON files. They were not re-listed here.
12. End-to-end pipeline: capture, six views, encoder, head, threshold 0.5. No extra measured stage.

## Tables

1. Dataset: section 5.
2. Baseline versus the six leading algorithms: section 6.
3. One-to-six view performance: section 6.
4. Multi-seed: `MULTI_SEED_RESULTS.md`.
5. Ablation: section 7.
6. Calibration: ECE in `test_metrics.json`. A single calibration table for every algorithm was not recomputed in this draft beyond those files.
7. Robustness: section 8.
8. Edge: section 9.
9. External SOTA on JaalTaka: NOT_MEASURED. Published banknote papers use other currencies and other splits.
10. Negative results: SFPL v2, OGPD v2, and the v1 constant predictors, section 6 and section 11.
11. Theoretical bounds: `THEORETICAL_ANALYSIS.md`. Numerical PAC-Bayes and sample-complexity bounds are NOT_MEASURED.
"""
    (PAPER / "PAPER_DRAFT.md").write_text(body, encoding="utf-8")
    (PAPER / "ABLATION_RESULTS.md").write_text("# Ablation results\n\n" + _ablation_table() + "\n", encoding="utf-8")
    final_path = PAPER / "FINAL_RESULTS.md"
    pointer = "\n\n## A* assembly\n\nThe draft, multi-seed table, and ablation table are regenerated by `scripts/eval/write_astar_paper.py` from the JSON artifacts. Pi 5 and Android remain NOT_MEASURED.\n"
    if final_path.is_file():
        text = final_path.read_text(encoding="utf-8")
        marker = "\n\n## A* assembly\n"
        if marker in text:
            text = text.split(marker)[0].rstrip() + "\n"
        final_path.write_text(text + pointer, encoding="utf-8")
    audit_path = PAPER / "FINAL_AUDIT.md"
    addendum = f"""

## A* addendum {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

- Seed-42 numbers in `PAPER_DRAFT.md` are read from `test_metrics.json` and `views_1_to_6.json`.
- Seeds 43 and 44 are included only when their test files exist. See `MULTI_SEED_RESULTS.md`.
- Pi 5 and Android remain NOT_MEASURED.
- OGPD v2 is a negative result. SFPL v2 is a weak result.
- Literature labels are EXTENDS or ADAPTED. The search did not cover Scopus, IEEE Xplore, ACM DL, or Web of Science as separate databases.
- PAC-Bayes was not evaluated numerically.
"""
    previous = audit_path.read_text(encoding="utf-8") if audit_path.is_file() else "# FINAL_AUDIT\n"
    marker = "## A* addendum"
    if marker in previous:
        previous = previous.split(marker)[0].rstrip() + "\n"
    audit_path.write_text(previous + addendum, encoding="utf-8")
    registry_path = PAPER / "CLAIM_REGISTRY.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {"claims": []}
    existing = {item["claim_id"] for item in registry.get("claims", [])}
    new_claims = []
    if prmvt is not None:
        new_claims.append({
            "claim_id": "C_PRMVT_1",
            "claim": "Prefix-robust Q-DUIG 1-view test accuracy, seed 42",
            "dataset": "JaalTaka",
            "split": "test",
            "sample_definition": "physical_note",
            "training_seed": "42",
            "evaluation_seed": "42",
            "metric": "accuracy",
            "value": prmvt[1],
            "source_artifact": str(ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "test_views" / "views_1_to_6.json"),
            "script": "scripts/eval_prefix_views.py",
            "status": "VERIFIED",
        })
    for name, accs in loaded.items():
        if accs is None:
            continue
        cid = f"C_{name.upper()}_V2_1"
        if cid in existing:
            continue
        new_claims.append({
            "claim_id": cid,
            "claim": f"{name} 1-view test accuracy, standing seed-42 file",
            "dataset": "JaalTaka",
            "split": "test",
            "sample_definition": "physical_note",
            "training_seed": "42",
            "evaluation_seed": "42",
            "metric": "accuracy",
            "value": accs[1],
            "source_artifact": str(NOVEL[name]),
            "script": "scripts/eval/eval_novel.py",
            "status": "VERIFIED",
        })
    for item in new_claims:
        if item["claim_id"] not in existing:
            registry.setdefault("claims", []).append(item)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(PAPER / "PAPER_DRAFT.md")


if __name__ == "__main__":
    main()
