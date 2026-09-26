"""Write one algorithm results note from its saved test metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT.parent / "paper_evidence"

NOTES = {
    "ogpd": "Teacher prefixes are computed on the training batches only. Test evaluation does not read labels into the policy.",
    "vcie": "The set encoder has no view-position embeddings. A collapsed constant prediction is invariant to order for a trivial reason and is reported as measured.",
    "apc": "Prefix length grows with epoch and with easy batches. Comparison against the other algorithms is the random-prefix control, because those runs use mixed random masks.",
    "sfaq": "JaalTaka has no hologram, watermark, thread, microprint, or serial labels. Security-feature detection is NOT_MEASURED. The measured head regresses generic contrast, color, and sharpness statistics.",
    "igcr": "The extra term is an InfoNCE agreement between the first two real views of the same note. It does not use entropy labels.",
    "ugf": "View gates are a function of uncertainty, quality, and a diversity proxy. The classifier is trained end to end.",
    "ndal": "Per-note loss weight is the detached batch loss, clamped to [0.25, 4].",
    "sfpl": "SIMULATED. Ten client ids change the maximum prefix length inside one process. There is no network. A separate FedAvg arm and a FedAvg+EWC arm were not trained in this run, so those two comparisons are NOT_MEASURED.",
    "cvs": "The training target is the absolute change in the genuine-minus-counterfeit logit when one view is masked. That is an intervention score, not a fitted causal graph. Do-calculus identification is NOT_MEASURED.",
    "mtpt": "Shared trunk with an authenticity head and a generic quality-proxy head. Denomination and emotion are NOT_MEASURED because those labels are not in the JaalTaka note records used here.",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True)
    args = parser.parse_args()
    algo = args.algo.lower()
    path = ROOT / "results" / "novel" / algo / "seed42" / "test" / "test_metrics.json"
    lines = [f"# {algo.upper()} results", "", NOTES.get(algo, ""), ""]
    if not path.is_file():
        reason_path = ROOT / "results" / "novel" / algo / "seed42" / "NOT_MEASURED.txt"
        reason = reason_path.read_text(encoding="utf-8").strip() if reason_path.is_file() else "test_metrics.json is missing"
        lines.append(f"Status: NOT_MEASURED. {reason}")
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        lines.append(f"Source: `{path.relative_to(ROOT.parent)}`")
        lines.append("")
        lines.append("| views | accuracy | macro F1 | ECE | n |")
        lines.append("|---:|---:|---:|---:|---:|")
        for row in payload["views"]:
            lines.append(
                f"| {row['k']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['ece']:.4f} | {row['n']} |"
            )
        lines.append("")
        lines.append("Status: MEASURED. Seed 42. Threshold 0.5 was not tuned on test.")
    text = "\n".join(lines) + "\n"
    for folder in (PAPER, ROOT / "paper_evidence"):
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / f"{algo.upper()}_RESULTS.md"
        dest.write_text(text, encoding="utf-8")
        print(dest)


if __name__ == "__main__":
    main()
