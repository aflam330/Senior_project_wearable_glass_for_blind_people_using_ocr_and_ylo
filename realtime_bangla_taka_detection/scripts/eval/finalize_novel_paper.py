"""Refresh comparison, per-algorithm notes, claims, and the results section.

Numbers are copied from test_metrics.json. Missing files stay NOT_MEASURED.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT.parent / "paper_evidence"
ALGOS = ["ogpd", "vcie", "apc", "sfaq", "igcr", "ugf", "ndal", "sfpl", "cvs", "mtpt"]
PY = sys.executable

NOVELTY = """# Novelty declaration

Labels describe how the implementation relates to published ideas. They are not performance claims. Experimental numbers are only in `NOVEL_ALGORITHMS_COMPARISON.md` and the per-algorithm results files.

| algorithm | label | why |
|---|---|---|
| OGPD | EXTENDS | Learning using privileged information (Vapnik and Vashist, 2009) and generalized distillation (Lopez-Paz et al., 2015, arXiv:1511.03643) already train with extra inputs that are absent at test time. Multi-view privileged distillation is also published (Lambert et al. style multi-view LUPI, arXiv:1903.03694; MPIRL). This run distills a shortest label-consistent prefix into a stop head. It does not claim a new distillation theory. |
| VCIE | ADAPTED | The Set Transformer (Lee et al., ICML 2019, arXiv:1810.00825) and Deep Sets (Zaheer et al., 2017) are permutation-invariant set encoders. AttSets (Yang et al., 2018) already pools multi-view features with attention. This run uses two self-attention blocks and attention pooling with no view-index embedding. |
| APC | EXTENDS | Curriculum learning (Bengio et al., 2009) and self-paced learning (Kumar, Packer, and Koller, 2010) are prior work. Multi-view self-paced learning (Xu, Tao, and Xu, IJCAI 2015) and self-paced multi-view co-training (JMLR 2020) already treat views as a difficulty axis. The schedule here increases prefix length. |
| SFAQ | ADAPTED | No-reference image quality (BRISQUE, Mittal et al., 2012; NIQE) is prior work. Security-feature detection is NOT_MEASURED: JaalTaka has no hologram, thread, watermark, microprint, or serial labels. The trained head uses generic image statistics only. |
| IGCR | EXTENDS | InfoNCE (van den Oord et al., 2018) is prior work. The added term treats two views of the same note as a positive pair. It is not a labeled-entropy information-gain model. |
| UGF | ADAPTED | Gated and uncertainty-weighted fusion is prior work (attention fusion and uncertainty-aware multi-view models). The gate is a two-layer map of uncertainty, quality, and a diversity proxy. |
| NDAL | ADAPTED | Focal loss (Lin et al., 2017) and hard-example reweighting are prior work. The weight is the detached per-note loss, normalized inside the batch. |
| SFPL | ADAPTED | FedAvg (McMahan et al., 2017) and EWC (Kirkpatrick et al., 2017) are prior work. This run is SIMULATED: one process, client ids only change the allowed prefix length. FedAvg and FedAvg+EWC arms are NOT_MEASURED. |
| CVS | EXTENDS | Leave-one-out intervention scores are a standard ablation. The head is trained to match the change in the class logit when a view is masked. A causal graph and a do-calculus identification result are NOT_MEASURED. |
| MTPT | ADAPTED | Multi-task learning (Caruana, 1997) is prior work. Measured heads are authenticity and a generic quality proxy. Denomination and emotion are NOT_MEASURED. |

The prefix-robust Q-DUIG result remains the strongest measured accuracy result in this repository (see `FINAL_RESULTS.md`). These ten runs do not replace that result. No algorithm is declared first or state of the art.

Searches used to place the labels: oracle distillation and privileged information for view selection; set transformer and permutation-invariant multi-view encoders; curriculum and self-paced multi-view selection; security-feature and banknote image quality; contrastive information gain and mutual information for views; uncertainty-gated and Bayesian multi-view fusion; difficulty-aware and focal losses; federated learning with heterogeneous views; causal and interventional view selection; multi-task prefix transformers.
"""


def _section() -> str:
    comparison = PAPER / "NOVEL_ALGORITHMS_COMPARISON.md"
    body = comparison.read_text(encoding="utf-8") if comparison.is_file() else "NOT_MEASURED"
    return (
        "<!-- NOVEL_ALGORITHMS_START -->\n"
        "## Ten additional algorithms, seed 42\n\n"
        "Prefix-robust training (PRMVT) remains the main accuracy result in this file. "
        "The ten runs below are separate checkpoints. SFAQ does not detect security features. "
        "MTPT does not predict denomination or emotion. SFPL is simulated in one process. "
        "Negative and flat results are left as measured.\n\n"
        + body
        + "\n<!-- NOVEL_ALGORITHMS_END -->\n"
    )


def _splice(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    start = "<!-- NOVEL_ALGORITHMS_START -->"
    end = "<!-- NOVEL_ALGORITHMS_END -->"
    if start in text and end in text:
        pre = text.split(start)[0]
        post = text.split(end, 1)[1]
        path.write_text(pre + block + post.lstrip("\n"), encoding="utf-8")
        return
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text + "\n" + block, encoding="utf-8")


def _claims() -> None:
    path = PAPER / "CLAIM_REGISTRY.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    claims = [row for row in payload.get("claims", []) if not str(row.get("claim_id", "")).startswith("C_NOVEL_")]
    for algo in ALGOS:
        metric_path = ROOT / "results" / "novel" / algo / "seed42" / "test" / "test_metrics.json"
        if not metric_path.is_file():
            claims.append({
                "claim_id": f"C_NOVEL_{algo.upper()}",
                "claim": f"{algo.upper()} test accuracy",
                "dataset": "JaalTaka",
                "split": "test",
                "sample_definition": "physical_note",
                "training_seed": "42",
                "evaluation_seed": "42",
                "metric": "accuracy",
                "value": "NOT_MEASURED",
                "source_artifact": str(metric_path),
                "script": "scripts/eval/eval_novel.py",
                "status": "NOT_MEASURED",
            })
            continue
        rows = json.loads(metric_path.read_text(encoding="utf-8"))["views"]
        for row in rows:
            claims.append({
                "claim_id": f"C_NOVEL_{algo.upper()}_{row['k']}VIEW",
                "claim": f"{algo.upper()} {row['k']}-view test accuracy",
                "dataset": "JaalTaka",
                "split": "test",
                "sample_definition": "physical_note",
                "training_seed": "42",
                "evaluation_seed": "42",
                "metric": "accuracy",
                "value": row["accuracy"],
                "source_artifact": str(metric_path),
                "script": "scripts/eval/eval_novel.py",
                "status": "VERIFIED",
            })
    payload["claims"] = claims
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    subprocess.run([PY, str(ROOT / "scripts" / "eval" / "write_novel_comparison.py")], cwd=ROOT, check=False)
    for algo in ALGOS:
        subprocess.run([PY, str(ROOT / "scripts" / "eval" / "write_algo_results.py"), "--algo", algo], cwd=ROOT, check=False)
    (PAPER / "NOVELTY_DECLARATION.md").write_text(NOVELTY, encoding="utf-8")
    _splice(PAPER / "FINAL_RESULTS.md", _section())
    audit = (
        "<!-- NOVEL_ALGORITHMS_START -->\n"
        "## Additional algorithms\n\n"
        "Ten extra algorithms were trained at seed 42 or marked NOT_MEASURED. "
        "See `NOVEL_ALGORITHMS_COMPARISON.md`. PRMVT stays the primary accuracy result. "
        "SFAQ security-feature labels, MTPT denomination and emotion, SFPL FedAvg and FedAvg+EWC arms, "
        "and CVS do-calculus identification are NOT_MEASURED.\n"
        "<!-- NOVEL_ALGORITHMS_END -->\n"
    )
    _splice(PAPER / "FINAL_AUDIT.md", audit)
    _claims()
    print("paper files updated")


if __name__ == "__main__":
    main()
