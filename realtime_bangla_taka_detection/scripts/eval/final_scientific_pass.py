"""Re-read saved artifacts and write a dated audit JSON. Does not train."""
from __future__ import annotations

import json
import math
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
OUT = WORK / "paper_evidence" / "final_pass_20260929.json"
SKIP = {"venv", ".venv", "site-packages", "__pycache__", "cache", "Unused"}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> dict:
    phat = k / n
    den = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / den
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return {"k": k, "n": n, "phat": phat, "low": centre - half, "high": centre + half}


def load(rel: str):
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")), str(path)


def acc_to_k(acc: float, n: int = 208) -> int:
    k = int(round(acc * n))
    if abs(k / n - acc) > 1e-9:
        raise SystemExit(f"accuracy {acc} is not k/{n}")
    return k


def main() -> None:
    py_errors = []
    py_ok = 0
    for path in ROOT.rglob("*.py"):
        if any(part in SKIP for part in path.parts):
            continue
        if "Thesis Report and paper" in str(path):
            continue
        try:
            py_compile.compile(str(path), doraise=True)
            py_ok += 1
        except py_compile.PyCompileError as exc:
            py_errors.append(str(exc))

    foreign_hits = []
    needles = ("USD", "EUR", "INR", "CNY", "JPY", "GBP", "dollar", "euro", "rupee")
    for path in ROOT.rglob("*.py"):
        if any(part in SKIP for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        low = text.lower()
        for needle in needles:
            if needle.lower() in low and "do not" not in low and "not downloaded" not in low:
                foreign_hits.append(f"{path.name}:{needle}")

    split, split_path = load("results/camva/splits/split_metadata.json")
    oracle, oracle_path = load("results/qduig/eval/seed42/oracle.json")
    occ, occ_path = load("results/robustness/occlusion_decision.json")
    bbox, bbox_path = load("results/bbox/bbox_eval.json")
    multi, multi_path = load("results/bbox/bbox_multinote.json")
    vcie, vcie_path = load("results/novel_v2/vcie_k1/seed42/test/test_metrics.json")
    mtpt, mtpt_path = load("results/novel_v2/mtpt_prefix_ft/seed42/test/test_metrics.json")
    pac, pac_path = load("results/theory/pacbayes_d1_bound_seed42.json")
    prefix, prefix_path = load("results/qduig/prefix_ft/seed42/test_views_20260928/views_1_to_6.json")  # after the NaN fix

    n = int(oracle["n"])
    oracle_k = acc_to_k(oracle["oracle_accuracy"], n)
    learned_k = acc_to_k(oracle["learned_accuracy"], n)
    gap_k = oracle_k - learned_k
    if abs(gap_k / n - oracle["accuracy_gap_oracle_minus_learned"]) > 1e-9:
        raise SystemExit("oracle gap does not match counts")
    if oracle["notes_unsolvable_by_any_subset"] != n - oracle_k:
        raise SystemExit("unsolvable count does not match oracle accuracy")

    def view_acc(blob, k):
        if "new" in blob:  # re-evaluation files: {"new": {k: acc}, "stored": {k: acc}}
            return float(blob["new"][str(k)])
        rows = blob.get("views") or blob.get("proposed")
        if rows is None:
            raise SystemExit("no view list: " + ",".join(blob.keys()))
        for row in rows:
            if int(row["k"]) == k:
                return float(row["accuracy"])
        raise SystemExit(f"missing view {k}")

    counted = {
        "prefix_1": wilson(acc_to_k(view_acc(prefix, 1), n), n),
        "prefix_6": wilson(acc_to_k(view_acc(prefix, 6), n), n),
        "vcie_1": wilson(acc_to_k(view_acc(vcie, 1), n), n),
        "vcie_6": wilson(acc_to_k(view_acc(vcie, 6), n), n),
        "mtpt_1": wilson(acc_to_k(view_acc(mtpt, 1), n), n),
        "mtpt_6": wilson(acc_to_k(view_acc(mtpt, 6), n), n),
        "ensemble_occ": wilson(acc_to_k(occ["ensemble"]["test"]["occ55"], n), n),
        "ensemble_clean": wilson(acc_to_k(occ["ensemble"]["test"]["clean"], n), n),
    }

    report = {
        "py_compiled_ok": py_ok,
        "py_errors": py_errors,
        "foreign_currency_code_hits": foreign_hits[:40],
        "foreign_currency_code_hit_count": len(foreign_hits),
        "split": {
            "path": split_path,
            "leakage": split["leakage_check"],
            "n_notes": split["n_notes_total"],
            "train_val_test": [
                split["train"]["n_notes"],
                split["val"]["n_notes"],
                split["test"]["n_notes"],
            ],
        },
        "oracle": {
            "path": oracle_path,
            "oracle_correct": oracle_k,
            "learned_correct": learned_k,
            "gap_notes": gap_k,
            "unsolvable": oracle["notes_unsolvable_by_any_subset"],
            "wilson_oracle": wilson(oracle_k, n),
            "wilson_learned": wilson(learned_k, n),
        },
        "occlusion_keys": sorted(occ.keys()) if isinstance(occ, dict) else "not_dict",
        "bbox_keys": sorted(bbox.keys()) if isinstance(bbox, dict) else "not_dict",
        "multinote_keys": sorted(multi.keys()) if isinstance(multi, dict) else "not_dict",
        "vcie_keys": sorted(vcie.keys()) if isinstance(vcie, dict) else "not_dict",
        "mtpt_keys": sorted(mtpt.keys()) if isinstance(mtpt, dict) else "not_dict",
        "pac_keys": sorted(pac.keys()) if isinstance(pac, dict) else "not_dict",
        "prefix_type": type(prefix).__name__,
        "paths": {
            "occ": occ_path,
            "bbox": bbox_path,
            "multi": multi_path,
            "vcie": vcie_path,
            "mtpt": mtpt_path,
            "pac": pac_path,
            "prefix": prefix_path,
        },
        "wilson": counted,
        "bbox_synthetic": bbox["synthetic_best.pt"]["with_note"],
        "bbox_false_alarms": bbox["synthetic_best.pt"]["no_note_false_positive_rate"],
        "ensemble_test": occ["ensemble"]["test"],
        "rejection_test_occ": occ["rejection"]["test_occ"],
        "pacbayes_d1_mcallester": pac["mcallester"],
        "pacbayes_d1_kl_upper": pac["pacbayes_kl_upper"],
        "test_used_for_pacbayes": pac["test_used"],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("WROTE", OUT)
    print("py_ok", py_ok, "py_errors", len(py_errors))
    print("foreign_hits", len(foreign_hits))
    print("oracle", oracle_k, learned_k, gap_k)


if __name__ == "__main__":
    main()
