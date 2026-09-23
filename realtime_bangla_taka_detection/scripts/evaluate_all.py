"""Evaluate baseline and Q-DUIG: 1-6 views, 9 policies, calibration, robustness, oracle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from roboeye.camva.data import NoteViewDataset, collate_notes
from roboeye.camva.engine import load_baseline, make_loader, predict_baseline
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.artifacts import QDUIG_ROOT, init_run, save_json
from roboeye.qduig.calibration import HERCalibrator, evaluate_calibration_suite, fit_temperature
from roboeye.qduig.config_io import load_config
from roboeye.qduig.corruptions import apply_corruption
from roboeye.qduig.engine import (
    cost_from_dict,
    load_qduig,
    mc_dropout_probs,
    pack_and_save,
    predict_qduig,
    run_oracle,
    sequential_predict,
)
from roboeye.qduig.metrics_ext import full_binary_report, holm_bonferroni, mcnemar, wilcoxon_signed
from roboeye.qduig.policy import PolicyName, pareto_front, select_operating_point

POLICIES: list[PolicyName] = [
    "random",
    "fixed_order",
    "quality_only",
    "confidence_only",
    "diversity_only",
    "quality_diversity",
    "uncertainty_quality",
    "uncertainty_diversity",
    "full_proposed",
]


class CorruptNoteDataset(NoteViewDataset):
    def __getitem__(self, i: int):
        item = super().__getitem__(i)
        return item

    def _read(self, path, corruption, severity):
        import cv2
        from PIL import Image

        from roboeye.authenticity import default_transform

        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        if corruption:
            bgr = apply_corruption(bgr, corruption, severity)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return default_transform(train=False)(Image.fromarray(rgb))


def eval_baseline_views(ckpt: Path, splits, records, out_root: Path, seed: int) -> list[dict]:
    model = load_baseline(ckpt)
    rows = []
    for k in range(1, 7):
        out = out_root / f"baseline_{k}view"
        out.mkdir(parents=True, exist_ok=True)
        loader = make_loader(splits["test"], records, n_views=k, train=False, batch=8)
        pred = predict_baseline(model, loader, DEVICE)
        m = pack_and_save(pred, f"baseline_{k}view", out)
        m["n_views_requested"] = k
        rows.append(m)
        print(f"baseline {k}view acc={m['accuracy']:.4f}", flush=True)
    save_json(out_root / "views_1_to_6.json", rows)
    return rows


def eval_policies(model, cfg, splits, records, out_root: Path, seed: int) -> list[dict]:
    cost = cost_from_dict(cfg.get("cost", cfg))
    rows = []
    for policy in POLICIES:
        for k in range(1, 7):
            out = out_root / f"{policy}_{k}view"
            out.mkdir(parents=True, exist_ok=True)
            pred = sequential_predict(
                model, splits["test"], records, policy=policy, cost=cost, force_k=k, seed=seed
            )
            m = pack_and_save(pred, f"{policy}_{k}view", out)
            m["policy"] = policy
            m["n_views_requested"] = k
            m["selected_indices"] = pred["selected_indices"]
            rows.append(m)
            print(f"{policy} {k}view acc={m['accuracy']:.4f} avg_views={m['average_views']:.2f}", flush=True)
        # adaptive stop (full proposed only uses CRIQP stop; others consume all 6 in policy order)
        out = out_root / f"{policy}_adaptive"
        pred = sequential_predict(
            model, splits["test"], records, policy=policy, cost=cost, force_k=None, seed=seed
        )
        m = pack_and_save(pred, f"{policy}_adaptive", out)
        m["policy"] = policy
        m["n_views_requested"] = "adaptive"
        rows.append(m)
        print(f"{policy} adaptive acc={m['accuracy']:.4f} avg_views={m['average_views']:.2f}", flush=True)
    save_json(out_root / "all_policies.json", [{k: v for k, v in r.items() if k != "selected_indices"} for r in rows])
    return rows


def eval_pareto_val(model, cfg, splits, records, out: Path, seed: int) -> dict:
    """Sweep λ on validation only; pick operating point; never touch test for selection."""
    base = cfg.get("cost", {})
    points = []
    for lamb in (0.02, 0.05, 0.08, 0.12, 0.20, 0.35):
        trial = dict(base)
        trial["lambda_cost_policy"] = lamb
        cost = cost_from_dict(trial)
        pred = sequential_predict(
            model, splits["val"], records, policy="full_proposed", cost=cost, force_k=None, seed=seed
        )
        m = full_binary_report(np.array(pred["y_true"]), np.array(pred["genuine_score"]))
        points.append(
            {
                "lambda_cost_policy": lamb,
                "accuracy": m["accuracy"],
                "average_views": pred["average_views"],
                "average_cost": pred["average_cost"],
                "split": "val",
            }
        )
    front = pareto_front(points)
    floor = float(cfg.get("operating_point", {}).get("min_val_accuracy", 0.90))
    chosen = select_operating_point(front, floor)
    payload = {"points": points, "pareto": front, "chosen": chosen, "min_val_accuracy": floor}
    save_json(out / "pareto_val.json", payload)
    return payload


def eval_calibration(model, cfg, splits, records, out: Path) -> dict:
    val_loader = make_loader(splits["val"], records, n_views=6, train=False, batch=8)
    test_loader = make_loader(splits["test"], records, n_views=6, train=False, batch=8)
    val_pred = predict_qduig(model, val_loader, DEVICE)
    test_pred = predict_qduig(model, test_loader, DEVICE)
    t = fit_temperature(np.array(val_pred["logits"]), np.array(val_pred["y_true"]))
    her = HERCalibrator()
    her.fit(np.array(val_pred["logits"]), np.array(val_pred["y_true"]), np.array(val_pred["quality"]))
    mc = mc_dropout_probs(model, test_loader, DEVICE, n_samples=6)
    suite = evaluate_calibration_suite(
        np.array(test_pred["logits"]),
        np.array(test_pred["y_true"]),
        np.array(test_pred["quality"]),
        her,
        t,
        mc_probs=mc,
    )
    payload = {"temperature_val": t, "her_params": her.params, "test": suite, "fit_split": "val"}
    save_json(out / "calibration.json", payload)
    return payload


def eval_robustness(model, baseline, splits, records, out: Path, corruptions: list[dict]) -> list[dict]:
    from PIL import Image
    import cv2
    from torch.utils.data import DataLoader
    from roboeye.authenticity import default_transform

    class CDataset(NoteViewDataset):
        def __getitem__(self, i: int):
            nid = self.ids[i]
            rec = self.records[nid]
            paths = self._ordered_paths(rec)
            if self.n_views is not None:
                paths = paths[: self.n_views]
            tensors = []
            for pth in paths:
                bgr = cv2.imread(str(pth))
                if bgr is None:
                    continue
                if self.corruption:
                    bgr = apply_corruption(bgr, self.corruption, self.corruption_severity)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                tensors.append(self.tf(Image.fromarray(rgb)))
            import torch

            views = torch.stack(tensors, dim=0)
            return {
                "note_id": nid,
                "views": views,
                "n": views.size(0),
                "label": torch.tensor(int(rec["label"]), dtype=torch.long),
            }

    rows = []
    clean_loader = make_loader(splits["test"], records, n_views=6, train=False, batch=8)
    base_clean = full_binary_report(
        np.array(predict_baseline(baseline, clean_loader, DEVICE)["y_true"]),
        np.array(predict_baseline(baseline, clean_loader, DEVICE)["genuine_score"]),
    )
    q_clean = full_binary_report(
        np.array(predict_qduig(model, clean_loader, DEVICE)["y_true"]),
        np.array(predict_qduig(model, clean_loader, DEVICE)["genuine_score"]),
    )
    rows.append({"corruption": "clean", "severity": 0.0, "baseline": base_clean, "proposed": q_clean})

    for spec in corruptions:
        name = spec["name"]
        sev = float(spec["severity"])
        tag = spec.get("tag", name)
        ds = CDataset(splits["test"], records, n_views=6, train=False, corruption=name, corruption_severity=sev)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_notes)
        bp = predict_baseline(baseline, loader, DEVICE)
        qp = predict_qduig(model, loader, DEVICE)
        bm = full_binary_report(np.array(bp["y_true"]), np.array(bp["genuine_score"]))
        qm = full_binary_report(np.array(qp["y_true"]), np.array(qp["genuine_score"]))
        row = {
            "corruption": tag,
            "severity": sev,
            "baseline": bm,
            "proposed": qm,
            "baseline_degradation": bm["accuracy"] - base_clean["accuracy"],
            "proposed_degradation": qm["accuracy"] - q_clean["accuracy"],
        }
        rows.append(row)
        print(f"robust {tag} base={bm['accuracy']:.3f} qduig={qm['accuracy']:.3f}", flush=True)
    save_json(out / "robustness.json", rows)
    return rows


def eval_stats(base_pred: dict, prop_pred: dict, out: Path) -> dict:
    y = np.array(base_pred["y_true"])
    pa = (np.array(base_pred["genuine_score"]) >= 0.5).astype(int)
    pb = (np.array(prop_pred["genuine_score"]) >= 0.5).astype(int)
    mc = mcnemar(y, pa, pb)
    wil = wilcoxon_signed(np.array(base_pred["genuine_score"]), np.array(prop_pred["genuine_score"]))
    payload = {
        "predefined_comparison": "FULL PROPOSED vs CNN+ViT BASELINE",
        "mcnemar": mc,
        "wilcoxon_scores": wil,
        "holm_adjusted_p": holm_bonferroni([mc["p_value"], wil.get("p_value") or 1.0]),
    }
    save_json(out / "statistics.json", payload)
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "final_research.yaml"))
    p.add_argument("--proposed-checkpoint", default=None)
    p.add_argument("--baseline-checkpoint", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--skip-policies", action="store_true")
    p.add_argument("--skip-robustness", action="store_true")
    p.add_argument("--skip-oracle", action="store_true")
    args = p.parse_args()

    master = load_config(args.config)
    proposed_cfg = load_config(ROOT / master.get("proposed", "configs/proposed.yaml"))
    splits, records = load_splits()
    out = Path(args.output_dir) if args.output_dir else QDUIG_ROOT / "eval" / f"seed{args.seed}"
    init_run(out, master, args.seed, "evaluate_all", "Same note-disjoint split as CAMVA.")

    bckpt = Path(
        args.baseline_checkpoint
        or ROOT / f"results/camva/checkpoints/baseline_cnnvit_seed{args.seed}.pt"
    )
    pckpt = Path(
        args.proposed_checkpoint
        or ROOT / "results" / "qduig" / "proposed" / f"seed{args.seed}" / "checkpoint.pt"
    )

    if bckpt.is_file():
        eval_baseline_views(bckpt, splits, records, out / "baseline", args.seed)
        baseline = load_baseline(bckpt)
    else:
        print("BASELINE CHECKPOINT MISSING", bckpt)
        baseline = None

    if not pckpt.is_file():
        print("PROPOSED CHECKPOINT MISSING", pckpt)
        save_json(out / "status.json", {"proposed": "NOT_MEASURED", "reason": str(pckpt)})
        return

    model = load_qduig(pckpt, proposed_cfg)
    if not args.skip_policies:
        eval_policies(model, proposed_cfg, splits, records, out / "policies", args.seed)
        eval_pareto_val(model, proposed_cfg, splits, records, out, args.seed)
    eval_calibration(model, proposed_cfg, splits, records, out)
    if baseline is not None and not args.skip_robustness:
        eval_robustness(model, baseline, splits, records, out, master.get("corruptions", []))
    if not args.skip_oracle:
        oracle = run_oracle(model, splits["test"], records)
        save_json(out / "oracle.json", oracle["summary"])
        save_json(out / "oracle_per_note.json", oracle["per_note"])

    if baseline is not None:
        base_pred = json.loads((out / "baseline" / "baseline_6view" / "test_predictions.json").read_text(encoding="utf-8"))
        loader = make_loader(splits["test"], records, n_views=6, train=False, batch=8)
        prop_pred = predict_qduig(model, loader, DEVICE)
        pack_and_save(prop_pred, "proposed_6view", out / "proposed_6view")
        eval_stats(base_pred, prop_pred, out)

    save_json(
        out / "generalization.json",
        {
            "camera_disjoint": "NOT_MEASURED",
            "session_disjoint": "NOT_MEASURED",
            "reason": "JaalTaka manifest camera_id=unknown session_id=unknown",
        },
    )
    print("evaluate_all done", out)


if __name__ == "__main__":
    main()
