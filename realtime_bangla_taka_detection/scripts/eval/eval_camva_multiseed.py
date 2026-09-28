"""Evaluate the CNN+ViT baseline and CAMVA over several training seeds, 1-6 views.

Each seed's baseline is compared with the same seed's CAMVA on the fixed note-disjoint
test split. Per-seed results go to results/camva/multiseed/seed<N>.json and the
aggregate (mean, SD, per-seed CAMVA - baseline difference) to
results/camva/multiseed/summary.json. The single-seed files in results/camva/metrics
are not touched.

Usage:
  python scripts/eval/eval_camva_multiseed.py                 # seeds 42 43 44
  python scripts/eval/eval_camva_multiseed.py --seeds 42 43
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.camva.engine import (
    CKPT_DIR, load_baseline, load_camva, make_loader, pack_eval,
    predict_baseline, predict_camva, save_json,
)
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE

OUT = ROOT / "results" / "camva" / "multiseed"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--fusion", default="quality_attention")
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    splits, records = load_splits()
    per_seed = {}
    for seed in args.seeds:
        bpath = CKPT_DIR / f"baseline_cnnvit_seed{seed}.pt"
        cpath = CKPT_DIR / f"camva_{args.fusion}_seed{seed}.pt"
        missing = [x.name for x in (bpath, cpath) if not x.is_file()]
        if missing:
            print(f"seed {seed}: missing {missing}, skipped", flush=True)
            continue
        base, camva = load_baseline(bpath), load_camva(cpath, args.fusion)
        rows = []
        for k in range(1, 7):
            loader = make_loader(splits["test"], records, n_views=k, train=False, batch=args.batch)
            bm, _ = pack_eval(predict_baseline(base, loader, DEVICE), f"baseline_s{seed}_{k}view")
            loader = make_loader(splits["test"], records, n_views=k, train=False, batch=args.batch)
            cm, _ = pack_eval(predict_camva(camva, loader, DEVICE), f"camva_s{seed}_{k}view")
            rows.append({"n_views": k, "baseline_accuracy": bm["accuracy"], "camva_accuracy": cm["accuracy"],
                         "baseline_f1": bm["f1"], "camva_f1": cm["f1"],
                         "difference": cm["accuracy"] - bm["accuracy"]})
            print(f"seed {seed} {k}view baseline={bm['accuracy']:.4f} camva={cm['accuracy']:.4f}", flush=True)
        save_json(OUT / f"seed{seed}.json", {"seed": seed, "baseline_ckpt": bpath.name,
                                               "camva_ckpt": cpath.name, "views": rows})
        per_seed[seed] = rows

    if not per_seed:
        raise SystemExit("no seed had both checkpoints")
    seeds = sorted(per_seed)
    summary = []
    for k in range(1, 7):
        b = [per_seed[s][k - 1]["baseline_accuracy"] for s in seeds]
        c = [per_seed[s][k - 1]["camva_accuracy"] for s in seeds]
        d = [ci - bi for bi, ci in zip(b, c)]
        sd = (lambda xs: statistics.stdev(xs) if len(xs) > 1 else None)
        summary.append({
            "n_views": k,
            "baseline_mean": statistics.fmean(b), "baseline_sd": sd(b),
            "camva_mean": statistics.fmean(c), "camva_sd": sd(c),
            "difference_mean": statistics.fmean(d), "difference_sd": sd(d),
            "differences_per_seed": dict(zip(map(str, seeds), d)),
            "camva_better_in_seeds": sum(x > 0 for x in d), "n_seeds": len(seeds),
        })
    save_json(OUT / "summary.json", {"seeds": seeds, "split": "results/camva/splits (seed 42, note-disjoint)",
                                     "test_notes": len(splits["test"]), "views": summary})
    for r in summary:
        print(f"{r['n_views']}view baseline {r['baseline_mean']:.4f} +/- {r['baseline_sd'] or 0:.4f} "
              f"camva {r['camva_mean']:.4f} +/- {r['camva_sd'] or 0:.4f} "
              f"diff {r['difference_mean']:+.4f} (CAMVA better in {r['camva_better_in_seeds']}/{r['n_seeds']} seeds)")


if __name__ == "__main__":
    main()
