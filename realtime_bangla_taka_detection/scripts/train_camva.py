"""Train CNN+ViT baseline and CAMVA on the same note-disjoint split.

Does not overwrite models/authenticity_cnn_vit.pt.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from roboeye.authenticity import MultiViewCNNVIT
from roboeye.camva.engine import CFG_DIR, CKPT_DIR, save_json, train_one
from roboeye.camva.model import CAMVANet
from roboeye.camva.notes import SEED, build_and_save_splits, load_splits
from roboeye.config import DEVICE


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--views", type=int, default=6)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--seeds", type=int, default=1, help="independent training seeds (1 or 3)")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-camva", action="store_true")
    p.add_argument("--fusion", default="quality_attention")
    args = p.parse_args()

    if not (ROOT / "results" / "camva" / "splits" / "train_note_ids.json").is_file():
        build_and_save_splits(seed=args.seed)
    splits, records = load_splits()
    split_meta = json.loads((ROOT / "results" / "camva" / "splits" / "split_metadata.json").read_text(encoding="utf-8"))
    seeds = list(range(args.seed, args.seed + args.seeds))
    # The default run (seed 42 only) keeps the original file names; other seed ranges get their
    # own files so a later run cannot overwrite the record of an earlier one.
    tag = "" if seeds == [SEED] else f"_seeds{seeds[0]}-{seeds[-1]}"

    cfg = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(DEVICE),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "views": args.views,
        "split_seed": split_meta["seed"],
        "training_seeds": seeds,
        "trained": [k for k, skip in (("baseline", args.skip_baseline), ("camva", args.skip_camva)) if not skip],
        "n_train_notes": len(splits["train"]),
        "n_val_notes": len(splits["val"]),
        "n_test_notes": len(splits["test"]),
        "independent_unit": "physical_note_id",
    }
    save_json(CFG_DIR / f"train_config{tag}.json", cfg)

    summaries = []
    for s in seeds:
        torch.manual_seed(s)
        if not args.skip_baseline:
            print("=== train baseline CNN+ViT seed", s, "===")
            base = MultiViewCNNVIT(freeze_cnn=True)
            summaries.append(
                train_one(
                    base, splits["train"], splits["val"], records,
                    epochs=args.epochs, batch=args.batch, lr=args.lr, n_views=args.views,
                    ckpt=CKPT_DIR / f"baseline_cnnvit_seed{s}.pt",
                    kind="baseline", seed=s,
                )
            )
        if not args.skip_camva:
            print("=== train CAMVA", args.fusion, "seed", s, "===")
            camva = CAMVANet(freeze_cnn=True, fusion=args.fusion)
            summaries.append(
                train_one(
                    camva, splits["train"], splits["val"], records,
                    epochs=args.epochs, batch=args.batch, lr=args.lr, n_views=args.views,
                    ckpt=CKPT_DIR / f"camva_{args.fusion}_seed{s}.pt",
                    kind="camva", seed=s, fusion=args.fusion,
                )
            )
    save_json(CFG_DIR / f"train_summary{tag}.json", summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
