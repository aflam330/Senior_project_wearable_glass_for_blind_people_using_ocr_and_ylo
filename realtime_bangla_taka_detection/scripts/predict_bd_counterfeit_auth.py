"""Predict REAL vs COUNTERFEIT with the BD_Counterfeit-trained model.

Usage:
  python scripts/predict_bd_counterfeit_auth.py path/to/note.jpg
  python scripts/predict_bd_counterfeit_auth.py path/to/folder --min-conf 0.90

CLI prints a strong REAL / COUNTERFEIT call when confidence is high.
Glass-safe wording is also printed (never says counterfeit to the wearer).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboeye.bd_counterfeit_auth import BDCounterfeitAuth
from roboeye.config import BD_COUNTERFEIT_AUTH_WEIGHTS


def format_line(path: Path, r: dict) -> str:
    if r["verdict"] == "CHECK_BY_HAND":
        tag = "UNCERTAIN -> CHECK_BY_HAND"
    elif r.get("strong"):
        tag = f"STRONG {r['verdict']}"
    else:
        tag = r["verdict"]
    return (
        f"{path}: {tag}  "
        f"(conf={r['confidence']:.3f}  "
        f"p_fake={r['counterfeit_prob']:.3f} p_real={r['genuine_prob']:.3f})  "
        f"glass='{r['glass_speak']}'"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--min-conf", type=float, default=None)
    ap.add_argument("--weights", type=Path, default=BD_COUNTERFEIT_AUTH_WEIGHTS)
    args = ap.parse_args()
    if not args.weights.exists():
        raise SystemExit(f"Missing weights: {args.weights}. Train first.")

    clf = BDCounterfeitAuth(weights=args.weights, min_conf=args.min_conf)
    if not clf.loaded:
        raise SystemExit(f"Failed to load weights: {args.weights}")

    paths: list[Path] = []
    if args.input.is_file():
        paths = [args.input]
    else:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(args.input.rglob(ext))
    if not paths:
        raise SystemExit(f"No images under {args.input}")

    for p in sorted(paths):
        print(format_line(p, clf.predict_path(p)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
