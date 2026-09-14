"""Run EWC + replay continual learning on two JaalTaka tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roboeye.ewc import run_ewc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-notes", type=int, default=160)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.epochs, args.max_notes = 1, 60
    print(run_ewc(epochs_per_task=args.epochs, max_notes=args.max_notes))


if __name__ == "__main__":
    main()
