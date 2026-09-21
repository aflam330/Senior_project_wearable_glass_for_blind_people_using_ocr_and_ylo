"""Create leakage-free JaalTaka note-ID splits for CAMVA (does not touch old auth files)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboeye.camva.notes import SEED, build_and_save_splits  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    build_and_save_splits(seed=args.seed)


if __name__ == "__main__":
    main()
