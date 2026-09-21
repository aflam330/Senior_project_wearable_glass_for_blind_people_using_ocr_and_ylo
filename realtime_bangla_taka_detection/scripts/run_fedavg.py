"""Run a short FedAvg simulation on JaalTaka authenticity shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roboeye.fedavg import run_fedavg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=int, default=3)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--max-notes", type=int, default=180)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.clients, args.rounds, args.max_notes = 3, 8, 90
    result = run_fedavg(
        n_clients=args.clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        max_notes=args.max_notes,
        alpha=args.alpha,
    )
    print(result)


if __name__ == "__main__":
    main()
