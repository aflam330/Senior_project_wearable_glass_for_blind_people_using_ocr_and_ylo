#!/bin/sh
# Run on the Raspberry Pi 5. Does not produce numbers on a laptop.
set -eu
ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  python -m pip install -r requirements.txt
fi
echo "Copy INT8 ONNX and the PRMVT checkpoint onto this Pi before the benchmark."
echo "Then: python scripts/benchmark_pi5.py --iters 100 --sustained 30"
