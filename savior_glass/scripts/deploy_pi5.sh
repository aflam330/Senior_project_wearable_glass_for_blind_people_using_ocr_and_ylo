#!/bin/sh
# Prepare this checkout on a Raspberry Pi 5. Does not invent latency numbers.
set -eu
ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/pi5_preflight.py
echo "On this Pi, currency mode uses best_int8.onnx and the watermark check is on."
echo "Benchmark: python scripts/benchmark_pi5.py --iters 100 --sustained 30"
echo "App: python main.py"
