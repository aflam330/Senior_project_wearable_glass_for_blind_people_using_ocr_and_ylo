"""Latency breakdown on the *current* host. Pi 5 is NOT_MEASURED unless this host is a Pi 5."""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from roboeye.camva.engine import load_baseline, make_loader
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.artifacts import init_run, save_json
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig


def is_pi5() -> bool:
    try:
        text = Path("/proc/device-tree/model").read_text(errors="ignore")
        return "Raspberry Pi 5" in text
    except Exception:
        return False


def bench(fn, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "n": int(len(arr)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "edge.yaml"))
    p.add_argument("--proposed-checkpoint", required=True)
    p.add_argument("--baseline-checkpoint", default=None)
    p.add_argument("--proposed-config", default=str(ROOT / "configs" / "proposed.yaml"))
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    edge = load_config(args.config)
    pcfg = load_config(args.proposed_config)
    splits, records = load_splits()
    out = Path(args.output_dir) if args.output_dir else ROOT / "results" / "qduig" / "edge"
    init_run(out, edge, 42, "edge benchmark", "Device is recorded honestly.")

    model = load_qduig(Path(args.proposed_checkpoint), pcfg)
    loader = make_loader(splits["test"][:16], records, n_views=6, train=False, batch=1)
    batch = next(iter(loader))
    views, mask = batch["views"].to(DEVICE), batch["mask"].to(DEVICE)

    warmup, repeats = int(edge.get("warmup", 10)), int(edge.get("repeats", 30))

    def full():
        model(views, mask)

    def encode_only():
        model.encode_views(views)

    breakdown = {
        "end_to_end": bench(full, warmup, repeats),
        "encode": bench(encode_only, warmup, repeats),
    }
    per_k = {}
    for k in edge.get("views", [1, 2, 3, 4, 5, 6]):
        vk = views[:, :k]
        mk = mask[:, :k]

        def run(vk=vk, mk=mk):
            model(vk, mk)

        per_k[str(k)] = bench(run, warmup, repeats)

    ram = None
    try:
        import psutil

        ram = psutil.virtual_memory()._asdict()
    except Exception:
        ram = "NOT_MEASURED"

    temp = "NOT_MEASURED"
    throttle = "NOT_MEASURED"
    try:
        if Path("/sys/class/thermal/thermal_zone0/temp").is_file():
            temp = float(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except Exception:
        pass

    ckpt = Path(args.proposed_checkpoint)
    payload = {
        "host": platform.platform(),
        "device": str(DEVICE),
        "is_raspberry_pi_5": is_pi5(),
        "pi5_metrics": "MEASURED" if is_pi5() else "NOT_MEASURED",
        "energy": "NOT_MEASURED",
        "startup_s": "NOT_MEASURED",
        "model_size_bytes": ckpt.stat().st_size if ckpt.is_file() else None,
        "breakdown_ms": breakdown,
        "views_1_to_6": per_k,
        "ram": ram,
        "temperature_c": temp,
        "throttling": throttle,
        "sustained": "NOT_MEASURED",
        "cpu": "NOT_MEASURED",
        "fps_end_to_end": 1000.0 / breakdown["end_to_end"]["median_ms"] if breakdown["end_to_end"]["median_ms"] else None,
    }
    save_json(out / "edge.json", payload)
    print(payload["device"], payload["pi5_metrics"], payload["breakdown_ms"])


if __name__ == "__main__":
    main()
