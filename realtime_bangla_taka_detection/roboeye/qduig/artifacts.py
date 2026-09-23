"""Experiment artifact helpers. Every run writes the required files."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..camva.notes import CAMVA_ROOT
from ..config import ROOT

QDUIG_ROOT = ROOT / "results" / "qduig"


def run_dir(experiment: str, seed: int, output_dir: Path | None = None) -> Path:
    base = Path(output_dir) if output_dir else QDUIG_ROOT
    d = base / experiment / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def write_environment(out: Path) -> None:
    try:
        import torch

        torch_s = torch.__version__
        cuda = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        torch_s, cuda, gpu = "unknown", None, "unknown"
    lines = [
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
        f"torch={torch_s}",
        f"cuda={cuda}",
        f"gpu={gpu}",
        f"cwd={os.getcwd()}",
        f"utc={datetime.now(timezone.utc).isoformat()}",
    ]
    (out / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_git_commit(out: Path) -> None:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        sha = "NOT_MEASURED"
    (out / "git_commit.txt").write_text(sha + "\n", encoding="utf-8")


def write_seed(out: Path, seed: int) -> None:
    (out / "seed.txt").write_text(str(seed) + "\n", encoding="utf-8")


def write_run_readme(out: Path, title: str, notes: str) -> None:
    (out / "README.md").write_text(f"# {title}\n\n{notes}\n", encoding="utf-8")


def init_run(out: Path, config: dict, seed: int, title: str, notes: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.yaml.json", config)
    # also dump a yaml-like config.yaml
    lines = [f"{k}: {json.dumps(v)}" for k, v in config.items()]
    (out / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_environment(out)
    write_git_commit(out)
    write_seed(out, seed)
    write_run_readme(out, title, notes)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
