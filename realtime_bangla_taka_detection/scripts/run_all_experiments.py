"""Train missing seeds / ablations and run evaluations. Resume-friendly."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "final_research.yaml"))
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-ablations", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if not args.skip_train:
        for seed in seeds:
            ckpt = ROOT / "results" / "qduig" / "proposed" / f"seed{seed}" / "checkpoint.pt"
            if ckpt.is_file():
                print("skip existing", ckpt)
                continue
            rc = run(
                [
                    PY,
                    str(ROOT / "scripts" / "train_qduig.py"),
                    "--config",
                    str(ROOT / "configs" / "proposed.yaml"),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(ckpt.parent),
                ]
            )
            if rc != 0:
                print("TRAIN FAIL seed", seed, "rc", rc)
                sys.exit(rc)

    if not args.skip_ablations:
        for yml in (ROOT / "configs").glob("ablation_*.yaml"):
            if yml.name == "ablation_baseline.yaml":
                continue
            out = ROOT / "results" / "qduig" / "ablations" / yml.stem.replace("ablation_", "") / "seed42"
            if (out / "checkpoint.pt").is_file():
                print("skip ablation", yml.name)
                continue
            rc = run(
                [
                    PY,
                    str(ROOT / "scripts" / "train_qduig.py"),
                    "--config",
                    str(yml),
                    "--seed",
                    "42",
                    "--output-dir",
                    str(out),
                ]
            )
            if rc != 0:
                print("ABLATION TRAIN FAIL", yml, rc)

    if not args.skip_eval:
        for seed in seeds:
            pckpt = ROOT / "results" / "qduig" / "proposed" / f"seed{seed}" / "checkpoint.pt"
            if not pckpt.is_file():
                print("skip eval missing", pckpt)
                continue
            run(
                [
                    PY,
                    str(ROOT / "scripts" / "evaluate_all.py"),
                    "--config",
                    args.config,
                    "--seed",
                    str(seed),
                    "--proposed-checkpoint",
                    str(pckpt),
                ]
            )

        p42 = ROOT / "results" / "qduig" / "proposed" / "seed42" / "checkpoint.pt"
        if p42.is_file():
            run([PY, str(ROOT / "scripts" / "run_continual_qduig.py"), "--checkpoint", str(p42), "--seed", "42"])
            run([PY, str(ROOT / "scripts" / "run_federated_qduig.py"), "--checkpoint", str(p42), "--seed", "42"])
            b42 = ROOT / "results" / "camva" / "checkpoints" / "baseline_cnnvit_seed42.pt"
            cmd = [PY, str(ROOT / "scripts" / "benchmark_edge_qduig.py"), "--proposed-checkpoint", str(p42)]
            if b42.is_file():
                cmd += ["--baseline-checkpoint", str(b42)]
            run(cmd)
        run([PY, str(ROOT / "scripts" / "eval_feedback_qduig.py")])
        run([PY, str(ROOT / "scripts" / "aggregate_seeds.py")])


if __name__ == "__main__":
    main()
