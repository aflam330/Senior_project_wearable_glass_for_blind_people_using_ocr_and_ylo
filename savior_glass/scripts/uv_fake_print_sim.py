"""UV check on REAL counterfeit print under SIMULATED UV lamp (no real UV fake photos needed).

Why this is stronger than generic blank sheets:
  - Uses actual counterfeit note photographs from BD_Counterfeit / fake_bd_cur (normal light).
  - Counterfeit notes lack security fibres; under a simulated UV lamp they should look fibre-free.
  - Lamp colour/noise model is measured from genuine uv_notes photographs (same as uv_stress_sim.py).

Honest limitation (must stay in the paper):
  This is still a simulation of the lamp — not a photograph of a counterfeit under a real UV lamp.

Usage:
  python scripts/uv_fake_print_sim.py
Output:
  results/uv_check/uv_fake_print_sim.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from modes.uv_check import fibre_mask, inner  # noqa: E402
from scripts.uv_dataset import DATA, OUT, _surface_patch, read  # noqa: E402
from scripts.uv_stress_sim import measure_real, sheet  # noqa: E402

FAKE_ROOT = ROOT.parent / "data set" / "BD_Counterfeit"
FAKE_FALLBACK = ROOT.parent / "data set" / "fake_bd_cur"
PER_DENOM = 40


def collect_fakes() -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for denom in ("200", "500", "1000"):
        # prefer unaugmented normals from class folders
        cands = sorted((FAKE_ROOT / denom).glob("F*_normal.jpg"))
        if not cands:
            cands = sorted((FAKE_FALLBACK / denom).glob("*.jpg")) + sorted((FAKE_FALLBACK / denom).glob("*.png"))
        if not cands:
            cands = sorted((FAKE_ROOT / "images" / "test").glob(f"F{denom}_*_normal.jpg"))
        for p in cands[:PER_DENOM]:
            paths.append((denom, p))
    return paths


def to_grey_print(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # mild contrast stretch so print survives UV recolour
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    g = np.clip((g.astype(np.float32) - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)
    return g


def main() -> int:
    real = measure_real()
    fakes = collect_fakes()
    if not fakes:
        raise SystemExit("No counterfeit print images found under BD_Counterfeit / fake_bd_cur")

    # genuine UV photos: how often fibres are found (sanity)
    gen_rows = [r for r in csv.DictReader(open(DATA / "boxes.csv", encoding="utf-8")) if r.get("x1")]
    gen_hit = gen_total = 0
    for r in gen_rows:
        img = read(DATA / r["value"] / r["file"])
        x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
        note = inner(img[y1:y2, x1:x2])
        _, n, _ = fibre_mask(note)
        gen_total += 1
        gen_hit += int(n >= int(getattr(config, "UV_MIN_FIBRES", 6)))

    # Paper-facing threshold (config default may be 1 for glass sensitivity).
    # Report both: deploy threshold and paper threshold with colour gate.
    thresholds = [
        {"tag": "deploy_default", "min_fibres": int(getattr(config, "UV_MIN_FIBRES", 1)),
         "colour": float(getattr(config, "UV_COLOUR_THRESHOLD", 0.0)), "lum": float(getattr(config, "UV_LUM_THRESHOLD", 0.0))},
        {"tag": "paper_strict", "min_fibres": 6, "colour": 0.22, "lum": 0.0},
    ]
    conditions = [
        {"name": "base", "d": 0.0, "sigma_mult": 1.0, "jpeg": 85},
        {"name": "glow", "d": 0.05, "sigma_mult": 1.0, "jpeg": 85},
        {"name": "noisy", "d": 0.05, "sigma_mult": 2.0, "jpeg": 60},
        {"name": "harsh_jpeg", "d": 0.08, "sigma_mult": 1.5, "jpeg": 35},
    ]
    rng = np.random.default_rng(42)
    sigma0 = float(real["noise_sigma_at_check_scale"])
    by_thr = {}
    for thr in thresholds:
        by_cond = {}
        for cond in conditions:
            passed = 0
            fibre_counts = []
            for denom, path in fakes:
                bgr = cv2.imread(str(path))
                if bgr is None:
                    continue
                grey = to_grey_print(bgr)
                grey = cv2.resize(
                    grey,
                    (600, max(200, int(600 * grey.shape[0] / max(grey.shape[1], 1)))),
                    interpolation=cv2.INTER_AREA,
                )
                sim = sheet(grey, real, cond["d"], sigma0 * cond["sigma_mult"], cond["jpeg"], rng)
                note = inner(sim)
                _, n, _ = fibre_mask(
                    note,
                    colour_threshold=thr["colour"],
                    lum_threshold=thr["lum"],
                )
                fibre_counts.append(int(n))
                passed += int(n >= thr["min_fibres"])
            n = len(fibre_counts)
            by_cond[cond["name"]] = {
                "n": n,
                "passed_as_genuine": passed,
                "reject_rate": round(1.0 - passed / max(n, 1), 4),
                "pass_rate": round(passed / max(n, 1), 4),
                "mean_fibres": round(float(np.mean(fibre_counts)) if fibre_counts else 0.0, 2),
                "median_fibres": int(np.median(fibre_counts)) if fibre_counts else 0,
                **cond,
            }
        by_thr[thr["tag"]] = {"threshold": thr, "conditions": by_cond}

    strict = by_thr["paper_strict"]["conditions"]
    by_cond = strict
    report = {
        "lamp_model_from_genuine_uv": real,
        "genuine_uv_accept_rate": round(gen_hit / max(gen_total, 1), 4),
        "genuine_n": gen_total,
        "fake_print_source": str(FAKE_ROOT if (FAKE_ROOT / "200").exists() else FAKE_FALLBACK),
        "fake_print_n": len(fakes),
        "thresholds": by_thr,
        "uv_min_fibres_paper_strict": 6,
        "conditions": by_cond,  # paper_strict for backward-compatible readers
        "caveat": (
            "Simulated UV lighting on real counterfeit print photographs. "
            "Not a substitute for real counterfeit-under-UV captures."
        ),
        "paper_claim": (
            f"On {len(fakes)} real counterfeit prints under simulated UV "
            f"(paper_strict min_fibres=6, colour=0.22), "
            f"reject_rate(base)={strict['base']['reject_rate']:.1%}; "
            f"genuine UV accept_rate={gen_hit / max(gen_total, 1):.1%} (n={gen_total})."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "uv_fake_print_sim.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
