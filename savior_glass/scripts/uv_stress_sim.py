"""UV fibre check: a SIMULATED stress test with fibre-free printed sheets. This is not a test on counterfeit notes.

Question: if a sheet carries a note's printed design but no security fibres, can camera noise, JPEG compression or
glowing paper alone make the check count UV_MIN_FIBRES "fibres"?

How a fibre-free sheet is simulated (every assumption is stated; none is measured on a real counterfeit):
  * Print: normal-light photographs of genuine 200, 500 and 1,000 Taka notes (40 per value, from the BanglaTaka raw
    folder). Fibres are invisible in normal light, so this is real print without fibres.
  * Under the UV lamp the sheet reflects the lamp's blue light: B = level x grey, G = a x B, R = b x B, with a and b
    the medians measured on the real UV photographs and the level matched to their brightness.
  * Paper glow (optical brightener): where the sheet is not printed, extra green of strength d x B is added. d = 0 is
    dull paper; the plain surface in the real UV photographs has a green/blue ratio about 0.05 above the notes'.
  * Camera: Gaussian noise per channel (the level measured on the real photographs' plain surface, and multiples
    of it), then JPEG.
The sheet goes through the same code as a real note (modes/uv_check.fibre_mask, UV_MIN_FIBRES).

    python scripts/uv_stress_sim.py
Output: results/uv_check/uv_stress_sim.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config  # noqa: E402
from modes.uv_check import fibre_mask, inner, paper_ratios  # noqa: E402
from scripts.uv_dataset import DATA, OUT, _surface_patch, read  # noqa: E402

RAW = HERE.parents[1] / "data set" / "Bangladeshi_Paper_Currency_Raw" / "Bangladeshi_Paper_Currency_Raw"
PER_VALUE = 40


def measure_real():
    """Lamp ratios, brightness and sensor noise from the real UV photographs."""
    rows = [r for r in csv.DictReader(open(DATA / "boxes.csv", encoding="utf-8")) if r["x1"]]
    a, b, lev, noise = [], [], [], []
    for r in rows:
        img = read(DATA / r["value"] / r["file"])
        x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
        note = inner(img[y1:y2, x1:x2])
        ab = paper_ratios(note)
        if ab is None:
            continue
        a.append(ab[0]); b.append(ab[1]); lev.append(float(np.median(note[..., 0])))
        p = _surface_patch(img, x1, y1, x2, y2)
        if p is not None:
            s = 600.0 / (x2 - x1)                               # the scale at which the check looks at the note
            q = cv2.resize(p, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
            hp = q - cv2.GaussianBlur(q, (0, 0), 3)
            noise.append(float(np.median(np.abs(hp - np.median(hp))) * 1.4826))
    return {"a": float(np.median(a)), "b": float(np.median(b)), "blue_level": float(np.median(lev)),
            "noise_sigma_at_check_scale": float(np.median(noise)), "photographs": len(a)}


def sheet(grey, real, d, sigma, jpeg, rng):
    g = grey.astype(np.float32)
    g = g / max(np.percentile(g, 95), 1.0)
    B = np.clip(g * real["blue_level"] * 1.25, 0, 255)
    unprinted = np.clip((g - 0.6) / 0.3, 0, 1)                    # 1 on bare paper, 0 on dense print
    G = real["a"] * B + d * unprinted * B
    R = real["b"] * B + 0.3 * d * unprinted * B
    img = np.stack([B, G, R], -1)
    if sigma:
        img = img + rng.normal(0, sigma, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if jpeg:
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def main():
    real = measure_real()
    thr = int(config.UV_MIN_FIBRES)
    rng = np.random.default_rng(0)
    notes = []
    for v in ("200", "500", "1000"):
        files = sorted(p for p in (RAW / v).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:PER_VALUE]
        for f in files:
            g = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            if g.shape[0] > g.shape[1]:
                g = cv2.rotate(g, cv2.ROTATE_90_CLOCKWISE)
            notes.append(cv2.resize(g, (1000, int(g.shape[0] * 1000 / g.shape[1])), interpolation=cv2.INTER_AREA))
    s0 = real["noise_sigma_at_check_scale"]
    grid = [("dull paper, no noise, no JPEG", 0.0, 0.0, 0), ("dull paper, measured noise", 0.0, s0, 0),
            ("dull paper, measured noise, JPEG 90", 0.0, s0, 90), ("dull paper, measured noise, JPEG 50", 0.0, s0, 50),
            ("dull paper, 2x noise, JPEG 90", 0.0, 2 * s0, 90), ("dull paper, 4x noise, JPEG 90", 0.0, 4 * s0, 90),
            ("dull paper, 8x noise, JPEG 90", 0.0, 8 * s0, 90),
            ("glowing paper d=0.05, measured noise, JPEG 90", 0.05, s0, 90), ("glowing paper d=0.10, measured noise, JPEG 90", 0.10, s0, 90),
            ("glowing paper d=0.20, measured noise, JPEG 90", 0.20, s0, 90), ("glowing paper d=0.40, measured noise, JPEG 90", 0.40, s0, 90)]
    out = {"date": "2026-10-13", "kind": "SIMULATION with stated assumptions; no counterfeit note was photographed",
           "measured_on_real_uv_photographs": real, "UV_MIN_FIBRES": thr, "sheets": len(notes), "conditions": {}}
    for name, d, sigma, jpeg in grid:
        counts = np.array([fibre_mask(inner(sheet(g, real, d, sigma, jpeg, rng)))[1] for g in notes])
        acc = int((counts >= thr).sum())
        out["conditions"][name] = {"glow_d": d, "noise_sigma": round(sigma, 2), "jpeg": jpeg, "wrongly_accepted": acc, "n": len(counts),
                                   "rate": round(acc / len(counts), 4), "fibres_median": float(np.median(counts)),
                                   "fibres_95th": float(np.percentile(counts, 95)), "fibres_max": int(counts.max())}
        print(name, out["conditions"][name], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "uv_stress_sim.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["measured_on_real_uv_photographs"], indent=1))


if __name__ == "__main__":
    main()
