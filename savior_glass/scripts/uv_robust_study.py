"""UV fibre check: make it reject fibre-free printed sheets, and measure both sides with nothing chosen on test data.

scripts/uv_stress_sim.py showed that JPEG compression alone made the check accept 93 % of SIMULATED fibre-free
printed sheets: the colour smear at print edges was counted as fibres. A real fibre glows whitish, so a brightness
condition was added to the fibre finder (modes/uv_check.fibre_mask, lum_threshold).

Protocol (declared before the run, 2026-10-13):
  * Genuine side: the 111 UV photographs, in the same 5 contiguous blocks per value as uv_blocked_cv.py.
  * Fibre-free side: simulated printed sheets (uv_stress_sim.sheet) from 120 normal-light note photographs.
    Per value, the first 20 are DEVELOPMENT sheets and the last 20 are HELD-OUT sheets. Four camera/paper
    conditions each: measured noise + JPEG 90; measured noise + JPEG 50; glow 0.10 + JPEG 90; glow 0.20 + JPEG 90.
  * First attempt, a brightness condition (lum_threshold 0 to 0.15): no pair separated the two sides (balanced
    accuracy 0.50 in every fold; kept in results/uv_check/uv_robust_brightness_failed.json).
  * Second attempt, reported here: a COLOUR condition (a fibre pixel is less blue than the paper by more than the
    threshold, and lies on lit paper). It was explored on the development sheets and the genuine photographs before
    this run, so the genuine side of the figures below is not fully blind; the held-out sheets were not looked at.
  * For each fold, the colour threshold (0, 0.10 ... 0.30) and the fibre count T (1..60) are chosen together on
    the four training blocks and the development sheets: the pair with the highest balanced accuracy (mean of
    genuine accepted and sheets rejected); ties go to the smaller colour threshold, then to the smaller T. The pair is then applied to the held-out block and to the held-out sheets.
  * Reported: genuine acceptance over the five held-out blocks, and rejection of held-out sheets (mean over folds).

The fibre-free side is a simulation. A real counterfeit has still not been photographed.

    python -m scripts.uv_robust_study
Output: results/uv_check/uv_robust_study.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from modes.uv_check import fibre_mask, inner  # noqa: E402
from scripts.uv_blocked_cv import K, wilson  # noqa: E402
from scripts.uv_dataset import DATA, OUT, VALUES, read  # noqa: E402
from scripts import uv_stress_sim as S  # noqa: E402

LUMS = (0.0, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30)   # colour thresholds (the brightness idea failed: see the note)
CONDS = (("noise + JPEG 90", 0.0, 1, 90), ("noise + JPEG 50", 0.0, 1, 50), ("glow 0.10 + JPEG 90", 0.10, 1, 90),
         ("glow 0.20 + JPEG 90", 0.20, 1, 90))


def main():
    import cv2
    real = S.measure_real()
    rng = np.random.default_rng(0)
    # genuine photographs: counts for every brightness threshold
    rows = [r for r in csv.DictReader(open(DATA / "boxes.csv", encoding="utf-8")) if r["x1"] != ""]
    gen = []
    for v in VALUES:
        rv = sorted((r for r in rows if r["value"] == v), key=lambda r: r["file"])
        for i, r in enumerate(rv):
            img = read(DATA / v / r["file"])
            crop = inner(img[int(r["y1"]):int(r["y2"]), int(r["x1"]):int(r["x2"])])
            c = [fibre_mask(crop, colour_threshold=l)[1] for l in LUMS]
            if fibre_mask(crop)[2] is not None:
                gen.append({"value": v, "block": min(K - 1, i * K // len(rv)), "counts": c})
    # simulated fibre-free sheets
    sheets = []
    for v in VALUES:
        files = sorted(p for p in (S.RAW / v).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:S.PER_VALUE]
        for j, f in enumerate(files):
            g = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            if g.shape[0] > g.shape[1]:
                g = cv2.rotate(g, cv2.ROTATE_90_CLOCKWISE)
            g = cv2.resize(g, (1000, int(g.shape[0] * 1000 / g.shape[1])), interpolation=cv2.INTER_AREA)
            for name, d, nz, q in CONDS:
                im = inner(S.sheet(g, real, d, nz * real["noise_sigma_at_check_scale"], q, rng))
                sheets.append({"value": v, "part": "dev" if j < S.PER_VALUE // 2 else "held", "cond": name,
                               "counts": [fibre_mask(im, colour_threshold=l)[1] for l in LUMS]})
    G = np.array([g["counts"] for g in gen]); gb = np.array([g["block"] for g in gen])
    Sx = np.array([s["counts"] for s in sheets]); sp = np.array([s["part"] == "held" for s in sheets])

    def choose(gtr, str_):
        best = None
        for li in range(len(LUMS)):
            for T in range(1, 61):
                bal = 0.5 * ((gtr[:, li] >= T).mean() + (str_[:, li] < T).mean())
                if best is None or bal > best[0] + 1e-12:
                    best = (bal, li, T)
        return best

    folds, acc_g, rej_s = [], [], []
    gen_ok = np.zeros(len(gen), bool)
    for b in range(K):
        bal, li, T = choose(G[gb != b], Sx[~sp])
        te = gb == b
        gen_ok[te] = G[te, li] >= T
        r = float((Sx[sp][:, li] < T).mean())
        rej_s.append(r)
        folds.append({"held_out_block": b, "colour_threshold": LUMS[li], "T": T, "balanced_accuracy_on_training": round(float(bal), 4),
                      "genuine_accepted": int(gen_ok[te].sum()), "genuine_n": int(te.sum()), "held_out_sheets_rejected_rate": round(r, 4)})
    k = int(gen_ok.sum())
    per_value = {v: {"accepted": int(gen_ok[[g["value"] == v for g in gen]].sum()), "n": int(sum(g["value"] == v for g in gen))} for v in VALUES}
    # the setting for the glass: the same choice on ALL genuine photographs and the development sheets
    bal, li, T = choose(G, Sx[~sp])
    held = Sx[sp]
    per_cond = {}
    for name, *_ in CONDS:
        idx = np.array([s["cond"] == name and s["part"] == "held" for s in sheets])
        per_cond[name] = {"rejected": int((Sx[idx][:, li] < T).sum()), "n": int(idx.sum())}
    # what the check did before the brightness condition, on the same held-out sheets, at the threshold then in use (6)
    before = {"held_out_sheets_rejected": int((held[:, 0] < 6).sum()), "n": int(len(held)), "genuine_accepted_all_photos": int((G[:, 0] >= 6).sum())}
    res = {"date": "2026-10-13", "kind": "genuine side: real UV photographs; fibre-free side: SIMULATED printed sheets",
           "measured_on_real_uv_photographs": real, "folds": folds,
           "nested_result": {"genuine_accepted": k, "n": len(gen), "rate": round(k / len(gen), 4), "wilson95": wilson(k, len(gen)),
                             "per_value": per_value, "held_out_sheets_rejected_mean_over_folds": round(float(np.mean(rej_s)), 4),
                             "held_out_sheets": int(sp.sum())},
           "setting_for_the_glass": {"UV_COLOUR_THRESHOLD": LUMS[li], "UV_MIN_FIBRES": T, "chosen_on": "all genuine photographs + development sheets",
                                     "held_out_sheets_rejected": int((held[:, li] < T).sum()), "held_out_sheets": int(len(held)),
                                     "held_out_sheets_per_condition": per_cond,
                                     "genuine_accepted_all_photos_not_a_test": int((G[:, li] >= T).sum())},
           "before_brightness_condition_T6": before,
           "not_measured": ["a real counterfeit note", "real printed sheets under the UV lamp", "the glass camera"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "uv_robust_study.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
