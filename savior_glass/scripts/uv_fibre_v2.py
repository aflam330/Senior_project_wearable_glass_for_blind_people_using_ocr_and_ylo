"""UV fibre check, version 2: count only ISOLATED fibres (2026-11-24).

Why: on the 36 counterfeit UV pictures (fake_bd_cur/uv_fake_cur) the deployed check counts glowing printed ink as
fibres - strokes of letters, guilloche and outlines, which lie close together - so it accepted all 36. The counterfeits
carry no fibres at all (stated by the photographer). A genuine note's fibres are short threads scattered on their own
over dull paper. Rule family, fixed before any test picture was read:

  candidate marks = every component of the deployed fibre-pixel mask (modes/uv_check.fibre_mask), at 600 px width
  isolated fibre  = a fibre-shaped component with no other candidate mark (fibre-shaped or not) within r px
  R0              deployed: fibres >= 1
  R1(r, a)        isolated fibres >= a                                  r in {10,15,20,30,40}, a in {1,2,3}
  R2(r, a, b)     isolated fibres >= a and non-isolated fibres <= b     b in {2,5,10,20,40}
  S (shortcut)    one threshold on colour saturation (not a UV feature), direction chosen - for the robustness check

Split: by group (serial for counterfeit fronts, the photo otherwise), each class on its own, seed 0, 60/20/20.
Choice: on train + validation, the rule with the highest (genuine accepted rate - counterfeit accepted rate); ties go
to the lower counterfeit rate, then to the simpler rule. Then the test groups, once.
Robustness (test groups only): every picture is given the OTHER class's colour statistics (Reinhard transfer in Lab to
the mean and spread of that class's train pictures); counterfeits are also blurred a little and JPEG-compressed (the
glass camera's path). A rule that depends on how the two sets were photographed fails here; one that depends on the
fibres should not.

    python scripts/uv_fibre_v2.py
Output: results/uv_auth/fibre_v2.json
"""
from __future__ import annotations

import csv
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modes.uv_check import fibre_mask, inner  # noqa: E402
from scripts.uv_auth_eval import DS, OUT, group_of  # noqa: E402

RS, AS, BS = (10, 15, 20, 30, 40), (1, 2, 3), (2, 5, 10, 20, 40)


def load(r):
    img = cv2.imread(str(DS / r["uv_path"]))
    if r["note_box"]:
        x1, y1, x2, y2 = (int(float(v)) for v in r["note_box"].split())
        img = img[max(y1, 0):y2, max(x1, 0):x2]
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def feats(img):
    """The numbers every rule needs, for one note crop (as UVChecker.check sees it: inner part of the box)."""
    _, n, ab, parts = fibre_mask(inner(img), return_parts=True)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    f = {"n": int(n), "saturation": float(hsv[..., 1].mean())}
    if parts is None or n == 0:
        f.update({f"iso{r}": 0 for r in RS})
        return f
    fc = parts["fibre_centres"]
    allc = np.vstack([fc, parts["other_centres"]]) if len(parts["other_centres"]) else fc
    d = np.sqrt(((fc[:, None, :] - allc[None, :, :]) ** 2).sum(-1))
    d[np.arange(len(fc)), np.arange(len(fc))] = np.inf          # a fibre is not its own neighbour
    nearest = d.min(1)
    f.update({f"iso{r}": int((nearest > r).sum()) for r in RS})
    return f


def rules():
    out = {"R0 fibres>=1": lambda f: f["n"] >= 1}
    for r, a in itertools.product(RS, AS):
        out[f"R1 iso{r}>={a}"] = (lambda f, r=r, a=a: f[f"iso{r}"] >= a)
    for r, a, b in itertools.product(RS, AS, BS):
        out[f"R2 iso{r}>={a}, rest<={b}"] = (lambda f, r=r, a=a, b=b: f[f"iso{r}"] >= a and f["n"] - f[f"iso{r}"] <= b)
    return out


def rates(rule, rows):
    g = [rule(r["f"]) for r in rows if r["y"]]
    c = [rule(r["f"]) for r in rows if not r["y"]]
    return {"genuine_accepted": f"{sum(g)}/{len(g)}", "gar": round(sum(g) / max(len(g), 1), 4),
            "counterfeits_accepted": f"{sum(c)}/{len(c)}", "far": round(sum(c) / max(len(c), 1), 4)}


def reinhard(img, mean, std):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    m, s = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
    out = (lab - m) / s * std + mean
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def lab_stats(imgs):
    st = np.array([np.r_[cv2.cvtColor(i, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32).mean(0),
                         cv2.cvtColor(i, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32).std(0)] for i in imgs])
    return st[:, :3].mean(0), st[:, 3:].mean(0)


def main():
    man = list(csv.DictReader((DS / "uv_auth" / "manifest.csv").open(encoding="utf-8")))
    by_class = defaultdict(set)
    for r in man:
        by_class[r["genuine_or_counterfeit"]].add(group_of(r))
    split = {}
    for c, gs in sorted(by_class.items()):
        gs = sorted(gs)
        random.Random(0).shuffle(gs)
        n = len(gs)
        split.update({g: ("train" if i < 0.6 * n else "val" if i < 0.8 * n else "test") for i, g in enumerate(gs)})
    rows = []
    for r in man:
        img = load(r)
        rows.append({"y": r["genuine_or_counterfeit"] == "genuine", "split": split[group_of(r)], "img": img,
                     "f": feats(img), "path": r["uv_path"]})
    dev = [r for r in rows if r["split"] in ("train", "val")]
    test = [r for r in rows if r["split"] == "test"]

    R = rules()
    sat_rules = {}
    for t in sorted({r["f"]["saturation"] for r in dev}):
        sat_rules[f"S sat>={t:.1f}"] = (lambda f, t=t: f["saturation"] >= t)
        sat_rules[f"S sat<={t:.1f}"] = (lambda f, t=t: f["saturation"] <= t)

    def pick(cands):
        def key(name):
            x = rates(cands[name], dev)
            return (x["gar"] - x["far"], -x["far"], -len(name))
        return max(cands, key=key)
    chosen, sat = pick(R), pick(sat_rules)

    # robustness pictures: each test picture with the other class's colour statistics (from TRAIN pictures only)
    tr = [r for r in rows if r["split"] == "train"]
    gm, gs_ = lab_stats([r["img"] for r in tr if r["y"]])
    fm, fs_ = lab_stats([r["img"] for r in tr if not r["y"]])
    swapped = []
    for r in test:
        if r["y"]:
            im = reinhard(r["img"], fm, fs_)
        else:
            im = reinhard(r["img"], gm, gs_)
            im = cv2.GaussianBlur(im, (0, 0), 1.0)
            ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 70])
            im = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        swapped.append({**r, "f": feats(im)})

    res = {"date": "2026-11-24", "rule_family": __doc__.split("Split:")[0].split("over dull paper.")[1].strip(),
           "split_sizes": {s: {"genuine": sum(r["y"] for r in rows if r["split"] == s),
                               "counterfeit": sum(not r["y"] for r in rows if r["split"] == s)} for s in ("train", "val", "test")},
           "n_candidates": len(R), "chosen": chosen, "saturation_shortcut": sat,
           "train_val": {n: rates(f, dev) for n, f in (("R0 fibres>=1", R["R0 fibres>=1"]), (chosen, R[chosen]), (sat, sat_rules[sat]))},
           "test": {n: rates(f, test) for n, f in (("R0 fibres>=1", R["R0 fibres>=1"]), (chosen, R[chosen]), (sat, sat_rules[sat]))},
           "test_colour_swapped": {n: rates(f, swapped) for n, f in (("R0 fibres>=1", R["R0 fibres>=1"]), (chosen, R[chosen]), (sat, sat_rules[sat]))},
           "per_picture_test": [{"path": r["path"], "genuine": r["y"], **r["f"]} for r in test]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fibre_v2.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_picture_test", "rule_family")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
