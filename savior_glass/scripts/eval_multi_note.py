"""Several notes in one picture: is the count right, and the spoken total?

Pictures are built for this test only:
  * notes: source notes of the detector's TEST split (never trained on), with the same random tilt, rotation,
    colour and brightness changes as the training generator, but smaller so that up to three fit;
  * backgrounds: COCO train2017 photographs (the training pictures used val2017 backgrounds);
  * 300 pictures each with 1, 2 and 3 notes that do not overlap by more than 5 % of the smaller note; then the
    generator's blur, noise and JPEG steps.
The counting rule (CurrencyMode.distinct_notes) has two numbers, both fixed before this test: the announce
confidence 0.60 and an overlap of 0.5 for "the same note seen twice". Nothing is tuned here.

    python scripts/eval_multi_note.py
Output: results/multi_note_total.json
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "realtime_bangla_taka_detection" / "scripts"))
import generate_synthetic_dataset as G  # noqa: E402
import config  # noqa: E402
from modes.currency_mode import CurrencyMode  # noqa: E402

TEST_DIR = ROOT / "data set" / "currency_yolo_data" / "test" / "images"
BG = ROOT / "data set" / "coco2017" / "train2017"
PER_COUNT, S = 300, 640


def test_sources():
    out = {}
    for f in TEST_DIR.iterdir():
        m = re.match(r"^(\d+)_taka_(.+)_\d+$", f.stem)
        if m:
            v, stem = m.group(1), m.group(2)
            for ext in (".jpg", ".jpeg", ".png", ".JPG"):
                p = G.SOURCE_DIR / v / (stem + ext)
                if p.exists():
                    out[(v, stem)] = p
                    break
    return sorted(out.items())


def picture(sources, n, bgs, rng):
    bg = Image.open(rng.choice(bgs)).convert("RGB")
    w, h = bg.size
    s = S / min(w, h)
    bg = bg.resize((max(S, int(w * s) + 1), max(S, int(h * s) + 1)))
    x0, y0 = rng.randint(0, bg.size[0] - S), rng.randint(0, bg.size[1] - S)
    canvas = bg.crop((x0, y0, x0 + S, y0 + S)).convert("RGBA")
    placed, values = [], []
    for (v, _), path in rng.sample(sources, n):
        note = G.augment_note(G.load_source(path))
        scale = rng.uniform(0.30, 0.48) * S / max(note.size)          # smaller than in training, so that three fit
        note = note.resize((max(8, int(note.size[0] * scale)), max(8, int(note.size[1] * scale))), Image.LANCZOS)
        nw, nh = note.size
        for _ in range(200):
            x, y = rng.randint(0, S - nw), rng.randint(0, S - nh)
            ok = True
            for (px, py, pw, ph) in placed:
                iw, ih = max(0, min(x + nw, px + pw) - max(x, px)), max(0, min(y + nh, py + ph) - max(y, py))
                if iw * ih > 0.05 * min(nw * nh, pw * ph):
                    ok = False
                    break
            if ok:
                break
        else:
            continue
        canvas.paste(note, (x, y), note)
        placed.append((x, y, nw, nh)); values.append(int(v))
    img = canvas.convert("RGB")
    if rng.random() < 0.2:
        img = G.motion_blur(img)
    if rng.random() < 0.5:
        img = G.add_noise(img)
    img = G.jpeg_compress(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), values


def main():
    rng = random.Random(7)
    random.seed(7); np.random.seed(7)
    sources = test_sources()
    bgs = sorted(BG.glob("*.jpg"))[:3000]
    mode = CurrencyMode()
    mode.activate()
    mode.verdict_enabled = False
    res = {"date": "2026-10-13", "test_source_notes": len(sources), "pictures_per_count": PER_COUNT,
           "announce_conf": config.CURRENCY_ANNOUNCE_CONF, "same_note_iou": config.CURRENCY_COUNT_IOU, "by_count": {}}
    all_tot = all_cnt = n_all = 0
    for n in (1, 2, 3):
        cnt_ok = tot_ok = made = 0
        over = under = 0
        while made < PER_COUNT:
            img, values = picture(sources, n, bgs, rng)
            if len(values) != n:
                continue
            made += 1
            notes = mode.distinct_notes(mode.detect_live(img))
            got = sorted(int(x["name"].split("_")[0]) for x in notes)
            cnt_ok += int(len(got) == n)
            tot_ok += int(got == sorted(values))
            over += int(len(got) > n); under += int(len(got) < n)
        res["by_count"][str(n)] = {"pictures": made, "count_right": cnt_ok, "every_note_right_so_total_right": tot_ok,
                                   "too_many_counted": over, "too_few_counted": under,
                                   "count_right_rate": round(cnt_ok / made, 4), "total_right_rate": round(tot_ok / made, 4)}
        all_cnt += cnt_ok; all_tot += tot_ok; n_all += made
        print(n, res["by_count"][str(n)], flush=True)
    res["all"] = {"pictures": n_all, "count_right_rate": round(all_cnt / n_all, 4), "total_right_rate": round(all_tot / n_all, 4)}
    res["limits"] = ["synthetic pictures; notes do not overlap by more than 5 %", "laptop path (PyTorch weights), not the INT8 graph of the Pi",
                     "not measured through the glass camera"]
    (HERE.parent / "results" / "multi_note_total.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["all"]))


if __name__ == "__main__":
    main()
