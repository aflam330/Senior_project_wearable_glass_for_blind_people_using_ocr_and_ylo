"""Text-finding settings for real photographs: chosen on a REAL validation sample, then tested once.

The first test on real words (real_words.py: Bangla CER 41.7 %, Latin 64.9 %) showed that on a quarter to a half of
the words nothing is read. The detector thresholds and the confidence filter had been chosen on rendered text.

Protocol, fixed before the run (2026-10-13):
  * Real validation sample: 150 Bangla + 150 Latin-script words, seed 1, drawn from the pool WITHOUT the 1,000 test
    words of real_words.py. Same framing of each word.
  * Candidates (all declared here):
      A  current:  text 0.8, low-text 0.3, link 0.4, minimum confidence 0.2
      B  EasyOCR's own detector defaults: text 0.7, low-text 0.4, link 0.4, minimum confidence 0.2
      C  current thresholds, minimum confidence 0.1
      D  more permissive: text 0.6, low-text 0.3, link 0.4, minimum confidence 0.1
  * A candidate is admissible only if its CER on the rendered benchmark's validation set (432 images) is within
    0.5 points of A's. Among the admissible ones the lowest real-validation CER (mean of the two scripts) wins.
  * Only the winner is then run on the 1,000 real test words, once. If A wins, nothing changes and the first test
    stands.

    python -m ocr_bench.real_words_tune
Output: results/ocr_bench/real_words_tune.json
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from . import bench as B
from .page_bench import OUT, _ocr_mode, reader_fn
from .real_words import DATA, N, frame_of

CANDIDATES = {
    "A current (0.8 / 0.3 / 0.4, conf 0.2)": ({"text_threshold": 0.8, "low_text": 0.3, "link_threshold": 0.4}, 0.2),
    "B EasyOCR defaults (0.7 / 0.4 / 0.4, conf 0.2)": ({"text_threshold": 0.7, "low_text": 0.4, "link_threshold": 0.4}, 0.2),
    "C current thresholds, conf 0.1": ({"text_threshold": 0.8, "low_text": 0.3, "link_threshold": 0.4}, 0.1),
    "D permissive (0.6 / 0.3 / 0.4, conf 0.1)": ({"text_threshold": 0.6, "low_text": 0.3, "link_threshold": 0.4}, 0.1),
}


def pools():
    import pyarrow.parquet as pq
    p = {"bn_IN": [], "en_XX": []}
    for f in sorted(DATA.glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            for i, t in enumerate(pf.read_row_group(g, columns=["text"]).column(0).to_pylist()):
                lang, _, txt = t.partition(" ")
                if lang in p and txt and txt != "###":
                    p[lang].append((f.name, g, i, txt))
    return p


def fetch(picks):
    import pyarrow.parquet as pq
    want = {(f, g, i): (lang, txt) for lang, v in picks.items() for f, g, i, txt in v}
    items = []
    for f in sorted(DATA.glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            idx = [i for (ff, gg, i) in want if ff == f.name and gg == g]
            if idx:
                col = pf.read_row_group(g, columns=["image"]).column(0).to_pylist()
                for i in idx:
                    lang, txt = want[(f.name, g, i)]
                    items.append({"lang": "bn" if lang == "bn_IN" else "en", "gt": txt,
                                  "img": cv2.imdecode(np.frombuffer(col[i]["bytes"], np.uint8), cv2.IMREAD_COLOR)})
    return items


def score_real(read, items):
    out = {}
    for lang in ("bn", "en"):
        rs = [(it["gt"], read(frame_of(it["img"]))) for it in items if it["lang"] == lang]
        out[lang] = {"n": len(rs), "cer": round(float(np.mean([B.cer(g, h) for g, h in rs])), 4),
                     "nothing_read": round(float(np.mean([h.strip() == "" for g, h in rs])), 4),
                     "word_exact": round(float(np.mean([B.nfc(g) == B.nfc(h) for g, h in rs])), 4)}
    out["mean_cer"] = round((out["bn"]["cer"] + out["en"]["cer"]) / 2, 4)
    return out


def main(threads=4):
    import torch
    import config
    torch.set_num_threads(threads)
    P = pools()
    rng = np.random.default_rng(0)                         # the same draw as real_words.py: these are the test words
    test = {k: [v[j] for j in sorted(rng.choice(len(v), N, replace=False))] for k, v in P.items()}
    taken = {k: set(test[k]) for k in P}
    rest = {k: [x for x in P[k] if x not in taken[k]] for k in P}
    rng1 = np.random.default_rng(1)
    val = {k: [rest[k][j] for j in sorted(rng1.choice(len(rest[k]), 150, replace=False))] for k in rest}
    val_items = fetch(val)
    m = _ocr_mode()
    base_params, base_conf = dict(config.OCR_V2_PARAMS), config.OCR_V2_MIN_CONF
    res = {"date": "2026-10-13", "real_validation": "150 Bangla + 150 Latin-script words, seed 1, disjoint from the 1,000 test words",
           "candidates": {}}
    out_path = OUT / "real_words_tune.json"
    for name, (params, conf) in CANDIDATES.items():
        config.OCR_V2_PARAMS = dict(base_params, **params)
        config.OCR_V2_MIN_CONF = conf
        read = reader_fn(m, 1200, 1, config.OCR_V2_DECODER)
        real = score_real(read, val_items)
        synth = B.score(read, "val")
        res["candidates"][name] = {"real_val": real, "rendered_val_cer": round(synth["overall"]["cer"], 4),
                                   "rendered_val_cer_bn": round(synth["bn"]["cer"], 4), "rendered_val_cer_en": round(synth["en"]["cer"], 4)}
        out_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(name, res["candidates"][name], flush=True)
    names = list(CANDIDATES)
    base = res["candidates"][names[0]]["rendered_val_cer"]
    ok = [n for n in names if res["candidates"][n]["rendered_val_cer"] - base <= 0.005]
    winner = min(ok, key=lambda n: res["candidates"][n]["real_val"]["mean_cer"])
    res["admissible"], res["winner"] = ok, winner
    if winner != names[0]:
        params, conf = CANDIDATES[winner]
        config.OCR_V2_PARAMS = dict(base_params, **params)
        config.OCR_V2_MIN_CONF = conf
        read = reader_fn(m, 1200, 1, config.OCR_V2_DECODER)
        res["winner_on_real_test"] = score_real(read, fetch(test))
        first = json.loads((OUT / "real_words_mlt19.json").read_text(encoding="utf-8"))
        res["first_test_with_current_settings"] = {l: {"cer": first[l]["all"]["cer"], "nothing_read": first[l]["all"]["nothing_read"],
                                                       "word_exact": first[l]["all"]["word_exact"]} for l in ("bn", "en")}
    config.OCR_V2_PARAMS, config.OCR_V2_MIN_CONF = base_params, base_conf
    out_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "candidates"}, indent=1))


if __name__ == "__main__":
    main()
