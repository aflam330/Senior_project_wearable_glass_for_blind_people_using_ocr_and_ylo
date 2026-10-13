"""The glass's text reading on REAL photographs: word images from the ICDAR 2019 MLT scene-text set.

Data: the Hugging Face copy sai-santhosh/mlt19-train-ocr-data (word crops from street and indoor photographs with the
text typed out; labels look like "bn_IN <word>" or "en_XX <word>"; "###" marks a word its annotators could not read).
Put its three parquet files in <project>/data set/mlt19_words/ (not in the repository).

Protocol, fixed before the first run (2026-10-13). Nothing is tuned on these images; every image is a test image.
  * Bangla: 500 words drawn at random (seed 0) from the readable bn_IN words. Latin script: 500 from en_XX
    (English, French, German, Italian). "###" words are left out, as the benchmark itself does.
  * Each word image is put the way the glass would see a sign: scaled so that its longer side is 400 pixels and
    placed in the middle of a 640 x 480 mid-grey frame. That frame goes through the pipeline that is in the glass
    (page_bench.reader_fn with the glass's settings: 1,200 px width, greedy decoder, main-text-block rule, clean-up).
  * Scored with the benchmark's own CER and WER (Unicode NFC, case-sensitive), per script.
  * Also reported, without choosing between them: the words that are wider than tall (upright), because some crops
    are rotated by 90 degrees and no setting of the glass reads those.

    python -m ocr_bench.real_words
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import bench as B
from .page_bench import OUT, _ocr_mode, reader_fn

DATA = Path(__file__).resolve().parents[2] / "data set" / "mlt19_words"
N = 500


def load():
    import pyarrow.parquet as pq
    pools = {"bn_IN": [], "en_XX": []}
    for f in sorted(DATA.glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            for i, t in enumerate(pf.read_row_group(g, columns=["text"]).column(0).to_pylist()):
                lang, _, txt = t.partition(" ")
                if lang in pools and txt and txt != "###":
                    pools[lang].append((f.name, g, i, txt))
    rng = np.random.default_rng(0)
    picks = {k: [v[j] for j in sorted(rng.choice(len(v), N, replace=False))] for k, v in pools.items()}
    want = {(f, g, i): (lang, txt) for lang, v in picks.items() for f, g, i, txt in v}
    items = []
    for f in sorted(DATA.glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            idx = [i for (ff, gg, i) in want if ff == f.name and gg == g]
            if not idx:
                continue
            col = pf.read_row_group(g, columns=["image"]).column(0).to_pylist()
            for i in idx:
                lang, txt = want[(f.name, g, i)]
                items.append({"lang": "bn" if lang == "bn_IN" else "en", "gt": txt, "bytes": col[i]["bytes"]})
    return items, {k: len(v) for k, v in pools.items()}


def frame_of(word_bgr):
    h, w = word_bgr.shape[:2]
    s = 400.0 / max(h, w)
    im = cv2.resize(word_bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    canvas = np.full((480, 640, 3), 128, np.uint8)
    y, x = (480 - im.shape[0]) // 2, (640 - im.shape[1]) // 2
    canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
    return canvas


def main(threads=4):
    import torch
    import config
    torch.set_num_threads(threads)
    items, pool = load()
    m = _ocr_mode()
    read = reader_fn(m, 1200, 1, config.OCR_V2_DECODER)
    read(frame_of(np.full((60, 200, 3), 200, np.uint8)))            # warm-up
    rows = []
    for k, it in enumerate(items):
        img = cv2.imdecode(np.frombuffer(it["bytes"], np.uint8), cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        hyp = read(frame_of(img))
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"lang": it["lang"], "gt": it["gt"], "hyp": hyp, "cer": B.cer(it["gt"], hyp), "wer": B.wer(it["gt"], hyp),
                     "exact": B.nfc(it["gt"]) == B.nfc(hyp), "empty": hyp.strip() == "", "upright": img.shape[1] >= img.shape[0],
                     "h": int(img.shape[0]), "w": int(img.shape[1]), "ms": ms})
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(items)}", flush=True)

    def summ(rs):
        if not rs:
            return None
        cer = np.array([r["cer"] for r in rs])
        boot = np.random.default_rng(1).choice(cer, (2000, len(cer))).mean(1)
        return {"n": len(rs), "cer": round(float(cer.mean()), 4), "cer_ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
                "median_cer": round(float(np.median(cer)), 4), "word_exact": round(float(np.mean([r["exact"] for r in rs])), 4),
                "nothing_read": round(float(np.mean([r["empty"] for r in rs])), 4)}

    res = {"date": "2026-10-13", "data": "ICDAR 2019 MLT word crops (HF sai-santhosh/mlt19-train-ocr-data), real photographs",
           "pool_readable_words": pool, "sample": f"{N} per script, seed 0; no tuning on these images",
           "pipeline": f"glass pipeline v2: 1,200 px, {config.OCR_V2_DECODER} decoder, main text block, clean-up; CPU, {threads} threads",
           "latency_ms_median": round(float(np.median([r["ms"] for r in rows])), 1)}
    for lang in ("bn", "en"):
        rs = [r for r in rows if r["lang"] == lang]
        res[lang] = {"all": summ(rs), "upright_only": summ([r for r in rs if r["upright"]]),
                     "rotated_only": summ([r for r in rs if not r["upright"]]),
                     "at_least_32px_high_and_upright": summ([r for r in rs if r["upright"] and r["h"] >= 32])}
    res["rows"] = [{k: r[k] for k in ("lang", "gt", "hyp", "cer", "upright", "h", "w")} for r in rows]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "real_words_mlt19.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
