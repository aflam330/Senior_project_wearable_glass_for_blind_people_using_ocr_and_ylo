"""PaddleOCR (PP-OCRv5, lang='en') on the English items of a benchmark set. Run with Python 3.10 (no paddle wheel for 3.14).

PaddleOCR has no Bangla model (lang 'bn' / 'bengali' -> "No models are available"), so Bangla is not scored.
Usage: py -3.10 ocr_bench/paddle_run.py val
Output: results/ocr_bench/<set>_t3_paddle_en.json (English only)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_bench import bench as B  # noqa: E402


def main() -> None:
    from paddleocr import PaddleOCR
    name = sys.argv[1] if len(sys.argv) > 1 else "val"
    ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, enable_mkldnn=False)
    rows = []
    for seed in B.SEEDS:
        for it in B.build(name, seed):
            if it["lang"] != "en":
                continue
            img = B.decode(it)
            t0 = time.perf_counter()
            res = ocr.predict(img)
            ms = (time.perf_counter() - t0) * 1000
            texts = []
            import numpy as np
            for page in res or []:
                rt = page.get("rec_texts") or []
                polys = [np.asarray(q, np.float32) for q in (page.get("rec_polys") or [])]
                if len(polys) != len(rt):
                    texts += list(rt)
                    continue
                order = sorted(range(len(rt)), key=lambda i: (round(float(polys[i][:, 1].mean()) / 20), float(polys[i][:, 0].min())))
                texts += [rt[i] for i in order]
            hyp = " ".join(" ".join(texts).split())
            rows.append({"seed": seed, "lang": "en", "cond": it["cond"], "gt": it["gt"], "hyp": hyp, "cer": B.cer(it["gt"], hyp),
                         "wer": B.wer(it["gt"], hyp), "exact": B.nfc(it["gt"]) == B.nfc(hyp), "ms": ms})
    r = B.summarise(rows)
    out = Path(__file__).resolve().parents[1] / "results" / "ocr_bench" / f"{name}_t3_paddle_en.json"
    B.save(r, out, {"set": name, "task": 3, "engine": "PaddleOCR PP-OCRv5 en", "note": "English items only; no Bangla model"})
    print("paddle EN", name, "CER", round(100 * r["overall"]["cer"], 1), "scene", round(100 * r["by_cond"]["scene"]["all"]["cer"], 1),
          "seeds", [round(100 * x, 1) for x in r["per_seed_cer"]], "ms", round(r["latency_ms_median"]))


if __name__ == "__main__":
    main()
