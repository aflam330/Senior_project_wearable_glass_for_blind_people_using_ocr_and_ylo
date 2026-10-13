"""The rendered benchmark's TEST set with the pipeline as it is in the glass now (one read per setting).

    python -m ocr_bench.test_current
Output: results/ocr_bench/test_current_<date>.json
"""
from __future__ import annotations

import datetime as dt
import json

from . import bench as B
from .page_bench import OUT, _ocr_mode, reader_fn


def main(threads=4):
    import torch
    import config
    torch.set_num_threads(threads)
    m = _ocr_mode()
    read = reader_fn(m, 1200, 1, config.OCR_V2_DECODER)
    r = B.score(read, "test")
    out = {"date": dt.date.today().isoformat(), "set": "rendered benchmark, TEST, 432 images (48 phrases x 3 conditions x 3 seeds)",
           "setting": {"decoder": config.OCR_V2_DECODER, "min_conf": config.OCR_V2_MIN_CONF, "params": config.OCR_V2_PARAMS,
                       "detector": type(m._reader.detector).__name__},
           "cer": round(r["overall"]["cer"], 4), "cer_bn": round(r["bn"]["cer"], 4), "cer_en": round(r["en"]["cer"], 4),
           "wer": round(r["overall"]["wer"], 4), "exact": round(r["overall"]["exact"], 4),
           "by_condition_cer": {c: round(v["all"]["cer"], 4) for c, v in r["by_cond"].items()},
           "per_seed_cer": [round(x, 4) for x in r["per_seed_cer"]], "latency_ms_median": round(r["latency_ms_median"], 0)}
    p = OUT / f"test_current_{out['date']}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
