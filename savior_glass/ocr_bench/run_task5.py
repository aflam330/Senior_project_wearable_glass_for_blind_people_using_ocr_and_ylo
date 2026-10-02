"""Task 5: post-processing, scored on the saved val readings of the Task 4 pipeline (no new OCR).

Each method in postprocess.METHODS is applied to every val reading (3 seeds); CER / WER recomputed;
text-only latency per line is measured. The best method on val is recorded.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_bench import bench as B, postprocess as PO  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "ocr_bench"


def main() -> None:
    t4 = json.loads((OUT / "task4_best.json").read_text(encoding="utf-8"))
    src = OUT / f"val_t4_cfg{t4['best_cfg_index']:02d}.json"
    rows0 = json.loads(src.read_text(encoding="utf-8"))["per_sample"]
    res = {}
    for name, fn in PO.METHODS.items():
        rows, ms = [], []
        for r in rows0:
            t0 = time.perf_counter()
            hyp = fn(r["hyp"])
            ms.append((time.perf_counter() - t0) * 1000)
            rows.append({**r, "hyp": hyp, "cer": B.cer(r["gt"], hyp), "wer": B.wer(r["gt"], hyp), "exact": B.nfc(r["gt"]) == B.nfc(hyp), "ms": 0.0})
        s = B.summarise(rows)
        s["postprocess_ms_median"] = float(np.median(ms))
        B.save(s, OUT / f"val_t5_{name.replace('+', '_')}.json", {"set": "val", "task": 5, "method": name, "source": src.name})
        res[name] = s["overall"]["cer"]
        print(f"{name:18s} CER {100*s['overall']['cer']:.2f} BN {100*s['bn']['cer']:.2f} EN {100*s['en']['cer']:.2f} "
              f"WER {100*s['overall']['wer']:.2f} exact {100*s['overall']['exact']:.1f}  {s['postprocess_ms_median']:.2f} ms/line", flush=True)
    best = min(res, key=res.get)
    (OUT / "task5_best.json").write_text(json.dumps({"best": best, "val_cer": res[best], "all": res, "source": src.name}, indent=1),
                                         encoding="utf-8")
    print("BEST", best, res[best])


if __name__ == "__main__":
    main()
