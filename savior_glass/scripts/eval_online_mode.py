"""Measure the online vision mode on the OCR benchmark's held-out test set (text reading), with latency.

Needs internet and a key in savior_glass/.env (ANTHROPIC_API_KEY or OPENAI_API_KEY). Costs API credit:
by default 144 calls (test set, image seed 0: 48 phrases x clean / distorted / photo scene). --limit N for a smaller run.
The model is asked to return only the text it reads, so the result is comparable with the offline pipelines
(same images, same CER / WER). The first failed call stops the run; nothing is written unless every call succeeded.

  python scripts/eval_online_mode.py            # 144 images
  python scripts/eval_online_mode.py --limit 12 # quick check
Output: results/online_mode_ocr_test.json and one record in research_results/inbox/ (run
`python research_results/results_db.py collect` afterwards).
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modes.claude_mode import ClaudeMode  # noqa: E402
from ocr_bench import bench as B  # noqa: E402

PROMPT = ("Read the main printed text in this photo (a sign, label or page). Reply with that text only, exactly as written, "
          "in its original script (Bangla or English). No translation, no explanation, no quotes.")


def main() -> None:
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    mode = ClaudeMode()
    provider, key = mode._provider()
    if not key:
        sys.exit("No API key in the environment or savior_glass/.env")
    model = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini") if provider == "openai" else os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")
    items = B.build("test", 0)[:limit]
    rows = []
    for i, it in enumerate(items):
        jpeg = mode._encode_jpeg(B.decode(it))
        t0 = time.perf_counter()
        try:
            hyp = mode.describe(jpeg, prompt=PROMPT, provider=provider, key=key)
        except urllib.error.HTTPError as exc:
            sys.exit(f"call {i + 1} failed: HTTP {exc.code} {exc.read().decode('utf-8', 'replace')[:300]}\nNothing was written.")
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"seed": 0, "lang": it["lang"], "cond": it["cond"], "gt": it["gt"], "hyp": hyp, "cer": B.cer(it["gt"], hyp),
                     "wer": B.wer(it["gt"], hyp), "exact": B.nfc(it["gt"]) == B.nfc(hyp), "ms": ms})
        print(f"{i + 1}/{len(items)} {ms:.0f} ms  {it['gt']} => {hyp[:60]}", flush=True)
    res = B.summarise(rows)
    out = ROOT / "results" / ("online_mode_ocr_test.json" if limit is None else f"online_mode_ocr_test_limit{limit}.json")
    if limit is None:
        B.save(res, out, {"set": "test", "method": f"online vision ({provider}, {model})", "prompt": PROMPT, "image_seeds": [0],
                          "note": "one image seed only (cost); needs internet"})
    else:
        out.write_text(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{provider} {model}: CER {100 * res['overall']['cer']:.1f} %  Bangla {100 * res['bn']['cer']:.1f} %  "
          f"English {100 * res['en']['cer']:.1f} %  median {res['latency_ms_median']:.0f} ms  -> {out}")


if __name__ == "__main__":
    main()
