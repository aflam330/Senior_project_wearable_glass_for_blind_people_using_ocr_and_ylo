"""A small vision-language model run locally, as a free stand-in for the online description mode: text reading on the
OCR benchmark (same images, prompt and scoring as scripts/eval_online_mode.py), plus latency.

    python scripts/eval_local_vlm.py --check      # 12 VALIDATION images: does it run, what does it answer
    python scripts/eval_local_vlm.py              # 144 TEST images (image seed 0), read once for this method

Model: HuggingFaceTB/SmolVLM-500M-Instruct by default (LOCAL_VLM to change). Downloaded from the Hugging Face hub on
first use. Nothing is sent anywhere: the pictures stay on this machine.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocr_bench import bench as B  # noqa: E402

MODEL = os.environ.get("LOCAL_VLM", "HuggingFaceTB/SmolVLM-500M-Instruct")
PROMPT = ("Read the main printed text in this photo (a sign, label or page). Reply with that text only, exactly as written, "
          "in its original script (Bangla or English). No translation, no explanation, no quotes.")


def main() -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    check = "--check" in sys.argv
    dev = "cuda" if torch.cuda.is_available() and "--cpu" not in sys.argv else "cpu"
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    items = B.build("val", 0)[:12] if check else B.build("test", 0)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    rows = []
    for i, it in enumerate(items):
        img = Image.fromarray(cv2.cvtColor(B.decode(it), cv2.COLOR_BGR2RGB))
        t0 = time.perf_counter()
        inp = proc(text=text, images=[img], return_tensors="pt").to(dev)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=64, do_sample=False)
        hyp = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"seed": 0, "lang": it["lang"], "cond": it["cond"], "gt": it["gt"], "hyp": hyp, "cer": B.cer(it["gt"], hyp),
                     "wer": B.wer(it["gt"], hyp), "exact": B.nfc(it["gt"]) == B.nfc(hyp), "ms": ms})
        print(f"{i + 1}/{len(items)} {ms:.0f} ms  {it['gt']} => {hyp[:60]}", flush=True)
    res = B.summarise(rows)
    tag = MODEL.split("/")[-1].lower().replace("-", "_")
    out = ROOT / "results" / (f"local_vlm_{tag}_val_check12.json" if check else f"local_vlm_{tag}_ocr_test.json")
    slim = {k: v for k, v in res.items() if k != "rows"}
    slim.update({"model": MODEL, "device": dev, "set": "val (first 12, sanity check)" if check else "test, image seed 0, 144 images",
                 "prompt": PROMPT, "rows": [{k: r[k] for k in ("lang", "cond", "gt", "hyp", "cer")} for r in rows]})
    out.write_text(json.dumps(slim, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{MODEL} on {dev}: CER {100 * res['overall']['cer']:.1f} %  Bangla {100 * res['bn']['cer']:.1f} %  "
          f"English {100 * res['en']['cer']:.1f} %  median {res['latency_ms_median']:.0f} ms  -> {out}")


if __name__ == "__main__":
    main()
