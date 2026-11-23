"""Recogniser fine-tune, version 2 (2026-11-23). Same data split and checkpoint rule as train_real_words.py; changes:
  * all upright Latin-script training words the recogniser can spell (version 1: 8,000), all Bangla training words;
  * batch 32 (8 Bangla real, 8 Latin real, 16 rendered) instead of 16, 6,000 steps instead of 3,000;
  * augmentation that copies the glass's path: the word is shrunk and enlarged again (low resolution), JPEG
    compressed, blurred, tilted a little, its contrast enhanced (CLAHE) or lowered.
TEST words (real_words.py, seed 0) are never loaded. The checkpoint is chosen on the real VALIDATION crops, only among
steps whose rendered validation crop CER is within 0.5 points of the pretrained model's.

    python -m ocr_bench.train_real_words_v2 <seed> [steps]
Output: ocr_bench/finetuned/recognizer_real_v2_seed<seed>.pth, results/ocr_bench/real_ft_v2_seed<seed>.json
"""
from __future__ import annotations

import json
import random
import sys

import cv2
import numpy as np
import torch

from . import bench as B
from . import train_task7 as T7
from .train_real_words import load_crops, split

DEV = T7.DEV


def augment(im, rng):
    h, w = im.shape[:2]
    if rng.random() < 0.5:                                         # low resolution, then enlarged (as the glass does)
        s = rng.uniform(0.3, 0.8)
        small = cv2.resize(im, (max(4, int(w * s)), max(4, int(h * s))), interpolation=cv2.INTER_AREA)
        im = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    if rng.random() < 0.3:
        ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 70)])
        im = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    if rng.random() < 0.5:
        im = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(2, 2)).apply(im)
    elif rng.random() < 0.3:
        a = rng.uniform(0.4, 0.8)
        im = np.clip(im.astype(np.float32) * a + 128 * (1 - a), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-4, 4), 1.0)
        im = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.3:
        im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.3, 1.2))
    return im


def main() -> None:
    import easyocr
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    reader = easyocr.Reader(["bn", "en"], gpu=DEV.type == "cuda", verbose=False)
    model = reader.recognizer.module if hasattr(reader.recognizer, "module") else reader.recognizer
    model = model.to(DEV).float().train()
    converter, charset = reader.converter, set(reader.character)
    tr, va, _ = split()
    bn = load_crops({"bn_IN": tr["bn_IN"]}, charset)
    en = load_crops({"en_XX": tr["en_XX"]}, charset)
    real_val = load_crops(va, None, upright=False)
    words = T7.corpus(charset, seed)
    train_fonts = {"bn": B.BN_FONT["train"], "en": B.EN_FONT["train"]}
    rv = T7.val_crops({"bn": B.BN_FONT["val"], "en": B.EN_FONT["val"]})

    def cer_on(items):
        model.eval()
        hyps = []
        with torch.no_grad():
            for i in range(0, len(items), 32):
                hyps += T7.greedy(model, converter, [im for _, im in items[i:i + 32]])
        model.train()
        return float(np.mean([B.cer(g, h) for (g, _), h in zip(items, hyps)]))

    rng = random.Random(seed)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-5, total_steps=steps, pct_start=0.1)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    base_real, base_rend = cer_on(real_val), cer_on(rv)
    hist = [{"step": 0, "real_val_crop_cer": base_real, "rendered_val_crop_cer": base_rend}]
    print(hist[-1], "train", {"bn": len(bn), "en": len(en)}, "real val", len(real_val), flush=True)
    best = (base_real, 0, None)
    for step in range(1, steps + 1):
        texts, imgs = [], []
        for j in range(32):
            if j < 16:
                pool = bn if j < 8 else en
                t, im = pool[rng.randrange(len(pool))]
                texts.append(t); imgs.append(augment(im, rng))
            else:
                t, lang = T7.sample_string(words, rng)
                texts.append(t); imgs.append(T7.render_line(t, lang, train_fonts, rng))
        x = T7.to_batch(imgs).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
            logits = model(x, None)
        preds = logits.float().log_softmax(2)
        tgt, lens = converter.encode(texts)
        loss = ctc(preds.permute(1, 0, 2), tgt.to(DEV), torch.full((len(texts),), preds.size(1), dtype=torch.long), lens.to(DEV))
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 500 == 0:
            r, s = cer_on(real_val), cer_on(rv)
            hist.append({"step": step, "loss": float(loss.item()), "real_val_crop_cer": r, "rendered_val_crop_cer": s})
            print(hist[-1], flush=True)
            if r < best[0] and s <= base_rend + 0.005:
                best = (r, step, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()})
                T7.FT.mkdir(parents=True, exist_ok=True)
                torch.save(best[2], T7.FT / f"recognizer_real_v2_seed{seed}.pth")
    (T7.OUT / f"real_ft_v2_seed{seed}.json").write_text(json.dumps({
        "date": "2026-11-23", "seed": seed, "steps": steps, "best_step": best[1], "best_real_val_crop_cer": best[0],
        "pretrained": hist[0], "history": hist, "train_real_words": {"bn": len(bn), "en": len(en)},
        "real_val_words": len(real_val), "batch": "16 real (8 Bangla, 8 Latin) + 16 rendered"}, indent=1), encoding="utf-8")
    print("best step", best[1], "real val crop CER", best[0], "pretrained", base_real)


if __name__ == "__main__":
    main()
