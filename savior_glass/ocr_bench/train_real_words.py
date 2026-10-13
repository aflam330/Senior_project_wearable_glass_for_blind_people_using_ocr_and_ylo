"""Fine-tune EasyOCR's Bangla+English recogniser on REAL word photographs (ICDAR 2019 MLT), mixed with rendered words.

Why: on real word photographs the glass has a CER of 39.2 % (Bangla) and 62.5 % (Latin script) after the settings
study (real_words_tune.py). Thresholds are exhausted; the remaining error is in the recogniser.

Data split of the MLT word pool (fixed before training, 2026-10-13):
  * TEST: the 500 + 500 words of real_words.py (seed 0). Never used here.
  * REAL VALIDATION: the 150 + 150 words of real_words_tune.py (seed 1). Used only to choose the checkpoint.
  * TRAIN: all other readable Bangla words whose characters the recogniser knows, and 8,000 Latin-script words
    (seed 2) under the same condition; upright crops only.
Each batch is half real crops and half rendered words (train_task7.py: frequency-list words with every benchmark
word removed, training fonts only), so that rendered text is not forgotten.

Checkpoint choice, per seed: every 250 steps, greedy CER on the real validation crops; the best step is kept, but
only steps whose CER on rendered validation line crops is within 0.5 points of the pretrained model's are eligible.

    python -m ocr_bench.train_real_words <seed> [steps]
Output: ocr_bench/finetuned/recognizer_real_seed<seed>.pth, results/ocr_bench/real_ft_seed<seed>.json
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
from .real_words import N
from .real_words_tune import pools

DEV = T7.DEV


def split():
    """(train, val, test) lists of (file, row group, index, text) per script, exactly as the earlier scripts drew them."""
    P = pools()
    rng = np.random.default_rng(0)
    test = {k: [v[j] for j in sorted(rng.choice(len(v), N, replace=False))] for k, v in P.items()}
    taken = {k: set(test[k]) for k in P}
    rest = {k: [x for x in P[k] if x not in taken[k]] for k in P}
    rng1 = np.random.default_rng(1)
    val = {k: [rest[k][j] for j in sorted(rng1.choice(len(rest[k]), 150, replace=False))] for k in rest}
    vset = {k: set(val[k]) for k in P}
    train = {k: [x for x in rest[k] if x not in vset[k]] for k in P}
    return train, val, test


def load_crops(picks, charset=None, upright=True, limit=None, seed=2):
    """[(text, grey crop at height <= 64)] for the picked rows; with a charset, only words it can spell."""
    import pyarrow.parquet as pq
    from .real_words import DATA
    rows = [(f, g, i, B.nfc(t)) for v in picks.values() for f, g, i, t in v]
    if charset is not None:
        rows = [r for r in rows if 1 <= len(r[3]) <= 25 and all(c in charset for c in r[3])]
    out = {}
    by = {}
    for f, g, i, t in rows:
        by.setdefault((f, g), []).append((i, t))
    for (f, g), lst in by.items():
        col = pq.ParquetFile(DATA / f).read_row_group(g, columns=["image"]).column(0).to_pylist()
        for i, t in lst:
            im = cv2.imdecode(np.frombuffer(col[i]["bytes"], np.uint8), cv2.IMREAD_GRAYSCALE)
            if im is None or (upright and im.shape[1] < im.shape[0]):
                continue
            if im.shape[0] > 64:
                im = cv2.resize(im, (max(8, int(im.shape[1] * 64 / im.shape[0])), 64), interpolation=cv2.INTER_AREA)
            out[(f, g, i)] = (t, im)
    items = [out[k] for k in sorted(out)]
    if limit and len(items) > limit:
        idx = np.random.default_rng(seed).choice(len(items), limit, replace=False)
        items = [items[j] for j in sorted(idx)]
    return items


def augment(im, rng):
    if rng.random() < 0.5:
        im = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(2, 2)).apply(im)        # the glass enhances contrast first
    if rng.random() < 0.3:
        M = cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] / 2), rng.uniform(-3, 3), 1.0)
        im = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.25:
        im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.3, 1.0))
    return im


def main() -> None:
    import easyocr
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    reader = easyocr.Reader(["bn", "en"], gpu=DEV.type == "cuda", verbose=False)
    model = reader.recognizer.module if hasattr(reader.recognizer, "module") else reader.recognizer
    model = model.to(DEV).float().train()
    converter, charset = reader.converter, set(reader.character)
    tr, va, _ = split()
    bn = load_crops({"bn_IN": tr["bn_IN"]}, charset)
    en = load_crops({"en_XX": tr["en_XX"]}, charset, limit=8000)
    real_val = load_crops(va, None, upright=False)                 # every validation word counts, also rotated ones
    words = T7.corpus(charset, seed)
    train_fonts = {"bn": B.BN_FONT["train"], "en": B.EN_FONT["train"]}
    rv = T7.val_crops({"bn": B.BN_FONT["val"], "en": B.EN_FONT["val"]})

    def cer_on(items):
        hyps = []
        for i in range(0, len(items), 16):
            hyps += T7.greedy(model, converter, [im for _, im in items[i:i + 16]])
        return float(np.mean([B.cer(g, h) for (g, _), h in zip(items, hyps)]))

    rng = random.Random(seed)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-5, total_steps=steps, pct_start=0.1)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    base_real, base_rend = cer_on(real_val), cer_on(rv)
    hist = [{"step": 0, "real_val_crop_cer": base_real, "rendered_val_crop_cer": base_rend}]
    print(hist[-1], "train", {"bn": len(bn), "en": len(en)}, "real val", len(real_val), flush=True)
    best = (base_real, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    for step in range(1, steps + 1):
        texts, imgs = [], []
        for j in range(16):
            if j < 8:                                              # real: half Bangla, half Latin script
                t, im = (bn if j < 4 else en)[rng.randrange(len(bn if j < 4 else en))]
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
        if step % 250 == 0:
            r, s = cer_on(real_val), cer_on(rv)
            hist.append({"step": step, "loss": float(loss.item()), "real_val_crop_cer": r, "rendered_val_crop_cer": s})
            print(hist[-1], flush=True)
            if r < best[0] and s <= base_rend + 0.005:
                best = (r, step, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()})
    T7.FT.mkdir(parents=True, exist_ok=True)
    torch.save(best[2], T7.FT / f"recognizer_real_seed{seed}.pth")
    (T7.OUT / f"real_ft_seed{seed}.json").write_text(json.dumps({
        "date": "2026-10-13", "seed": seed, "steps": steps, "best_step": best[1], "best_real_val_crop_cer": best[0],
        "pretrained": hist[0], "history": hist, "train_real_words": {"bn": len(bn), "en": len(en)},
        "real_val_words": len(real_val), "batch": "8 real (4 Bangla, 4 Latin) + 8 rendered"}, indent=1), encoding="utf-8")
    print("best step", best[1], "real val crop CER", best[0], "pretrained", base_real)


if __name__ == "__main__":
    main()
