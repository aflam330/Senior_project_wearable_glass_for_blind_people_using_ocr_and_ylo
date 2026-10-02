"""Task 7: fine-tune EasyOCR's Bangla+English recognizer (CRNN, CTC) on rendered text. No benchmark word is trained on.

Training strings: words of the general 50k frequency lists (Bangla and English), keeping only characters the
recognizer knows, and REMOVING every word that occurs in any benchmark phrase (lexicon / select / val / test).
Plus random numbers in both scripts and 2-3 word combinations. Batch 16, width <= 800 px, mixed precision (4 GB GPU). Rendered with TRAIN_FONTS only (never scored),
with blur, low resolution, rotation, noise, contrast and JPEG damage.
Checkpoint choice: line crops of the VAL phrases in VAL fonts (fixed seed), greedy-decoded CER, every 250 steps.
Usage: python ocr_bench/train_task7.py <seed> [steps]
Output: results/ocr_bench/t7_seed<seed>.json, ocr_bench/finetuned/recognizer_seed<seed>.pth (best val step)
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ocr_bench import bench as B  # noqa: E402
from ocr_bench.shaped_text import render_mask  # noqa: E402

OUT = HERE.parent / "results" / "ocr_bench"
FT = HERE / "finetuned"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_H = 64


def bench_words():
    w = set()
    for sets in B.PHRASES.values():
        for lang in ("bn", "en"):
            for p in sets[lang]:
                for t in B.nfc(p).split():
                    w.add(t)
                    w.add(t.lower())
    return w


def corpus(charset, seed):
    banned = bench_words()
    words = {"bn": [], "en": []}
    for lang in ("bn", "en"):
        for line in (HERE / "lexicon" / f"{lang}_50k.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            t = B.nfc(parts[0])
            if not (2 <= len(t) <= 14) or t in banned or t.lower() in banned or not all(c in charset for c in t):
                continue
            if lang == "en" and not re.fullmatch(r"[A-Za-z]+", t):
                continue
            if lang == "bn" and not re.search(r"[ঀ-৿]", t):
                continue
            words[lang].append(t)
    return words


def sample_string(words, rng):
    r = rng.random()
    if r < 0.08:
        d = str(rng.randint(1, 99999))
        return d.translate(str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")) if rng.random() < 0.5 else d, "bn" if rng.random() < 0.5 else "en"
    lang = "bn" if rng.random() < 0.6 else "en"
    n = rng.choice((1, 1, 2, 2, 3))
    toks = [rng.choice(words[lang]) for _ in range(n)]
    if lang == "en":
        style = rng.random()
        toks = [t.upper() if style < 0.2 else t.capitalize() if style < 0.7 else t.lower() for t in toks]
    return " ".join(toks), lang


def render_line(text, lang, fonts, rng, damage=True):
    path, idx = fonts[lang][rng.randrange(len(fonts[lang]))]
    m = render_mask(text, path, rng.randint(28, 46), idx).astype(np.float32) / 255
    ink, bg = rng.uniform(0, 70), rng.uniform(170, 255)
    img = bg * (1 - m) + ink * m
    pad = rng.randint(4, 14)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg)
    if damage:
        if rng.random() < 0.5:
            M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rng.uniform(-3, 3), 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=bg)
        if rng.random() < 0.6:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.3))
        if rng.random() < 0.5:
            s = rng.uniform(0.35, 0.8)
            img = cv2.resize(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (img.shape[1], img.shape[0]))
        img = img + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0, 8), img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        if rng.random() < 0.4:
            _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 80)])
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return np.clip(img, 0, 255).astype(np.uint8)


def to_batch(imgs):
    """Like EasyOCR's AlignCollate(keep_ratio_with_pad): height 64, width by ratio, right-padded."""
    ws = [min(int(np.ceil(IMG_H * im.shape[1] / im.shape[0])), 800) for im in imgs]
    W = max(ws)
    out = torch.zeros(len(imgs), 1, IMG_H, W)
    for i, (im, w) in enumerate(zip(imgs, ws)):
        r = cv2.resize(im, (w, IMG_H), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
        t = (torch.from_numpy(r) - 0.5) / 0.5
        out[i, 0, :, :w] = t
        if w < W:
            out[i, 0, :, w:] = t[:, -1:].expand(IMG_H, W - w)
    return out


def greedy(model, converter, imgs):
    model.eval()
    with torch.inference_mode():
        x = to_batch(imgs).to(DEV)
        preds = model(x, None)
        idx = preds.argmax(2).cpu().numpy()
    model.train()
    return converter.decode_greedy(idx.reshape(-1), np.array([idx.shape[1]] * len(imgs)))


def val_crops(fonts):
    rng = random.Random(1234)
    items = []
    for lang in ("bn", "en"):
        for p in B.PHRASES["val"][lang]:
            for _ in range(3):
                items.append((p, render_line(p, lang, fonts, rng, damage=True)))
    return items


def main() -> None:
    import easyocr
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    reader = easyocr.Reader(["bn", "en"], gpu=DEV.type == "cuda", verbose=False)
    model = reader.recognizer.module if hasattr(reader.recognizer, "module") else reader.recognizer
    model = model.to(DEV).float().train()
    converter = reader.converter
    charset = set(reader.character)
    words = corpus(charset, seed)
    train_fonts = {"bn": B.BN_FONT["train"], "en": B.EN_FONT["train"]}
    val_fonts = {"bn": B.BN_FONT["val"], "en": B.EN_FONT["val"]}
    vc = val_crops(val_fonts)

    def val_cer():
        hyps = []
        for i in range(0, len(vc), 16):
            hyps += greedy(model, converter, [im for _, im in vc[i:i + 16]])
        return float(np.mean([B.cer(g, h) for (g, _), h in zip(vc, hyps)]))

    rng = random.Random(seed)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-5, total_steps=steps, pct_start=0.1)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    hist = [{"step": 0, "val_crop_cer": val_cer()}]
    print(hist[-1], "train words", {k: len(v) for k, v in words.items()}, flush=True)
    best = (hist[0]["val_crop_cer"], 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    for step in range(1, steps + 1):
        batch = [sample_string(words, rng) for _ in range(16)]
        texts = [t for t, _ in batch]
        imgs = [render_line(t, l, train_fonts, rng) for t, l in batch]
        x = to_batch(imgs).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
            logits = model(x, None)
        preds = logits.float().log_softmax(2)
        tgt, lens = converter.encode(texts)
        T_ = preds.size(1)
        loss = ctc(preds.permute(1, 0, 2), tgt.to(DEV), torch.full((len(texts),), T_, dtype=torch.long), lens.to(DEV))
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 250 == 0:
            v = val_cer()
            hist.append({"step": step, "loss": float(loss.item()), "val_crop_cer": v})
            print(hist[-1], flush=True)
            if v < best[0]:
                best = (v, step, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()})
    FT.mkdir(parents=True, exist_ok=True)
    torch.save(best[2], FT / f"recognizer_seed{seed}.pth")
    (OUT / f"t7_seed{seed}.json").write_text(json.dumps({"seed": seed, "steps": steps, "best_step": best[1], "best_val_crop_cer": best[0],
                                                         "pretrained_val_crop_cer": hist[0]["val_crop_cer"], "history": hist,
                                                         "train_words": {k: len(v) for k, v in words.items()},
                                                         "banned_benchmark_words": len(bench_words())}, indent=1), encoding="utf-8")
    print("best step", best[1], "val crop CER", best[0], "pretrained", hist[0]["val_crop_cer"])


if __name__ == "__main__":
    main()
