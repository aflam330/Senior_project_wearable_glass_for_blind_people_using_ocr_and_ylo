"""Character n-gram language model for OCR post-processing, queried with KenLM (Task 5 follow-up).

KenLM's Python module was built on this laptop with a portable MinGW g++ (no admin rights needed):
  Cython kenlm.pyx -> C++, then g++ over util/*.cc lm/*.cc util/double-conversion/*.cc python/score_sentence.cc,
  linked against python310.dll  ->  kenlm.cp310-win_amd64.pyd   (run this file with Python 3.10)
KenLM's own trainer (lmplz) needs Boost, which is not available here, so the ARPA file is written by this script:
a 5-gram character model, absolute discounting (D = 0.75) with Katz-style backoff, trained on the words of the general
50k Bangla and English frequency lists (each word a sequence of characters, weighted by its frequency count).
No benchmark phrase is used for training.

  py -3.10 ocr_bench/kenlm_lm.py build      write ocr_bench/lexicon/char5.arpa
  py -3.10 ocr_bench/kenlm_lm.py val        score post-processing variants on the saved validation readings
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, r"C:\Users\LAM\tools\kenlm-master")  # where the built kenlm .pyd lives on this laptop
ARPA = HERE / "lexicon" / "char5.arpa"
ORDER, D = 5, 0.75
SPACE = "▁"


def chars(word: str) -> list:
    return [SPACE if c == " " else c for c in word]


def build() -> None:
    counts = [Counter() for _ in range(ORDER + 1)]
    for lang in ("bn", "en"):
        for line in (HERE / "lexicon" / f"{lang}_50k.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            w = min(int(parts[1]), 2000)  # cap so a handful of very frequent words do not dominate
            toks = ["<s>"] + chars(parts[0] if lang == "bn" else parts[0].lower()) + ["</s>"]
            for n in range(1, ORDER + 1):
                for i in range(len(toks) - n + 1):
                    counts[n][tuple(toks[i:i + n])] += w
    vocab = sorted({g[0] for g in counts[1]})
    total = sum(c for g, c in counts[1].items() if g != ("<s>",))
    prob = [dict() for _ in range(ORDER + 1)]
    for g, c in counts[1].items():
        prob[1][g] = max(c - D, D / 2) / total if g != ("<s>",) else 0.0
    left = 1.0 - sum(prob[1].values())
    prob[1][("<unk>",)] = max(left, 1e-7)
    by_ctx = [defaultdict(list) for _ in range(ORDER + 1)]
    for n in range(2, ORDER + 1):
        ctx_total = Counter()
        for g, c in counts[n].items():
            ctx_total[g[:-1]] += c
        for g, c in counts[n].items():
            prob[n][g] = max(c - D, 0.0) / ctx_total[g[:-1]]
            by_ctx[n][g[:-1]].append(g)
    bow = [dict() for _ in range(ORDER + 1)]
    for n in range(2, ORDER + 1):  # backoff weight of each (n-1)-gram context
        for ctx, grams in by_ctx[n].items():
            seen = sum(prob[n][g] for g in grams)
            lower = sum(prob[n - 1].get(g[1:], 0.0) for g in grams)
            bow[n - 1][ctx] = max(1.0 - seen, 1e-9) / max(1.0 - lower, 1e-9)
    with open(ARPA, "w", encoding="utf-8") as f:
        f.write("\\data\\\n")
        for n in range(1, ORDER + 1):
            f.write(f"ngram {n}={len(prob[n])}\n")
        for n in range(1, ORDER + 1):
            f.write(f"\n\\{n}-grams:\n")
            for g in sorted(prob[n]):
                p = prob[n][g]
                lp = -99.0 if p <= 0 else math.log10(p)
                b = bow[n].get(g) if n < ORDER else None
                f.write(f"{lp:.6f}\t{' '.join(g)}" + (f"\t{math.log10(b):.6f}" if b else "") + "\n")
        f.write("\n\\end\\\n")
    print("wrote", ARPA, {n: len(prob[n]) for n in range(1, ORDER + 1)}, "vocab", len(vocab))


_model = None


def score(word: str) -> float:
    """log10 probability per character of one word under the character model (KenLM query)."""
    global _model
    if _model is None:
        import kenlm
        _model = kenlm.Model(str(ARPA))
    toks = chars(word)
    return _model.score(" ".join(toks), bos=True, eos=True) / (len(toks) + 1)


def rescore(text: str, margin: float) -> str:
    """Replace a word by a dictionary word one edit away only when the character model prefers it by `margin`
    (log10 per character). Words with digits and words shorter than 3 characters are left alone."""
    from symspellpy import Verbosity
    from ocr_bench import postprocess as PO
    sp = PO._symspell()
    out = []
    for w in text.split():
        lang = "bn" if PO.is_bn(w) else "en"
        key = w if lang == "bn" else w.lower()
        if len(key) < 3 or any(c.isdigit() for c in key) or (lang == "en" and not key.isalpha()):
            out.append(w)
            continue
        cands = sp[lang].lookup(key, Verbosity.ALL, max_edit_distance=1)
        if not cands or cands[0].distance == 0:
            out.append(w)
            continue
        base = score(key)
        best = max(cands, key=lambda s: score(s.term))
        if score(best.term) - base >= margin:
            rep = best.term
            if lang == "en":
                rep = rep.upper() if w.isupper() else rep.capitalize() if w[:1].isupper() else rep
            out.append(rep)
        else:
            out.append(w)
    return " ".join(out)


def val() -> None:
    from ocr_bench import bench as B, postprocess as PO
    res_dir = HERE.parent / "results" / "ocr_bench"
    t4 = json.loads((res_dir / "task4_best.json").read_text(encoding="utf-8"))
    rows0 = json.loads((res_dir / f"val_t4_cfg{t4['best_cfg_index']:02d}.json").read_text(encoding="utf-8"))["per_sample"]
    out = {}
    methods = {"rules": PO.rules}
    for m in (0.1, 0.25, 0.5, 1.0):
        methods[f"rules+kenlm_margin{m}"] = lambda t, m=m: rescore(PO.rules(t), m)
    for name, fn in methods.items():
        rows = []
        for r in rows0:
            hyp = fn(r["hyp"])
            rows.append({**r, "hyp": hyp, "cer": B.cer(r["gt"], hyp), "wer": B.wer(r["gt"], hyp), "exact": B.nfc(r["gt"]) == B.nfc(hyp), "ms": 0.0})
        s = B.summarise(rows)
        changed = sum(a["hyp"] != PO.rules(b["hyp"]) for a, b in zip(rows, rows0))
        out[name] = {"cer": s["overall"]["cer"], "cer_bangla": s["bn"]["cer"], "cer_english": s["en"]["cer"], "wer": s["overall"]["wer"],
                     "exact": s["overall"]["exact"], "readings_changed_vs_rules": changed, "n": len(rows)}
        print(f"{name:26s} CER {100 * s['overall']['cer']:.2f} BN {100 * s['bn']['cer']:.2f} EN {100 * s['en']['cer']:.2f} "
              f"WER {100 * s['overall']['wer']:.2f} changed {changed}", flush=True)
    (res_dir / "val_t5_kenlm.json").write_text(json.dumps({"set": "val", "source": f"val_t4_cfg{t4['best_cfg_index']:02d}.json",
                                                           "lm": "5-gram character ARPA, absolute discounting, queried with KenLM",
                                                           "methods": out}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"build": build, "val": val}[sys.argv[1] if len(sys.argv) > 1 else "val"]()
