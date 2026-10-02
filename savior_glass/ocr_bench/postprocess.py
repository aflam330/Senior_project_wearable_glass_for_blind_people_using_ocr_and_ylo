"""Task 5: post-processing of OCR text. Nothing here is fitted to the val / test phrases.

rules        NFC, invisible characters, stray symbols at word edges, digit script that matches the words around it
dictionary   SymSpell over general 50k-word frequency lists (Bangla and English, OpenSubtitles 2018, MIT);
             an unknown word is replaced only when one in-vocabulary word is clearly best
byt5         Stup702/ByT5-Bengali-OCR-Correction (a sequence-to-sequence corrector), Bangla lines only
lexicon      the glass's existing ocr_repair (its lexicon is the old 24 test phrases; kept for comparison)
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

LEX = Path(__file__).resolve().parent / "lexicon"
BN_RANGE = re.compile(r"[ঀ-৿]")
BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
EN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
INVISIBLE = dict.fromkeys(map(ord, "​‌‍﻿­"), None)
EDGE = re.compile(r"^[\|\[\]\{\}\(\)<>~`'\"“”‘’_^*#@$%&;:,.!?\\/=+-]+|[\|\[\]\{\}\(\)<>~`'\"“”‘’_^*#@$%&;:,!?\\/=+-]+$")


def is_bn(tok: str) -> bool:
    return bool(BN_RANGE.search(tok))


def rules(text: str) -> str:
    t = unicodedata.normalize("NFC", text).translate(INVISIBLE)
    toks = [EDGE.sub("", w) for w in t.split()]
    toks = [w for w in toks if w]
    out = []
    for i, w in enumerate(toks):
        if w.isdigit() or all(c.isdigit() for c in w):
            nb = [x for x in (toks[i - 1] if i else "", toks[i + 1] if i + 1 < len(toks) else "") if x and not x.isdigit()]
            if nb and all(is_bn(x) for x in nb):
                w = w.translate(BN_DIGITS)
            elif nb and not any(is_bn(x) for x in nb):
                w = w.translate(EN_DIGITS)
        out.append(w)
    return " ".join(out)


@lru_cache(maxsize=1)
def _symspell():
    from symspellpy import SymSpell
    sp = {}
    for lang in ("bn", "en"):
        s = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        for line in (LEX / f"{lang}_50k.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                w = parts[0] if lang == "bn" else parts[0].lower()
                s.create_dictionary_entry(w, int(parts[1]))
        sp[lang] = s
    return sp


def dictionary(text: str, max_edit=1, min_len=3, margin=2.0) -> str:
    """Replace a word that is not in the vocabulary by the most frequent word within `max_edit` edits,
    only if it beats the runner-up by `margin`x in frequency. Case of English words is kept."""
    from symspellpy import Verbosity
    sp = _symspell()
    out = []
    for w in text.split():
        lang = "bn" if is_bn(w) else "en"
        key = w if lang == "bn" else w.lower()
        if len(key) < min_len or any(c.isdigit() for c in key) or not key.isalpha() and lang == "en":
            out.append(w)
            continue
        sugg = sp[lang].lookup(key, Verbosity.ALL, max_edit_distance=max_edit)
        if not sugg or sugg[0].distance == 0:
            out.append(w)
            continue
        best = sugg[0]
        rival = next((s for s in sugg[1:] if s.distance == best.distance), None)
        if rival is not None and best.count < margin * rival.count:
            out.append(w)
            continue
        rep = best.term
        if lang == "en":
            rep = rep.upper() if w.isupper() else rep.capitalize() if w[:1].isupper() else rep
        out.append(rep)
    return " ".join(out)


@lru_cache(maxsize=1)
def _byt5():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Stup702/ByT5-Bengali-OCR-Correction")
    m = AutoModelForSeq2SeqLM.from_pretrained("Stup702/ByT5-Bengali-OCR-Correction").to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return tok, m


def byt5(text: str) -> str:
    """Correct Bangla lines with the ByT5 corrector; English lines are left alone."""
    import torch
    if not is_bn(text):
        return text
    tok, m = _byt5()
    ids = tok(text, return_tensors="pt").input_ids.to(m.device)
    with torch.inference_mode():
        out = m.generate(ids, max_new_tokens=4 * len(text.encode("utf-8")) + 8, num_beams=4, repetition_penalty=2.0)
    return " ".join(tok.decode(out[0], skip_special_tokens=True).split())


def lexicon(text: str) -> str:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ocr_repair import repair_ocr_text
    return repair_ocr_text(text)


METHODS = {"none": lambda t: t, "rules": rules, "dictionary": dictionary, "rules+dictionary": lambda t: dictionary(rules(t)),
           "byt5": byt5, "rules+byt5": lambda t: byt5(rules(t)), "lexicon": lexicon}
