"""Offline OCR cleanup for the glass reader (no internet).

EasyOCR Bangla often drops matras and splits words. This module:
  1) NFC-normalizes Unicode and collapses extra spaces
  2) Nearest-matches short lines against a local lexicon (signs, directions)
  3) Optionally runs Tesseract `ben+eng` if it is installed (Pi / Windows)

Not used for currency. Currency stays on YOLO.
"""
from __future__ import annotations

import logging
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("smart_glass.ocr_repair")

_ASSETS = Path(__file__).resolve().parent / "assets" / "ocr_lexicon.txt"

# Relative Levenshtein at or below this → accept lexicon replacement.
_PHRASE_MAX_CER = 0.34
_WORD_MAX_CER = 0.40
_WORD_MAX_EDITS = 2


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize_ocr(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.split())


@lru_cache(maxsize=1)
def load_lexicon() -> tuple[tuple[str, ...], tuple[str, ...]]:
    phrases: list[str] = []
    words: set[str] = set()
    if _ASSETS.is_file():
        for raw in _ASSETS.read_text(encoding="utf-8").splitlines():
            line = normalize_ocr(raw.split("#", 1)[0])
            if not line:
                continue
            phrases.append(line)
            for tok in line.split():
                if len(tok) >= 2:
                    words.add(tok)
    phrases.sort(key=len, reverse=True)
    return tuple(phrases), tuple(sorted(words, key=len, reverse=True))


def _best_match(query: str, candidates: Iterable[str], max_cer: float, max_edits: int | None) -> str | None:
    q = normalize_ocr(query)
    if not q:
        return None
    best = None
    best_d = 10**9
    for c in candidates:
        d = _lev(q, c)
        n = max(len(q), len(c), 1)
        if d < best_d and d / n <= max_cer and (max_edits is None or d <= max_edits):
            best_d = d
            best = c
            if d == 0:
                break
    return best


def repair_ocr_text(text: str) -> str:
    """Return lexicon-corrected text. Leaves long / unknown lines mostly intact."""
    raw = normalize_ocr(text)
    if not raw:
        return raw
    phrases, words = load_lexicon()

    n_tok = len(raw.split())
    if n_tok <= 6:
        hit = _best_match(raw, phrases, _PHRASE_MAX_CER, max_edits=None)
        if hit:
            if hit != raw:
                logger.info("OCR lexicon phrase: %r -> %r", raw, hit)
            return hit

    out = []
    for tok in raw.split():
        if len(tok) < 3:
            out.append(tok)
            continue
        hit = _best_match(tok, words, _WORD_MAX_CER, _WORD_MAX_EDITS)
        out.append(hit if hit else tok)
    fixed = " ".join(out)
    if fixed != raw:
        logger.info("OCR lexicon words: %r -> %r", raw, fixed)
    return fixed


def tesseract_available() -> bool:
    try:
        import pytesseract  # noqa: F401
        from shutil import which

        return which("tesseract") is not None
    except Exception:
        return False


def tesseract_read(gray_or_bgr) -> str:
    """Offline Tesseract Bangla+English. Empty string if not installed."""
    try:
        import pytesseract
    except Exception:
        return ""
    try:
        return normalize_ocr(
            pytesseract.image_to_string(gray_or_bgr, lang="ben+eng", config="--psm 6")
        )
    except Exception as exc:
        logger.debug("Tesseract skipped: %s", exc)
        return ""
