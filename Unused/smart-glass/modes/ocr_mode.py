"""
OCR Mode — Bangla + English text reading via EasyOCR.

Triggered only on ACTION button press (not continuous) to avoid
blocking the CPU. EasyOCR is lazy-loaded on first activation so the
application starts quickly and ~600 MB of model weight doesn't sit
in RAM while another mode is active.
"""
import logging
import re
from typing import Optional

import cv2
import numpy as np

import config
from utils import detect_language
from .base_mode import BaseMode

logger = logging.getLogger("smart_glass.ocr_mode")

# A detection only "looks like text" if at least half of its non-space
# characters are letters or digits (Bangla, Latin, or Bangla numerals —
# prices, phone numbers, room numbers etc. are meaningful to read aloud).
# EasyOCR frequently emits short symbol/noise fragments (".. | --", "I I I",
# stray punctuation from edges and textures) that pass the confidence +
# length filters but are gibberish when read aloud — this catches those
# before they reach the TTS queue.
_CONTENT_RE = re.compile(r"[^\W_]", re.UNICODE)
_MIN_CONTENT_RATIO = 0.5


def _looks_like_text(text: str) -> bool:
    stripped = text.replace(" ", "")
    if not stripped:
        return False
    content_chars = len(_CONTENT_RE.findall(stripped))
    return (content_chars / len(stripped)) >= _MIN_CONTENT_RATIO


def _reading_order_key(item):
    """Sort EasyOCR detections top-to-bottom, then left-to-right.

    EasyOCR returns detections in whatever order its detector finds them,
    which often does NOT match the natural reading order of the page —
    joining them as-is interleaves unrelated lines into nonsense sentences.
    Using the bounding box's top-left corner restores natural reading order.
    """
    if item and len(item) >= 1 and item[0]:
        bbox = item[0]
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        return (min(ys), min(xs))
    return (0, 0)


class OCRMode(BaseMode):

    def __init__(self) -> None:
        self._reader = None   # lazy-loaded

        # Capture and playback are two separate steps: the CAPTURE button
        # snaps a frame, runs OCR, and stores the result here; the READ
        # button speaks whatever is currently stored. This lets the user
        # hold the camera steady only for the (quick) capture, then move
        # it away before listening to a (possibly long) read-out.
        self._stored_text: Optional[str] = None
        self._stored_lang: str = "en"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        logger.info("OCR mode activated")
        # Start each OCR session with a clean slate
        self._stored_text = None
        self._stored_lang = "en"
        if self._reader is None:
            logger.info("Loading EasyOCR model (Bangla + English) — first use…")
            try:
                import easyocr
                # gpu=False is correct for RPi 5 (no CUDA GPU)
                self._reader = easyocr.Reader(["bn", "en"], gpu=False, verbose=False)
                logger.info("EasyOCR ready")
            except Exception as exc:
                logger.error("Failed to load EasyOCR: %s", exc)
                self._reader = None

    def deactivate(self) -> None:
        logger.info("OCR mode deactivated")

    def cleanup(self) -> None:
        self._reader = None

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        """
        CAPTURE step (triggered by the capture/action button): snap the
        frame, run OCR, store the cleaned-up result for later playback,
        and return only a short confirmation to speak — NOT the full text.
        Call read_stored() (triggered by the READ button) to hear it.
        """
        if self._reader is None:
            return "ইঞ্জিন লোড হয়নি, অনুগ্রহ করে অপেক্ষা করুন"  # engine not loaded

        if frame is None:
            return "ক্যামেরা প্রস্তুত নয়"  # camera not ready

        preprocessed = self._preprocess(frame)
        try:
            results = self._reader.readtext(
                preprocessed,
                detail=1,
                paragraph=False,   # paragraph=True changes tuple format; keep False for stability
                width_ths=0.7,
                height_ths=0.7,
            )
        except Exception as exc:
            logger.warning("EasyOCR inference error: %s", exc)
            return "টেক্সট পড়তে সমস্যা হয়েছে"  # error reading text

        # Restore natural top-to-bottom, left-to-right reading order before
        # filtering — EasyOCR's detection order can interleave separate
        # lines/blocks, which is the main reason combined output sounded
        # jumbled and nonsensical.
        results = sorted(results, key=_reading_order_key)

        # Filter by confidence, minimum length, and "looks like real text"
        # EasyOCR detail=1 always returns (bbox, text, conf) 3-tuples
        texts = []
        for item in results:
            if len(item) == 3:
                _bbox, text, conf = item
            elif len(item) == 2:          # fallback: (text, conf)
                text, conf = item
            else:
                continue
            text = text.strip()
            if conf < config.OCR_CONFIDENCE or len(text) < config.OCR_MIN_CHARS:
                continue
            if not _looks_like_text(text):
                logger.debug("Discarding non-text OCR fragment (conf=%.2f): %r", conf, text)
                continue
            texts.append(text)

        if not texts:
            self._stored_text = None
            self._stored_lang = "en"
            return "কোনো লেখা পাওয়া যায়নি"  # no text found

        combined = " ".join(texts)
        lang = detect_language(combined)
        logger.info("OCR result (lang=%s, %d chars): %s", lang, len(combined), combined[:80])

        # Store for the READ button — capture and playback are deliberately
        # separate steps (see __init__ note).
        self._stored_text = combined
        self._stored_lang = lang

        if lang == "bn":
            return "লেখা সংরক্ষণ করা হয়েছে। শুনতে রিড বাটন চাপুন"   # "Text saved. Press READ to listen"
        return "Text captured. Press the READ button to listen."

    def read_stored(self) -> Optional[str]:
        """
        READ step (triggered by the dedicated read/playback button):
        speak the most recently captured text, or a short notice if
        nothing has been captured yet in this session.
        """
        if not self._stored_text:
            return "কোনো লেখা সংরক্ষিত নেই। প্রথমে ক্যাপচার করুন"  # "Nothing saved yet. Capture first"

        if self._stored_lang == "bn":
            return "পড়া হচ্ছে: " + self._stored_text   # "Reading: ..."
        return "Reading: " + self._stored_text

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Upscale → denoise → CLAHE.
        2x upscaling is the single biggest accuracy boost for Bangla script.

        NOTE: deliberately stops at grayscale and does NOT binarize
        (no adaptiveThreshold). EasyOCR's recognizer is trained on
        natural grayscale/color images with anti-aliased glyph edges;
        measured side-by-side on the same frames, binarizing *lowered*
        average confidence (~0.69 vs ~0.84 on Latin text) and pushed
        some valid detections below OCR_CONFIDENCE, causing text to be
        dropped entirely — the opposite of the intended effect.
        """
        h, w = frame.shape[:2]

        # 1. Upscale small/medium frames to ~1200px wide — critical for Bangla.
        # (Tested 1800/2400 too — larger upscales reduced confidence further,
        # likely amplifying blur; 1200 is the measured sweet spot.)
        target_w = 1200
        if w < target_w:
            scale = target_w / w
            frame = cv2.resize(
                frame, (target_w, int(h * scale)), interpolation=cv2.INTER_CUBIC
            )
        elif w > 1600:
            # Downscale only if very large
            scale = 1600 / w
            frame = cv2.resize(
                frame, (1600, int(h * scale)), interpolation=cv2.INTER_AREA
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Bilateral filter — smooths noise while keeping text edges sharp
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # 3. CLAHE for contrast normalisation (helps with uneven lighting)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        return enhanced
