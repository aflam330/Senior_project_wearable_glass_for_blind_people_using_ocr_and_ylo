"""
OCR Mode — offline Bangla + English reading (EasyOCR, no internet).

Triggered only on ACTION button press (not continuous) to avoid
blocking the CPU. EasyOCR is lazy-loaded on first activation so the
application starts quickly and ~600 MB of model weight doesn't sit
in RAM while another mode is active.

After recognition, text is NFC-normalized and matched against
assets/ocr_lexicon.txt so typical Bangla sign errors get repaired
before TTS. Optional Tesseract (ben+eng) is used only if EasyOCR
returns nothing.
"""
import logging
import os
import re
import time
from typing import Optional

import cv2
import numpy as np

import config
from ocr_repair import repair_ocr_text, tesseract_available, tesseract_read
from ocr_text_region import _boxes as _text_boxes
from ocr_text_region import clean_text, dominant_block, order_lines
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
        self._reader = None   # lazy-loaded EasyOCR (printed)
        self._ekush = None    # lazy-loaded Ekush CNN (handwritten letters)

        # Capture and playback are two separate steps: the CAPTURE button
        # snaps a frame, runs OCR, and stores the result here; the READ
        # button speaks whatever is currently stored. This lets the user
        # hold the camera steady only for the (quick) capture, then move
        # it away before listening to a (possibly long) read-out.
        self._stored_text: Optional[str] = None
        self._stored_lang: str = "en"

    def _ensure_ekush(self):
        if self._ekush is None:
            try:
                from ekush_letters import EkushRecognizer
                self._ekush = EkushRecognizer()
            except Exception as exc:
                logger.warning("Ekush handwritten CNN not loaded: %s", exc)
                self._ekush = False
        return self._ekush if self._ekush not in (None, False) else None

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
                use_gpu = False
                try:
                    import torch
                    use_gpu = bool(torch.cuda.is_available()) and os.environ.get("OCR_GPU", "1") == "1"
                except Exception:  # noqa: BLE001
                    pass
                # GPU when there is one (a laptop): about ten times faster. The Pi has none and runs on CPU as before.
                self._reader = easyocr.Reader(["bn", "en"], gpu=use_gpu, verbose=False)
                logger.info("EasyOCR on %s", "GPU" if use_gpu else "CPU")
                logger.info("EasyOCR ready")
            except Exception as exc:
                logger.error("Failed to load EasyOCR: %s", exc)
                self._reader = None
        self._ensure_ekush()

    def deactivate(self) -> None:
        logger.info("OCR mode deactivated")

    def cleanup(self) -> None:
        self._reader = None
        self._ekush = None

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
        if frame is None:
            return "ক্যামেরা প্রস্তুত নয়"  # camera not ready

        if self._reader is None and self._ensure_ekush() is None:
            return "ইঞ্জিন লোড হয়নি, অনুগ্রহ করে অপেক্ষা করুন"  # engine not loaded

        texts = []
        self.last_boxes, self.last_boxes_t = [], time.time()
        v2 = getattr(config, "OCR_PIPELINE", "legacy") == "v2"
        if self._reader is not None and v2:
            texts = self._read_v2(frame)
        if self._reader is not None and not v2:
            preprocessed = self._preprocess(frame)
            decoder = getattr(config, "OCR_DECODER", "beamsearch")
            decode_kwargs = {
                "decoder": decoder,
                "canvas_size": getattr(config, "OCR_CANVAS_SIZE", 2560),
                "slope_ths": float(getattr(config, "OCR_SLOPE_THS", 0.1)),
            }
            if decoder == "beamsearch":
                decode_kwargs["beamWidth"] = getattr(config, "OCR_BEAM_WIDTH", 5)
            try:
                results = self._reader.readtext(
                    preprocessed,
                    detail=1,
                    paragraph=False,   # paragraph=True changes tuple format; keep False for stability
                    width_ths=0.7,
                    height_ths=0.7,
                    **decode_kwargs,
                )
            except TypeError:
                results = self._reader.readtext(
                    preprocessed,
                    detail=1,
                    paragraph=False,
                    width_ths=0.7,
                    height_ths=0.7,
                )
            except Exception as exc:
                logger.warning("EasyOCR inference error: %s", exc)
                results = []

            results = sorted(results, key=_reading_order_key)
            s = frame.shape[1] / float(preprocessed.shape[1])
            for b in _text_boxes(results):   # every box, for the preview; the kept ones are marked below
                self.last_boxes.append([int(b["x0"] * s), int(b["y0"] * s), int(b["x1"] * s), int(b["y1"] * s), b["c"], False])
            for item in results:
                if len(item) == 3:
                    _bbox, text, conf = item
                elif len(item) == 2:
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
                if len(item) == 3:
                    ys = [p[1] * s for p in item[0]]
                    for box in self.last_boxes:
                        if abs(box[1] - min(ys)) < 1 and abs(box[3] - max(ys)) < 1:
                            box[5] = True

            if not texts and tesseract_available():
                tess = tesseract_read(preprocessed)
                if tess:
                    texts = [tess]
                    logger.info("EasyOCR empty; using offline Tesseract")

        if not texts:
            ekush = self._ensure_ekush()
            if ekush is not None and ekush.ok:
                hw = ekush.read_frame(frame)
                if hw:
                    texts = [hw]
                    logger.info("Using Ekush handwritten letters: %s", hw[:80])

        if not texts:
            self._stored_text = None
            self._stored_lang = "en"
            return "কোনো লেখা পাওয়া যায়নি"  # no text found

        combined = clean_text(" ".join(texts)) if v2 else repair_ocr_text(" ".join(texts))
        lang = detect_language(combined)
        logger.info("OCR result (lang=%s, %d chars): %s", lang, len(combined), combined[:80])

        # Store for the READ button — capture and playback are deliberately
        # separate steps (see __init__ note).
        self._stored_text = combined
        self._stored_lang = lang

        if getattr(config, "OCR_SPEAK_ON_CAPTURE", True):
            # say what was read straight away, so the user hears the detection itself; READ repeats it
            return ("লেখা আছে: " if lang == "bn" else "The text says: ") + combined
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

    def _read_v2(self, frame: np.ndarray) -> list:
        """OCR pipeline v2: CLAHE, tuned EasyOCR, dominant text block, reading order (see config.OCR_PIPELINE)."""
        h, w = frame.shape[:2]
        w0 = w
        if w < 1200:
            frame = cv2.resize(frame, (1200, int(h * 1200 / w)), interpolation=cv2.INTER_CUBIC)
        elif w > 1600:
            frame = cv2.resize(frame, (1600, int(h * 1600 / w)), interpolation=cv2.INTER_AREA)
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        decoder = getattr(config, "OCR_DECODER", "beamsearch")
        kw = dict(getattr(config, "OCR_V2_PARAMS", {}))
        kw.update(decoder=decoder, canvas_size=getattr(config, "OCR_CANVAS_SIZE", 2560))
        if decoder == "beamsearch":
            kw["beamWidth"] = getattr(config, "OCR_BEAM_WIDTH", 5)
        try:
            det = self._reader.readtext(gray, detail=1, paragraph=False, width_ths=0.7, height_ths=0.7, **kw)
        except Exception as exc:
            logger.warning("EasyOCR inference error: %s", exc)
            return []
        block = dominant_block(det, getattr(config, "OCR_V2_MIN_CONF", 0.2))
        # boxes for the preview, in the camera picture's own pixels; the main text block is marked
        s, kept = w0 / float(gray.shape[1]), {(b["x0"], b["y0"]) for b in block}
        self.last_boxes = [[int(b["x0"] * s), int(b["y0"] * s), int(b["x1"] * s), int(b["y1"] * s), b["c"],
                            (b["x0"], b["y0"]) in kept] for b in _text_boxes(det)]
        text = order_lines(block)
        return [text] if text else []

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
