"""
Online Claude vision mode.

Sends the current camera frame to the Anthropic Messages API and returns a
short Bangla/English assistive description (text, objects, Taka notes).

This is the online path. Offline text reading stays in ocr_mode.py (EasyOCR)
and is not used here.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import ssl
import urllib.error
import urllib.request
from typing import Optional

import cv2
import numpy as np

import config
from .base_mode import BaseMode

logger = logging.getLogger("smart_glass.claude_mode")

_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_PROMPT = (
    "You are Savior Glass, a camera aid for a visually impaired user in Bangladesh. "
    "Look at this photo and reply in concise spoken Bangla (short sentences). "
    "Read every visible Bangla or English word accurately. "
    "If a Bangladeshi Taka note is visible, say the denomination. "
    "Name important nearby objects. "
    "Do not mention that you are an AI. Do not use markdown."
)


def _load_dotenv() -> None:
    path = os.path.join(config.BASE_DIR, ".env")
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError as exc:
        logger.warning("Could not read .env: %s", exc)


class ClaudeMode(BaseMode):
    """Online scene + text description via Claude. EasyOCR is unchanged."""

    def __init__(self) -> None:
        self._stored_text: Optional[str] = None

    def activate(self) -> None:
        _load_dotenv()
        logger.info("Online Claude mode activated")
        if not self._api_key():
            logger.warning(
                "No Anthropic API key. Set ANTHROPIC_API_KEY or put it in savior_glass/.env"
            )

    def deactivate(self) -> None:
        logger.info("Online Claude mode deactivated")

    def cleanup(self) -> None:
        self._stored_text = None

    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        if frame is None:
            return "ক্যামেরা প্রস্তুত নয়"

        key = self._api_key()
        if not key:
            return (
                "অনলাইন ক্লড কী নেই। ANTHROPIC_API_KEY সেট করুন "
                "অথবা savior_glass ফোল্ডারে .env ফাইলে রাখুন।"
            )

        jpeg = self._encode_jpeg(frame)
        if jpeg is None:
            return "ছবি পাঠানো যায়নি"

        try:
            text = self._call_claude(key, jpeg)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            logger.error("Claude HTTP %s: %s", exc.code, body)
            if exc.code in (401, 403):
                return "ক্লড এপিআই কী ভুল অথবা অনুমতি নেই।"
            return "ক্লড এপিআই ব্যর্থ হয়েছে। ইন্টারনেট ও কী চেক করুন।"
        except urllib.error.URLError as exc:
            logger.error("Claude network error: %s", exc)
            return "ইন্টারনেট সংযোগ নেই। অফলাইন টেক্সট মোডে A চাপুন।"
        except Exception as exc:
            logger.error("Claude request failed: %s", exc)
            return "অনলাইন বিবরণ তৈরি করা যায়নি।"

        if not text:
            return "ক্লড কিছু বলেনি।"

        self._stored_text = text
        logger.info("Claude result (%d chars): %s", len(text), text[:80])
        return text

    def read_stored(self) -> Optional[str]:
        if not self._stored_text:
            return "কোনো অনলাইন বিবরণ নেই। আগে C চাপুন।"
        return self._stored_text

    def _api_key(self) -> str:
        _load_dotenv()
        return (
            os.environ.get("ANTHROPIC_API_KEY", "").strip()
            or getattr(config, "ANTHROPIC_API_KEY", "")
            or ""
        ).strip()

    def _encode_jpeg(self, frame: np.ndarray) -> Optional[str]:
        h, w = frame.shape[:2]
        max_side = 1280
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _call_claude(self, api_key: str, jpeg_b64: str) -> str:
        payload = {
            "model": getattr(config, "CLAUDE_MODEL", "claude-sonnet-4-5"),
            "max_tokens": int(getattr(config, "CLAUDE_MAX_TOKENS", 600)),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": jpeg_b64,
                            },
                        },
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ],
        }
        raw = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            _API_URL,
            data=raw,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
            },
        )
        context = ssl.create_default_context()
        with urllib.request.urlopen(request, timeout=60, context=context) as response:
            data = json.loads(response.read().decode("utf-8"))

        parts = []
        for block in data.get("content") or []:
            if block.get("type") == "text" and block.get("text"):
                parts.append(block["text"].strip())
        return "\n".join(parts).strip()
