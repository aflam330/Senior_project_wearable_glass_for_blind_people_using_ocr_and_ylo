"""SARA Trust Index — blind-safe multi-signal authenticity packet.

Unique deployment contract (for paper + glass):
  - NEVER speaks or returns the word "counterfeit" / "জাল" to the wearer.
  - Fuses denomination confidence, authenticity probability, image quality, and
    optional UV/watermark hints into one 0–100 Trust Index + Bangla next-action.
  - Every decision is a "Trust Ticket" (structured, auditable).

This is the assistive safety layer on top of BDCounterfeitAuth / YOLO.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class TrustTicket:
    trust: int  # 0..100
    stars: int  # 0..5
    verdict: str  # LIKELY_GENUINE | CHECK_BY_HAND | UNCERTAIN_NOTE
    reason_code: str
    reason_bn: str
    next_action_bn: str
    speak_bn: str
    signals: dict = field(default_factory=dict)
    refuses_fake_speech: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _blur_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def _pose_ok(crop_bgr: np.ndarray) -> bool:
    h, w = crop_bgr.shape[:2]
    if h < 24 or w < 24:
        return False
    ar = max(w, h) / max(min(w, h), 1)
    return 1.25 <= ar <= 3.6


def quality_signals(crop_bgr: Optional[np.ndarray]) -> dict:
    if crop_bgr is None or crop_bgr.size == 0:
        return {"brightness": 0.0, "blur_var": 0.0, "pose_ok": False, "too_dark": True, "too_blurry": True}
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    bri = _brightness(gray)
    blur = _blur_var(gray)
    return {
        "brightness": round(bri, 2),
        "blur_var": round(blur, 2),
        "pose_ok": _pose_ok(crop_bgr),
        "too_dark": bri < 42,
        "too_bright": bri > 225,
        "too_blurry": blur < 22.0,
    }


def sara_trust(
    *,
    denom_name: str,
    denom_conf: float,
    p_real: float,
    auth_min_conf: float = 0.95,
    supported_dens: tuple[str, ...] = ("200_taka", "500_taka", "1000_taka"),
    crop_bgr: Optional[np.ndarray] = None,
    uv_fibres: Optional[int] = None,
    watermark_prob: Optional[float] = None,
    denom_bn: str = "",
) -> TrustTicket:
    """Build a Trust Ticket. Never returns a counterfeit-speaking verdict."""
    q = quality_signals(crop_bgr)
    denom_conf = float(np.clip(denom_conf, 0, 1))
    p_real = float(np.clip(p_real, 0, 1))

    # Weighted trust (paper-facing, simple, auditable)
    # 45% auth + 25% denom + 20% quality + 10% optional extras
    quality = 1.0
    if q["too_dark"] or q["too_bright"]:
        quality *= 0.55
    if q["too_blurry"]:
        quality *= 0.6
    if not q["pose_ok"]:
        quality *= 0.7

    extra = 0.5
    extra_n = 0
    if uv_fibres is not None:
        extra += 1.0 if uv_fibres >= 6 else 0.15
        extra_n += 1
    if watermark_prob is not None:
        extra += float(np.clip(watermark_prob, 0, 1))
        extra_n += 1
    extra = extra / max(extra_n, 1) if extra_n else 0.5

    supported = denom_name in supported_dens
    auth_component = p_real if supported else 0.45
    trust_f = 100.0 * (0.45 * auth_component + 0.25 * denom_conf + 0.20 * quality + 0.10 * extra)
    trust = int(np.clip(round(trust_f), 0, 100))
    stars = int(np.clip(trust // 20, 0, 5))

    # Decision policy (never say fake)
    reason = "ok"
    reason_bn = "পরীক্ষা সম্পন্ন"
    next_bn = "লেনদেন সতর্কভাবে করুন"
    verdict = "CHECK_BY_HAND"

    if denom_conf < 0.35:
        reason, reason_bn, next_bn = "low_denom_conf", "নোট নিশ্চিত নয়", "আরও কাছে ধরুন"
        verdict = "UNCERTAIN_NOTE"
    elif not supported:
        reason, reason_bn, next_bn = "denom_not_in_auth_set", "এই মানের জাল-পরীক্ষা নেই", "হাতে যাচাই করুন"
        verdict = "CHECK_BY_HAND"
    elif q["too_dark"]:
        reason, reason_bn, next_bn = "too_dark", "অন্ধকারে ধরা পড়েছে", "আলো বাড়িয়ে আবার ধরুন"
    elif q["too_bright"]:
        reason, reason_bn, next_bn = "too_bright", "অতিরিক্ত উজ্জ্বল", "আলো কমিয়ে আবার ধরুন"
    elif q["too_blurry"]:
        reason, reason_bn, next_bn = "too_blurry", "ছবি ঝাপসা", "স্থির করে ধরুন"
    elif not q["pose_ok"]:
        reason, reason_bn, next_bn = "bad_pose", "নোট সোজা নয়", "নোট সোজা করে ধরুন"
    elif p_real >= auth_min_conf and trust >= 80:
        reason, reason_bn, next_bn = "high_trust", "উচ্চ আস্থা", "সম্ভবত আসল"
        verdict = "LIKELY_GENUINE"
    else:
        reason, reason_bn, next_bn = "auth_uncertain", "আস্থা কম", "আসল কিনা হাতে যাচাই করুন"
        verdict = "CHECK_BY_HAND"

    name = denom_bn or denom_name.replace("_", " ")
    if verdict == "LIKELY_GENUINE":
        speak = f"{name}। সম্ভবত আসল। আস্থা {stars} তারকা।"
    elif verdict == "UNCERTAIN_NOTE":
        speak = f"{reason_bn}। {next_bn}।"
    else:
        speak = f"{name}। {reason_bn}। {next_bn}।"

    return TrustTicket(
        trust=trust,
        stars=stars,
        verdict=verdict,
        reason_code=reason,
        reason_bn=reason_bn,
        next_action_bn=next_bn,
        speak_bn=speak,
        signals={
            "denom": denom_name,
            "denom_conf": round(denom_conf, 4),
            "p_real": round(p_real, 4),
            "auth_min_conf": auth_min_conf,
            "supported_denom": supported,
            "quality": q,
            "uv_fibres": uv_fibres,
            "watermark_prob": watermark_prob,
            "trust_raw": round(trust_f, 2),
        },
    )
