"""Vision-language descriptions: Qwen2-VL, LLaVA, BLIP, then a structured template."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _try_qwen():
    try:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except Exception:
        return None
    name = "Qwen/Qwen2-VL-2B-Instruct"
    try:
        proc = AutoProcessor.from_pretrained(name)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            name, torch_dtype=torch.float32
        )
        model.eval()
    except Exception:
        return None

    def caption(crop_bgr: np.ndarray) -> str:
        rgb = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "Describe this Bangladeshi Taka banknote in one short sentence "
                            "for a blind user: denomination if visible, condition, and orientation."
                        ),
                    },
                ],
            }
        ]
        text = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=[text], images=[rgb], return_tensors="pt")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=48)
        return proc.batch_decode(out, skip_special_tokens=True)[0].strip()

    return caption


def _try_llava():
    try:
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration
    except Exception:
        return None
    # Tiny-LLaVA-class checkpoint that fits a laptop; 7B LLaVA is optional if already cached.
    for name in (
        "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B",
        "llava-hf/llava-1.5-7b-hf",
    ):
        try:
            proc = AutoProcessor.from_pretrained(name)
            model = LlavaForConditionalGeneration.from_pretrained(name)
            model.eval()
        except Exception:
            continue

        def caption(crop_bgr: np.ndarray, _proc=proc, _model=model) -> str:
            rgb = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            prompt = "USER: <image>\nDescribe this Bangladeshi Taka banknote in one sentence.\nASSISTANT:"
            inputs = _proc(images=rgb, text=prompt, return_tensors="pt")
            with torch.inference_mode():
                out = _model.generate(**inputs, max_new_tokens=48)
            return _proc.decode(out[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

        return caption
    return None


def _try_blip():
    try:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor
    except Exception:
        return None
    try:
        proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model.eval()
    except Exception:
        return None

    def caption(crop_bgr: np.ndarray) -> str:
        rgb = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        inputs = proc(rgb, return_tensors="pt")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=30)
        return proc.decode(out[0], skip_special_tokens=True).strip()

    return caption


class NoteDescriber:
    def __init__(self, prefer: str | None = None):
        self._fn = None
        self.backend = "template"
        order = [prefer] if prefer else []
        order += ["qwen", "llava", "blip"]
        loaders = {"qwen": _try_qwen, "llava": _try_llava, "blip": _try_blip}
        seen = set()
        for name in order:
            if not name or name in seen:
                continue
            seen.add(name)
            fn = loaders[name]()
            if fn is not None:
                self._fn = fn
                self.backend = name
                break

    def describe(
        self,
        crop_bgr: np.ndarray | None,
        name: str | None,
        conf: float,
        auth_label: str,
        genuine_prob: float,
        pose: dict | None = None,
    ) -> str:
        denom = (name or "unknown").replace("_", " ")
        auth = {
            "genuine": "The note appears genuine",
            "counterfeit": "Warning: the note appears counterfeit",
            "unknown": "Authenticity is uncertain",
        }.get(auth_label, "Authenticity is uncertain")
        pose_txt = ""
        if pose and pose.get("ok"):
            if pose.get("needs_straighten"):
                pose_txt = " Please straighten the note so it faces the camera."
            else:
                pose_txt = (
                    f" Orientation is about {pose['roll']:.0f} degrees roll "
                    f"and {pose['pitch']:.0f} degrees pitch."
                )
        template = (
            f"This looks like a {denom} banknote, confidence {conf * 100:.0f} percent. "
            f"{auth} ({genuine_prob * 100:.0f} percent genuine).{pose_txt}"
        )
        if self._fn is not None and crop_bgr is not None and crop_bgr.size:
            try:
                cap = self._fn(crop_bgr)
                if cap:
                    return f"{cap}. {template}"
            except Exception:
                pass
        return template
