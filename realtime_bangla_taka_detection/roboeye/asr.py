"""Voice-command ASR for RoboEye (“What is this note?”, “Is it real?”)."""

from __future__ import annotations

import re
import threading

COMMANDS = [
    (r"what is this( note)?", "describe"),
    (r"describe", "describe"),
    (r"(is it )?(real|genuine|authentic)", "authenticity"),
    (r"(is it )?(fake|counterfeit)", "authenticity"),
    (r"help", "help"),
    (r"straighten|pose", "pose"),
    (r"emotion|how (do I|am I) look", "emotion"),
]


def parse_command(text: str) -> str | None:
    t = (text or "").strip().lower()
    if not t:
        return None
    for pattern, intent in COMMANDS:
        if re.search(pattern, t):
            return intent
    return "describe"


class VoiceCommands:
    """Push-to-talk listener. SPACE still speaks; V starts a listen window."""

    def __init__(self):
        self.backend = None
        self._recognizer = None
        self._mic = None
        self.last_text = ""
        self.last_intent = None
        self.error = None
        try:
            import speech_recognition as sr

            self._recognizer = sr.Recognizer()
            self._mic = sr.Microphone()
            self.backend = "speech_recognition"
            with self._mic as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.3)
        except Exception as exc:
            self.error = str(exc)
            self.backend = None

    @property
    def available(self) -> bool:
        return self._recognizer is not None and self._mic is not None

    def listen(self, timeout: float = 4.0, phrase_time_limit: float = 4.0) -> tuple[str | None, str | None]:
        if not self.available:
            return None, None
        import speech_recognition as sr

        try:
            with self._mic as source:
                audio = self._recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )
        except Exception:
            return None, None
        text = None
        try:
            text = self._recognizer.recognize_google(audio)
        except Exception:
            try:
                text = self._recognizer.recognize_sphinx(audio)
            except Exception:
                text = None
        self.last_text = text or ""
        self.last_intent = parse_command(text) if text else None
        return text, self.last_intent

    def listen_async(self, callback) -> None:
        def _run():
            text, intent = self.listen()
            callback(text, intent)

        threading.Thread(target=_run, daemon=True).start()
