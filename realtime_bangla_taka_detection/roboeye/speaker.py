"""Emotion-adaptive Windows SAPI speaker (background thread)."""

from __future__ import annotations

import queue
import threading

from .fer_emotion import tts_style_for_emotion


class Speaker:
    def __init__(self, rate: int = 0):
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._rate = rate
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        voice.Rate = self._rate
        try:
            while not self._stop.is_set():
                try:
                    item = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                phrase, rate = item
                try:
                    voice.Rate = rate
                    voice.Speak(phrase)
                except Exception:
                    pass
        finally:
            pythoncom.CoUninitialize()

    def say(self, phrase: str, emotion: str = "neutral") -> None:
        style = tts_style_for_emotion(emotion)
        text = f"{style['prefix']}{phrase}"
        self._q.put((text, int(self._rate + style["rate"])))

    def stop(self):
        self._stop.set()
        self._q.put(None)
