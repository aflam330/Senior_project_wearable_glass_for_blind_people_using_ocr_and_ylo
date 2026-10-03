"""Speech for testing the glass app on a Windows laptop (the Pi uses espeak-ng / Piper in utils.TTSEngine).

Windows has no espeak-ng and its built-in voices cannot speak Bangla, so:
  Bangla  -> Google TTS (gtts package, needs internet) saved as MP3 and played with Windows' media player
  English -> the built-in Windows voice (System.Speech)
If a Bangla line cannot be spoken (no internet), it is logged; the preview window still shows the text.
Same interface as utils.TTSEngine: speak, stop_current, set_volume, drain.
"""
from __future__ import annotations

import logging
import os
import queue
import re
import subprocess
import tempfile
import threading

logger = logging.getLogger("smart_glass.windows_tts")
_BN = re.compile(r"[ঀ-৿]")
_PLAY = (
    "Add-Type -AssemblyName PresentationCore; $p = New-Object System.Windows.Media.MediaPlayer; "
    "$p.Open([uri]'{path}'); $p.Volume = {vol}; $p.Play(); "
    "$n = 0; while (-not $p.NaturalDuration.HasTimeSpan -and $n -lt 50) {{ Start-Sleep -Milliseconds 100; $n++ }}; "
    "if ($p.NaturalDuration.HasTimeSpan) {{ Start-Sleep -Milliseconds ([int]$p.NaturalDuration.TimeSpan.TotalMilliseconds + 150) }}; $p.Close()"
)
_SAPI = (
    "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$s.Volume = {vol}; $s.Rate = 0; $s.Speak([Console]::In.ReadToEnd())"
)


class WindowsTTS:
    def __init__(self, volume: int = 80):
        self._volume = max(0, min(100, volume))
        self._q: queue.Queue = queue.Queue(maxsize=5)
        self._proc = None
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        threading.Thread(target=self._run, daemon=True, name="tts-worker").start()
        logger.info("Windows speech: Google TTS for Bangla (internet), Windows voice for English")

    def speak(self, text: str, *args, **kwargs) -> None:
        if not text or not text.strip():
            return
        if self._q.full():   # keep the newest lines, as the Pi engine does
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        self._idle.clear()
        self._q.put_nowait(text.strip())

    def stop_current(self) -> None:
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except OSError:
                    pass

    def set_volume(self, pct: int) -> None:
        self._volume = max(0, min(100, pct))

    def drain(self, timeout: float = 3.0) -> None:
        self._idle.wait(timeout)

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            text = self._q.get()
            try:
                stripped = text.replace(" ", "")
                if len(_BN.findall(stripped)) / max(len(stripped), 1) > 0.25:
                    self._bangla(text)
                else:
                    self._english(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Speech failed (%s): %s", exc, text[:60])
            finally:
                if self._q.empty():
                    self._idle.set()

    def _powershell(self, script: str, stdin_text: str | None = None) -> None:
        proc = subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", script],
                                stdin=subprocess.PIPE if stdin_text is not None else None,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with self._lock:
            self._proc = proc
        if stdin_text is not None:
            proc.stdin.write(stdin_text.encode("utf-8", "replace"))
            proc.stdin.close()
        proc.wait()

    def _bangla(self, text: str) -> None:
        from gtts import gTTS
        path = os.path.join(tempfile.gettempdir(), f"glass_tts_{threading.get_ident()}.mp3")
        gTTS(text=text, lang="bn").save(path)
        self._powershell(_PLAY.format(path=path.replace("\\", "/").replace("'", "''"), vol=round(self._volume / 100, 2)))

    def _english(self, text: str) -> None:
        self._powershell(_SAPI.format(vol=self._volume), stdin_text=text)
