"""Voice commands: run the glass by speaking instead of pressing buttons. Offline (Vosk, small English model).

The recogniser is given a closed list of command phrases, so it can only answer with one of them or with "unknown";
it is not open dictation. Commands are English words because no offline Bangla recogniser of this size exists.

  say                                          what happens
  "read text" / "read" / "text"                text mode, and one capture is read
  "money" / "check money" / "taka"             currency mode, and one note is checked
  "objects" / "find objects" / "around me"     object mode (it announces by itself)
  "describe" / "describe scene"                online description mode, and one picture is described
  "emotion" / "face"                           emotion mode, and one face is read
  "next mode"                                  the MODE button
  "capture" / "scan" / "check"                 the ACTION button, in the current mode
  "repeat" / "say again"                       the READ button (last text again)
  "louder" / "volume up", "quieter" / "volume down"
  "stop" / "quiet"                             stops the voice

The microphone is ignored while the glass itself is speaking, so it cannot command itself. Every command starts with
the wake word "glass" ("glass, read text"); VOICE_WAKE_WORD= (empty) turns that off. Chosen on design voices and
measured once on held-out voices: paper_evidence/VOICE_COMMANDS_FAST.md.

Needs: pip install vosk; python scripts/get_speech_models.py; a microphone (USB or headset), read through
sounddevice when it is installed and through ALSA's arecord otherwise.
Test without a microphone: python scripts/test_voice_commands.py
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time

import config

logger = logging.getLogger("smart_glass.voice_commands")

# intent -> phrases. Every phrase is made of words the small Vosk model knows.
PHRASES = {
    "mode:ocr": ["read text", "read", "text", "read this", "reading"],
    "mode:currency": ["money", "check money", "taka", "currency", "bank note", "check note"],
    "mode:object": ["objects", "object", "find objects", "around me", "what is around"],
    "mode:claude": ["describe", "describe scene", "describe this"],
    "mode:emotion": ["emotion", "face", "read face"],
    "next_mode": ["next mode", "change mode"],
    "action": ["capture", "scan", "check", "take picture"],
    "read": ["repeat", "say again", "again"],
    "vol_up": ["louder", "volume up"],
    "vol_down": ["quieter", "volume down"],
    "stop": ["stop", "quiet", "be quiet", "silence"],
}
MODE_INDEX = {"mode:ocr": config.MODE_OCR, "mode:currency": config.MODE_CURRENCY, "mode:object": config.MODE_OBJECT,
              "mode:claude": config.MODE_CLAUDE, "mode:emotion": config.MODE_EMOTION}
_BY_PHRASE = {p: intent for intent, ps in PHRASES.items() for p in ps}
# phrases that are the start of a longer phrase ("read" of "read text"): never decided early
_PREFIXES = {p for p in _BY_PHRASE for q in _BY_PHRASE if q != p and q.startswith(p + " ")}


def parse(text: str, wake: str = "", strip_edges: bool | None = None) -> str | None:
    """Intent for what the recogniser returned, or None. The whole utterance has to be one command phrase: a command
    word inside other speech ("I will read it later") is not a command.

    strip_edges (wake-word mode only): unknown words before the wake word and after the command are dropped, because
    with people talking nearby their speech is often glued to the edges of the user's command
    ("[unk] glass read text [unk]"). Without a wake word this is never done: "[unk] stop [unk]" is too often a
    sentence that merely contains "stop"."""
    if strip_edges is None:
        strip_edges = getattr(config, "VOICE_CMD_STRIP_EDGES", False)
    words = [w for w in (text or "").lower().split()]
    if wake and strip_edges:
        while words and words[0] == "[unk]":
            words = words[1:]
        while words and words[-1] == "[unk]":
            words = words[:-1]
    if not words or "[unk]" in words:
        return None
    if wake:
        if words[0] != wake:
            return None
        words = words[1:]
    return _BY_PHRASE.get(" ".join(words))


def available() -> str:
    """'' when voice commands can run, otherwise the reason they cannot."""
    try:
        import vosk  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"vosk not installed ({type(exc).__name__})"
    if not os.path.isdir(config.VOICE_CMD_MODEL):
        return "recogniser model missing (python scripts/get_speech_models.py)"
    return ""


class CommandRecognizer:
    """Vosk with the command phrases as its grammar. feed() takes 16-bit mono PCM and returns the commands heard."""

    def __init__(self, rate: int = 16000, wake: str | None = None, min_conf: float | None = None,
                 early: bool | None = None, frontend: str | None = None):
        import vosk
        vosk.SetLogLevel(-1)
        self.wake = (config.VOICE_CMD_WAKE if wake is None else wake).strip().lower()
        self.min_conf = config.VOICE_CMD_MIN_CONF if min_conf is None else min_conf
        # Early decision: act on a partial result as soon as it is a whole command phrase that cannot grow into a
        # longer one ("stop", "check money"), instead of waiting for Vosk to decide that speech has ended.
        self.early = getattr(config, "VOICE_CMD_EARLY", False) if early is None else early
        from voice_frontend import FrontEnd
        self.frontend = FrontEnd(getattr(config, "VOICE_CMD_FRONTEND", "none") if frontend is None else frontend)
        self._last_partial = ""
        self._fired_partial = False
        # End of speech: "vosk" = the recogniser's own timeout; "vad" = Silero voice-activity detection (faster)
        self.endpoint = getattr(config, "VOICE_CMD_ENDPOINT", "vosk")
        self.vad = None
        if self.endpoint == "vad":
            from voice_frontend import VadEndpointer
            self.vad = VadEndpointer(end_chunks=getattr(config, "VOICE_CMD_VAD_END_CHUNKS", 8))
        self.rate = rate
        self._model = _model()
        phrases = sorted(_BY_PHRASE)
        if self.wake:
            phrases = [f"{self.wake} {p}" for p in phrases]
        self._rec = vosk.KaldiRecognizer(self._model, rate, json.dumps(phrases + ["[unk]"]))
        self._rec.SetWords(True)
        self._rec.SetPartialWords(True)
        self.last_heard = ""

    def reset(self) -> None:
        self._rec.Reset()
        self._last_partial, self._fired_partial = "", False
        if self.vad is not None:
            self.vad.reset()

    def feed(self, pcm: bytes) -> list:
        """Commands completed by this piece of audio (usually none or one)."""
        pcm = self.frontend(pcm)
        final = self._rec.AcceptWaveform(pcm)
        if not final and self.vad is not None and self.vad.ended(pcm):
            res = json.loads(self._rec.FinalResult())          # speech has ended: decide now
            fired, self._fired_partial, self._last_partial = self._fired_partial, False, ""
            return [] if fired else [c for c in [self._intent(res)] if c]
        if final:
            res = json.loads(self._rec.Result())
            fired, self._fired_partial, self._last_partial = self._fired_partial, False, ""
            if fired:                                   # this utterance was already acted on from its partial result
                return []
            return [c for c in [self._intent(res)] if c]
        if not self.early or self._fired_partial:
            return []
        part = json.loads(self._rec.PartialResult())
        text = part.get("partial", "")
        stable, self._last_partial = text and text == self._last_partial, text
        if not stable:
            return []
        intent = parse(text, self.wake)
        words = text.split()[1:] if self.wake else text.split()
        if intent is None or " ".join(words) in _PREFIXES:
            return []
        confs = [w.get("conf", 0.0) for w in part.get("partial_result", [])]
        if confs and min(confs) < self.min_conf:
            return []
        self.last_heard = text
        self._fired_partial = True
        return [intent]

    def flush(self) -> list:
        return [c for c in [self._intent(json.loads(self._rec.FinalResult()))] if c]

    def _intent(self, res: dict) -> str | None:
        text = res.get("text", "")
        if not text:
            return None
        self.last_heard = text
        intent = parse(text, self.wake)
        confs = [w.get("conf", 0.0) for w in res.get("result", []) if w.get("word") != "[unk]"]
        if intent and confs and min(confs) < self.min_conf:      # a mumbled or far-away word: better not to act
            logger.debug("voice: heard %r but not sure enough (%.2f)", text, min(confs))
            return None
        return intent


_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        import vosk
        _MODEL = vosk.Model(config.VOICE_CMD_MODEL)
    return _MODEL


class VoiceCommands:
    """Listens on the microphone in its own thread and calls the application's own button handlers."""

    def __init__(self, app) -> None:
        self.app = app
        self.status = "voice commands: off"
        self.last_command = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="voice-commands")

    def start(self) -> bool:
        why = available()
        if why:
            self.status = "voice commands: off, " + why
            logger.info(self.status)
            return False
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    # -- microphone ----------------------------------------------------

    def _open(self, sd, q):
        """An input stream at 16 kHz, or at the microphone's own rate when it refuses 16 kHz."""
        want = config.VOICE_CMD_MIC.strip().lower()
        dev = None
        if want:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0 and want in d["name"].lower():
                    dev = i
                    break
        info = sd.query_devices(dev, kind="input")
        last = None
        for rate in (16000, int(info["default_samplerate"])):
            try:
                stream = sd.RawInputStream(device=dev, samplerate=rate, channels=1, dtype="int16", blocksize=rate // 5,
                                           callback=lambda data, frames, t, status: q.put(bytes(data)))
                stream.start()
                return stream, rate, info["name"]
            except Exception as exc:  # noqa: BLE001
                last = exc
        raise RuntimeError(f"microphone cannot be opened: {last}")

    def _open_arecord(self, q):
        """Microphone through ALSA's own recorder, for a Pi without PortAudio (sounddevice needs libportaudio2)."""
        import subprocess
        cmd = ["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", "16000", "-c", "1"]
        if config.VOICE_CMD_MIC.strip():
            cmd += ["-D", config.VOICE_CMD_MIC.strip()]          # here an ALSA device name, e.g. plughw:2,0
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        def pump():
            while True:
                data = proc.stdout.read(6400)                    # 0.2 s
                if not data:
                    break
                q.put(data)
        threading.Thread(target=pump, daemon=True, name="voice-arecord").start()
        time.sleep(0.5)
        if proc.poll() is not None:
            raise RuntimeError("arecord could not open a microphone")

        class Stream:
            def stop(self): proc.terminate()
            def close(self): pass
        return Stream(), 16000, "arecord"

    def _run(self) -> None:
        try:
            q: queue.Queue = queue.Queue()
            try:
                import sounddevice as sd
                stream, rate, name = self._open(sd, q)
            except Exception as exc:  # noqa: BLE001
                logger.info("sounddevice microphone not usable (%s); trying arecord", exc)
                stream, rate, name = self._open_arecord(q)
            rec = CommandRecognizer(rate)
        except Exception as exc:  # noqa: BLE001
            self.status = f"voice commands: off, {exc}"
            logger.warning(self.status)
            return
        self.status = f"voice commands: listening ({name}, {rate} Hz)"
        logger.info(self.status)
        deaf_until = 0.0
        try:
            while not self._stop.is_set():
                try:
                    pcm = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if getattr(self.app._tts, "speaking", False):          # the glass is talking: do not listen to it
                    deaf_until = time.time() + config.VOICE_CMD_DEAF_AFTER_SPEECH_S
                if time.time() < deaf_until:
                    rec.reset()
                    continue
                for intent in rec.feed(pcm):
                    self.last_command = intent
                    logger.info("voice command: %s (heard %r)", intent, rec.last_heard)
                    try:
                        dispatch(self.app, intent)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("voice command %s failed: %s", intent, exc)
        finally:
            stream.stop()
            stream.close()


def dispatch(app, intent: str) -> None:
    """Carry out one command through the same handlers the buttons use."""
    try:
        import field_log
        field_log.get().log("voice_command", "", result=intent)
    except Exception:  # noqa: BLE001
        pass
    if intent in MODE_INDEX:
        idx = MODE_INDEX[intent]
        changed = app.set_mode(idx)
        if idx != config.MODE_OBJECT:            # object mode announces by itself; the others need one capture
            if changed:
                time.sleep(config.VOICE_CMD_SETTLE_S)   # the mode name is being spoken; the capture follows it
            app._on_action_press()
    elif intent == "next_mode":
        app._on_mode_press()
    elif intent == "action":
        app._on_action_press()
    elif intent == "read":
        app._on_read_press()
    elif intent == "vol_up":
        app._on_vol_up()
    elif intent == "vol_down":
        app._on_vol_down()
    elif intent == "stop":
        app._tts.stop_current()
