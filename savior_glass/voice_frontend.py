"""Audio front end for the voice commands: noise reduction and level control, block by block, numpy only.

Both work on the 16 kHz microphone stream in the blocks the listener receives, keep their state between blocks,
and add no delay beyond one 32 ms frame. Which of them the glass uses is decided by scripts/voice_eval.py on the
design voices only (config.VOICE_CMD_FRONTEND).

  Denoiser   short-time spectral gain (decision-directed Wiener filter, Ephraim-Malah style) with a noise spectrum
             tracked by minimum statistics over the last ~1.5 s, a gain floor so speech is not carved away, and
             50 % overlap-add with a square-root Hann window.
  Level      automatic gain: brings speech-level blocks to a target level, at most +30 dB (for soft or whispered
             speech), attack fast and release slow, never amplifying blocks quieter than the tracked noise floor.
"""
from __future__ import annotations

import numpy as np

N = 512            # 32 ms frames at 16 kHz
H = N // 2
WIN = np.sqrt(np.hanning(N + 1)[:N]).astype(np.float32)


class Denoiser:
    def __init__(self, floor_db: float = -15.0, alpha: float = 0.96, history_frames: int = 47):
        self.floor = 10 ** (floor_db / 20)
        self.alpha = alpha
        self.hist = history_frames                      # ~1.5 s of frames for the minimum statistics
        self.buf = np.zeros(0, np.float32)
        self.out_tail = np.zeros(H, np.float32)
        self.prev = np.zeros(N, np.float32)
        self.powers: list = []
        self.noise = None
        self.prev_clean = None

    def process(self, x: np.ndarray) -> np.ndarray:
        """float32 samples in [-1, 1] -> denoised samples, same count, delayed by H samples."""
        self.buf = np.concatenate([self.buf, x.astype(np.float32)])
        out = []
        while len(self.buf) >= H:
            frame = np.concatenate([self.prev[H:], self.buf[:H]])
            self.prev = frame
            self.buf = self.buf[H:]
            spec = np.fft.rfft(frame * WIN)
            p = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
            self.powers.append(p)
            if len(self.powers) > self.hist:
                self.powers.pop(0)
            smooth = np.mean(self.powers[-3:], axis=0)
            mn = np.min(np.stack(self.powers), axis=0) * 1.5 if len(self.powers) >= 8 else smooth
            self.noise = mn if self.noise is None else np.minimum(np.maximum(mn, 1e-10), 4 * smooth + 1e-10)
            post = p / (self.noise + 1e-10)
            prior = np.maximum(post - 1, 0)
            if self.prev_clean is not None:
                prior = self.alpha * self.prev_clean / (self.noise + 1e-10) + (1 - self.alpha) * prior
            g = np.maximum(prior / (1 + prior), self.floor)
            self.prev_clean = (g ** 2) * p
            y = np.fft.irfft(spec * g, N).astype(np.float32) * WIN
            out.append(self.out_tail + y[:H])
            self.out_tail = y[H:]
        return np.concatenate(out) if out else np.zeros(0, np.float32)


class Level:
    def __init__(self, target_rms: float = 0.08, max_gain_db: float = 30.0):
        self.target = target_rms
        self.max_gain = 10 ** (max_gain_db / 20)
        self.gain = 1.0
        self.noise = None

    def process(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return x
        out = np.empty_like(x)
        for i in range(0, len(x), 160):                 # 10 ms steps
            b = x[i:i + 160]
            r = float(np.sqrt(np.mean(b * b)) + 1e-9)
            # noise floor: follows quiet blocks at once, rises slowly (about 22 % a second) through speech
            self.noise = r if self.noise is None or r < self.noise else self.noise * 1.002
            want = self.gain
            if r > 2.0 * self.noise:                     # louder than the noise floor: speech, steer the gain
                want = min(self.max_gain, self.target / r)
            k = 0.5 if want < self.gain else 0.02        # fast down (no clipping), slow up
            self.gain += k * (want - self.gain)
            out[i:i + 160] = np.clip(b * self.gain, -1, 1)
        return out


class FrontEnd:
    """kind: "none", "denoise", "level", or "denoise+level". Takes and returns int16 PCM bytes."""

    def __init__(self, kind: str = "none"):
        self.kind = kind
        self.dn = Denoiser() if "denoise" in kind else None
        self.lv = Level() if "level" in kind else None

    def __call__(self, pcm: bytes) -> bytes:
        if self.kind == "none":
            return pcm
        x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
        if self.dn is not None:
            x = self.dn.process(x)
        if self.lv is not None:
            x = self.lv.process(x)
        return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()


class VadEndpointer:
    """Silero VAD (ONNX) on 32 ms chunks: says when an utterance has ended, so the recogniser can decide at once
    instead of waiting for its own end-of-speech timeout. Speech = probability >= on for 3 chunks (~100 ms); end =
    probability < off for `end_chunks` chunks after speech (8 chunks ~ 256 ms)."""

    _model = None

    def __init__(self, on: float = 0.5, off: float = 0.35, end_chunks: int = 8):
        if VadEndpointer._model is None:
            from silero_vad import load_silero_vad
            VadEndpointer._model = load_silero_vad(onnx=True)
        self.on, self.off, self.end_chunks = on, off, end_chunks
        self.reset()

    def reset(self) -> None:
        self.buf = np.zeros(0, np.float32)
        self.speech_run = self.quiet_run = 0
        self.in_speech = False
        try:
            VadEndpointer._model.reset_states()
        except Exception:  # noqa: BLE001
            pass

    def ended(self, pcm: bytes) -> bool:
        """Feed int16 PCM; True once when an utterance has just ended."""
        import torch
        self.buf = np.concatenate([self.buf, np.frombuffer(pcm, np.int16).astype(np.float32) / 32768])
        hit = False
        while len(self.buf) >= 512:
            prob = float(VadEndpointer._model(torch.from_numpy(self.buf[:512].copy()), 16000))
            self.buf = self.buf[512:]
            if prob >= self.on:
                self.speech_run += 1
                self.quiet_run = 0
                if self.speech_run >= 3:
                    self.in_speech = True
            elif prob < self.off:
                self.speech_run = 0
                if self.in_speech:
                    self.quiet_run += 1
                    if self.quiet_run >= self.end_chunks:
                        self.in_speech, self.quiet_run, hit = False, 0, True
        return hit
