"""Field-test logger for real-time runs on the Raspberry Pi 5. Light enough to leave on.

One folder per run: logs/field/<YYYYmmdd_HHMMSS>_<host>/
  events.csv   one row per event: button presses, mode switches, every answer the glass spoke, how long it
               took, the denomination / confidence / verdict for currency, the text for OCR, errors
  system.csv   every FIELD_LOG_SYS_INTERVAL_S seconds: CPU temperature, throttling flags, CPU load,
               free memory, the app's own memory
  session.json host, start / end time, the settings that matter for the results (OCR pipeline, thresholds)
  frames/      optional (FIELD_LOG_SAVE_FRAMES=1): one small JPEG per ACTION press, named by event id,
               so a tester can check later what the glass saw. Off by default; capped by count and free disk.

Why it does not slow the Pi down:
  - the app thread only puts a tuple on a queue (a few microseconds); no file I/O, no formatting there
  - one daemon thread writes rows in batches and flushes about once a second (no fsync)
  - system readings come from /proc and /sys files (no subprocess, no psutil); `vcgencmd` is not called
  - JPEG encoding of optional frames happens in the writer thread, at 640 px, quality 80
  - if the queue ever fills (10,000 events), new events are counted as dropped instead of blocking the app

Open the CSV files in Excel, LibreOffice, Google Sheets or pandas. Summarise a run with
`python scripts/field_report.py logs/field/<run>` (also writes a labels.csv to fill in the true answers).
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import platform
import queue
import shutil
import threading
import time
from typing import Any, Optional

EVENT_COLUMNS = ["event_id", "time", "t_s", "event", "mode", "latency_ms", "result", "denomination", "confidence",
                 "verdict", "score", "lang", "detail"]
SYSTEM_COLUMNS = ["time", "t_s", "cpu_temp_c", "throttled_hex", "load_1m", "cpu_busy_pct", "mem_available_mb",
                  "app_rss_mb", "events_logged", "events_dropped"]


def _read(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


class FieldLogger:
    def __init__(self, root: str, enabled: bool = True, sys_interval_s: float = 10.0, save_frames: bool = False,
                 max_frames: int = 2000, min_free_mb: int = 500, settings: Optional[dict] = None):
        self.enabled = enabled
        self.t0 = time.monotonic()
        self._q: "queue.Queue" = queue.Queue(maxsize=10_000)
        self._id = 0
        self._id_lock = threading.Lock()
        self.dropped = 0
        self.logged = 0
        self.save_frames = save_frames
        self.max_frames = max_frames
        self.min_free_mb = min_free_mb
        self.frames_saved = 0
        self._stop = threading.Event()
        self._cpu_prev = None
        if not enabled:
            self.dir = None
            return
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(root, f"{stamp}_{platform.node() or 'host'}")
        os.makedirs(os.path.join(self.dir, "frames") if save_frames else self.dir, exist_ok=True)
        self._session = {"start": dt.datetime.now().isoformat(timespec="seconds"), "host": platform.node(),
                         "machine": platform.machine(), "python": platform.python_version(),
                         "device_model": _read("/proc/device-tree/model"), "settings": settings or {}}
        self._write_session()
        self._ev_file = open(os.path.join(self.dir, "events.csv"), "w", newline="", encoding="utf-8-sig")
        self._ev = csv.writer(self._ev_file)
        self._ev.writerow(EVENT_COLUMNS)
        self._sys_file = open(os.path.join(self.dir, "system.csv"), "w", newline="", encoding="utf-8-sig")
        self._sys = csv.writer(self._sys_file)
        self._sys.writerow(SYSTEM_COLUMNS)
        self._writer = threading.Thread(target=self._write_loop, daemon=True, name="field-log-writer")
        self._writer.start()
        self._sampler = threading.Thread(target=self._sample_loop, args=(sys_interval_s,), daemon=True, name="field-log-sys")
        self._sampler.start()

    # ---- called from the app (cheap) ----
    def log(self, event: str, mode: str = "", latency_ms: Optional[float] = None, result: str = "",
            frame=None, **fields: Any) -> int:
        """Queue one event row; returns its event id (0 when disabled). Never blocks, never raises."""
        if not self.enabled:
            return 0
        with self._id_lock:
            self._id += 1
            eid = self._id
        try:
            self._q.put_nowait(("ev", eid, time.monotonic(), dt.datetime.now(), event, mode, latency_ms, result, fields,
                                frame if self.save_frames else None))
        except queue.Full:
            self.dropped += 1
        return eid

    def close(self) -> None:
        if not self.enabled or self._stop.is_set():
            return
        self.log("session_end")
        self._stop.set()
        self._writer.join(timeout=5)
        self._session.update({"end": dt.datetime.now().isoformat(timespec="seconds"), "events_logged": self.logged,
                              "events_dropped": self.dropped, "frames_saved": self.frames_saved})
        self._write_session()
        for f in (self._ev_file, self._sys_file):
            try:
                f.close()
            except OSError:
                pass

    # ---- background threads ----
    def _write_session(self):
        with open(os.path.join(self.dir, "session.json"), "w", encoding="utf-8") as f:
            json.dump(self._session, f, ensure_ascii=False, indent=1)

    def _write_loop(self):
        last_flush = time.monotonic()
        while True:
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                item = None
            if item is not None:
                self._write_item(item)
                while True:  # drain what is waiting, then flush once
                    try:
                        self._write_item(self._q.get_nowait())
                    except queue.Empty:
                        break
            now = time.monotonic()
            if now - last_flush >= 1.0 or self._stop.is_set():
                try:
                    self._ev_file.flush()
                    self._sys_file.flush()
                except (OSError, ValueError):
                    pass
                last_flush = now
            if self._stop.is_set() and self._q.empty():
                return

    def _write_item(self, item):
        kind = item[0]
        try:
            if kind == "ev":
                _, eid, mono, wall, event, mode, lat, result, fields, frame = item
                f = dict(fields)
                row = [eid, wall.isoformat(timespec="milliseconds"), round(mono - self.t0, 3), event, mode,
                       "" if lat is None else round(lat, 1), str(result).replace("\n", " ")[:500],
                       f.pop("denomination", ""), _fmt(f.pop("confidence", "")), f.pop("verdict", ""), _fmt(f.pop("score", "")),
                       f.pop("lang", ""), json.dumps(f, ensure_ascii=False, default=str) if f else ""]
                self._ev.writerow(row)
                self.logged += 1
                if frame is not None:
                    self._save_frame(eid, frame)
            elif kind == "sys":
                self._sys.writerow(item[1])
        except Exception:  # a bad row must never stop logging
            self.dropped += 1

    def _save_frame(self, eid, frame):
        if self.frames_saved >= self.max_frames:
            return
        try:
            if shutil.disk_usage(self.dir).free < self.min_free_mb * 1024 * 1024:
                return
            import cv2
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(self.dir, "frames", f"{eid:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.frames_saved += 1
        except Exception:
            pass

    def _sample_loop(self, interval):
        while not self._stop.wait(interval):
            try:
                self._q.put_nowait(("sys", self._system_row()))
            except queue.Full:
                self.dropped += 1

    def _system_row(self):
        temp = _read("/sys/class/thermal/thermal_zone0/temp")
        thr = _read("/sys/devices/platform/soc/soc:firmware/get_throttled")
        load = _read("/proc/loadavg")
        mem = _read("/proc/meminfo") or ""
        avail = next((int(l.split()[1]) // 1024 for l in mem.splitlines() if l.startswith("MemAvailable:")), "")
        statm = _read("/proc/self/statm")
        rss = round(int(statm.split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6, 1) if statm and hasattr(os, "sysconf") else ""
        return [dt.datetime.now().isoformat(timespec="seconds"), round(time.monotonic() - self.t0, 1),
                round(int(temp) / 1000, 1) if temp and temp.isdigit() else "",
                f"0x{int(thr, 16):x}" if thr and _hexlike(thr) else (thr or ""),
                load.split()[0] if load else "", self._cpu_busy(), avail, rss, self.logged, self.dropped]

    def _cpu_busy(self):
        stat = _read("/proc/stat")
        if not stat:
            return ""
        v = [int(x) for x in stat.splitlines()[0].split()[1:]]
        idle, total = v[3] + (v[4] if len(v) > 4 else 0), sum(v)
        prev, self._cpu_prev = self._cpu_prev, (idle, total)
        if not prev or total == prev[1]:
            return ""
        return round(100 * (1 - (idle - prev[0]) / (total - prev[1])), 1)


def _hexlike(s):
    try:
        int(s, 16)
        return True
    except ValueError:
        return False


def _fmt(x):
    return round(x, 4) if isinstance(x, float) else x


_LOGGER: Optional[FieldLogger] = None


def get() -> FieldLogger:
    """The app-wide logger, created on first use from config (disabled logger if FIELD_LOG_ENABLED is off)."""
    global _LOGGER
    if _LOGGER is None:
        import config
        settings = {k: getattr(config, k) for k in ("OCR_PIPELINE", "OCR_CANVAS_SIZE", "OCR_V2_MIN_CONF", "CURRENCY_ANNOUNCE_CONF",
                                                   "JAAL_SAFE_POLICY_ENABLED", "JAAL_SAFE_TAU", "WATERMARK_CHECK_ENABLED",
                                                   "CAPTURE_GUIDE_ENABLED", "STUDY_CONDITION", "ESPEAK_SPEED", "ON_RASPBERRY_PI_5")
                    if hasattr(config, k)}
        _LOGGER = FieldLogger(getattr(config, "FIELD_LOG_DIR", os.path.join(config.BASE_DIR, "logs", "field")),
                              enabled=getattr(config, "FIELD_LOG_ENABLED", True),
                              sys_interval_s=getattr(config, "FIELD_LOG_SYS_INTERVAL_S", 10.0),
                              save_frames=getattr(config, "FIELD_LOG_SAVE_FRAMES", False), settings=settings)
        import atexit
        atexit.register(_LOGGER.close)  # normal exit closes the run; after a power cut the rows up to the last second are on disk
    return _LOGGER
