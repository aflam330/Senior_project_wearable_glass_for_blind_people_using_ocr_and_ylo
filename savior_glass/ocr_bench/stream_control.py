"""Control for page_bench.stream: the same validation scoring through OCRMode.process_frame, with line-by-line
page reading switched off, and a count of validation images that the streaming path would take."""
from __future__ import annotations

import json

from . import bench as B
from .page_bench import OUT, _ocr_mode


def main(threads=4):
    import torch
    torch.set_num_threads(threads)
    import config
    m = _ocr_mode()
    config.OCR_STREAM_PAGES = False

    def read(img):
        m.process_frame(img)
        return m._stored_text or ""
    off = B.score(read, "val")

    config.OCR_STREAM_PAGES = True
    m.speak_line = lambda t: None
    taken = {"n": 0, "total": 0}

    def read_on(img):
        m.process_frame(img)
        taken["total"] += 1
        taken["n"] += int(m.streamed)
        return m._stored_text or ""
    on = B.score(read_on, "val")
    res = {"stream_off": {"val_cer": off["overall"]["cer"], "bn": off["bn"]["cer"], "en": off["en"]["cer"]},
           "stream_on": {"val_cer": on["overall"]["cer"], "bn": on["bn"]["cer"], "en": on["en"]["cer"]},
           "val_images_streamed": taken}
    print(json.dumps(res, indent=1))
    (OUT / "page_streaming_control.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
