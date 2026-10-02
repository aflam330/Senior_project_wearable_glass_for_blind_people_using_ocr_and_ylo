"""Correctly shaped text rendering (HarfBuzz + FreeType) for OCR benchmarks.

Pillow on Windows has no Raqm, so its basic layout draws Bangla wrongly: pre-base vowel signs land
after the consonant and conjuncts fall apart (for example "দেশ" is drawn as "দশে"). HarfBuzz does the
shaping that a printer or browser does; FreeType rasterises the shaped glyphs.
"""
from __future__ import annotations

from functools import lru_cache

import freetype
import numpy as np
import uharfbuzz as hb


@lru_cache(maxsize=64)
def _faces(path: str, index: int):
    blob = hb.Blob.from_file_path(path)
    hface = hb.Face(blob, index)
    ft = freetype.Face(path, index=index)
    return hface, ft


def render_mask(text: str, font_path: str, size: int, index: int = 0) -> np.ndarray:
    """Return an 8-bit coverage mask (ink = 255) of one shaped line, tightly cropped."""
    hface, ft = _faces(font_path, index)
    font = hb.Font(hface)
    font.scale = (size * 64, size * 64)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {})
    ft.set_char_size(size * 64)
    width = sum(p.x_advance for p in buf.glyph_positions) // 64 + 4 * size
    height = 4 * size
    canvas = np.zeros((height, width), np.uint8)
    pen_x, base = 2 * size * 64, int(2.6 * size)
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        ft.load_glyph(info.codepoint, freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_RENDER)
        g = ft.glyph
        bm = g.bitmap
        if bm.width and bm.rows:
            arr = np.array(bm.buffer, np.uint8).reshape(bm.rows, bm.pitch)[:, :bm.width]
            x = (pen_x + pos.x_offset) // 64 + g.bitmap_left
            y = base - (pos.y_offset // 64) - g.bitmap_top
            y0, x0 = max(y, 0), max(x, 0)
            y1, x1 = min(y + bm.rows, height), min(x + bm.width, width)
            if y1 > y0 and x1 > x0:
                sub = arr[y0 - y:y1 - y, x0 - x:x1 - x]
                canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], sub)
        pen_x += pos.x_advance
    ys, xs = np.nonzero(canvas)
    if len(ys) == 0:
        return np.zeros((size, size), np.uint8)
    return canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
