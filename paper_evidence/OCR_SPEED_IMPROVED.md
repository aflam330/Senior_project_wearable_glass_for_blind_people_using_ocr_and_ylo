# Text reading speed: what was gained, and what is still unmeasured (2026-10-13)

Sources: `savior_glass/ocr_bench/detector_onnx.py` → `results/ocr_bench/detector_onnx.json`; `ocr_bench/page_bench.py` → `page_speed_compare.json`, `page_streaming.json`.

**The target was 3–5 s on the Raspberry Pi 5. No Pi measurement was made in this pass, so that target is neither met nor missed.** Everything below was measured on the development laptop's CPU limited to four threads, the number of cores of the Pi. The Pi is several times slower per core; only ratios can be expected to carry over, and that is an expectation.

## 1. The text detector through ONNX Runtime

On the Pi, the time of a text capture is in the text detector (CRAFT), not in the recogniser. The detector was exported to ONNX and run with ONNX Runtime; an INT8 version was made by static quantisation, calibrated on the "lexicon" and "select" phrase sets (not validation, not test).

| Detector | Detector alone, 1,200 × 900 | 12-line page | One picture (median of 432) | Validation CER | Bangla / English | File |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch (before) | 5.55 s | 9.99 s | 2.30 s | 2.14 % | 1.59 / 2.69 % | — |
| **ONNX Runtime, full precision (now the default)** | **2.63 s** | **5.59 s** | **1.22 s** | **2.14 %** | 1.59 / 2.69 % | 83 MB |
| ONNX Runtime, INT8 (opt-in) | 1.29 s | 4.28 s | 0.78 s | 2.79 % | 2.53 / 3.04 % | 21 MB |

- **Full precision: 2.1 times faster detector, 1.8 times faster page, identical error.** Its outputs differ from PyTorch's by about 10⁻⁷. It is now used whenever `models/craft_fp32.onnx` is present, with PyTorch as the fallback (`ocr_detector_onnx.py`, `config.OCR_DETECTOR`).
- **INT8: 4.3 times faster detector, but +0.65 points of validation error.** The project's rule admits a faster setting only within 0.5 points, so INT8 is not the default. It can be switched on with `OCR_DETECTOR=onnx_int8`.
- The laptop was slower in this run than on 11 October (PyTorch page 9.99 s here, 6.5 s then), so the three rows are comparable with each other, not with the earlier table.

## 2. Earlier in this pass (11 October)

- Greedy decoder instead of beam search: 6.5 s against 8.9 s per 12-line page, validation CER 2.14 % against 2.18 %. Adopted.
- Pages of four or more lines are spoken line by line: the first line is ready after about 5.5 s; validation CER unchanged.
- Reading at the camera's own size: 5.3 s, but +0.74 points. Rejected.

## 3. What the Pi figure is likely to be, and why this is not a result

The only Pi measurement is 18.7 s per capture with the PyTorch detector and beam search, before any of these changes, and 6.9 s against 20.2 s on one frame with a detector canvas of 640 (+0.55 points raw CER, opt-in). If the laptop ratio held, the default would be near 9–10 s and the INT8 detector near 5–6 s. These are projections and must not be quoted as measurements.

**To measure:** on the Pi, `python scripts/benchmark_pi5.py --iters 100 --sustained 30`, once as it is and once with `OCR_DETECTOR=onnx_int8`.

## 4. Not tried

- Quantising the recogniser beyond what EasyOCR already does on the CPU.
- A separate small text-region detector in front of CRAFT.
- Caching of common words. The recogniser is not the slow part.
