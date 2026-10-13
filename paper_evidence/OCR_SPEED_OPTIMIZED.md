# Text reading speed: second pass (2026-10-13 evening)

First pass and the main result: `OCR_SPEED_IMPROVED.md` (text detector through ONNX Runtime: 5.55 → 2.63 s, 12-line page 9.99 → 5.59 s on four laptop CPU threads, same validation error; now the default).

**The target was 5–8 s on the laptop. The 12-line page is at 5.6 s with the default settings**, so the laptop target is met by the first pass. Nothing below changes that figure. No Raspberry Pi measurement exists for any of it.

## What the ten requested approaches came to

| Approach | Status | Result |
|---|---|---|
| A. Canvas size 320 / 480 / 640 / 2560 | Measured earlier (2 October, on a selection set) | Raw CER 13.29 % at 2560, 13.84 % at 640 (+0.55), 15.37 % at 480, 19.15 % at 320. On the Pi, 640 took 6.9 s against 20.2 s on one frame. 640 fails the 0.5-point rule by 0.05 and is opt-in (`OCR_CANVAS_SIZE=640`) |
| B. INT8 detector | Measured, three calibrations | MinMax on 64 images: +0.65 points (page 4.28 s). MinMax on 128 images: validation CER 3.15 % against 2.58 %, +0.57 points. Percentile and entropy calibration ran out of memory on this 16 GB laptop; a Conv-only variant was stopped at 12 GB. **No INT8 version is within the rule; full precision stays the default.** INT8 is opt-in (`OCR_DETECTOR=onnx_int8`) |
| B. FP16 | Not applicable | Half precision gives no speed-up on these CPUs |
| C. Text-region detection | Done in the first pass | The ONNX detector is the optimisation of that step |
| D. Parallel processing | Done | ONNX Runtime uses all cores; line-by-line reading overlaps recognition with speech |
| E. Precomputed common words | Not done | The recogniser is not the slow part; a word cache cannot skip the detector |
| F. Batch processing | Measured on 11 October | Recognition batch 16: 8.85 s against 8.92 s per page, no gain |
| G. Pruning; H. distillation | Not done | Both need retraining the detector and a new accuracy study; INT8, the cheaper compression, already fails the accuracy rule |
| I. Lower picture resolution | Measured on 11 October | Camera's own size: 5.3 s against 8.9 s per page, +0.74 points. Rejected |
| J. Skip detection for small crops | Not applicable | The glass reads camera frames, not crops |

## What is ready for the Pi

- `models/craft_fp32.onnx` is in both repositories and is used automatically.
- On the Pi, run `python scripts/benchmark_pi5.py --iters 100 --sustained 30`, then the same with `OCR_DETECTOR=onnx_int8`, then with `OCR_CANVAS_SIZE=640`. Those three runs decide the Pi setting; nothing on the laptop can.
