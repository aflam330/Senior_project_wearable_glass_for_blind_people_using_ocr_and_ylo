# Bounding boxes: detector and app (2026-09-28)

Scripts: `realtime_bangla_taka_detection/scripts/eval/eval_bbox.py` (detector) and
`eval_bbox_app.py` (the box the glass app uses). Raw numbers: `realtime_bangla_taka_detection/results/bbox/`.
The top-confidence box (conf ≥ 0.25) is compared with the label; IoU is intersection over union.

## Detector (YOLOv8s `best.pt`)

| Set | Notes | Detected | IoU mean / median | IoU ≥ 0.5 | ≥ 0.75 | ≥ 0.9 | Class right | Box size vs label | Edge error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Synthetic test (exact labels) | 2052 | 100% | 0.908 / 0.925 | 100% | 98.4% | 64.9% | 99.7% | +3.8% area | 2.0% of box |
| Real photos, heuristic labels | 40 | 97.5% | 0.864 / 0.938 | 95.0% | 92.5% | 75.0% | 97.4% | +5.7% | 1.6% |
| Real close-ups (note fills photo) | 473 | 98.7% | box covers the photo: IoU 0.99 median, all ≥ 0.8 | | | | | | |

- **False alarms:** zero boxes on the 230 synthetic test images that contain no note, at confidence 0.25, 0.35 or 0.5.
- **INT8 model:** the same box quality on 553 synthetic notes (IoU 0.910, 100% ≥ 0.5).
- **The real-photo labels come from a background-difference heuristic**, not from people. Of the 513 real test photos, 473 labels fell back to the full frame; those photos are close-ups. The worst real IoUs, checked by eye in `results/bbox/worst_real_boxes.jpg`, are label errors: the heuristic box covers only part of the note (once just the "50" corner), while the model's box encloses the whole note. One over-exposed 20-taka photo is a real miss.

## App path (`CurrencyMode.detect_live`)

**Bug fixed:** frames with mean brightness < 80 were always brightened before detection. On dark validation images that lost 9 of 197 notes and loosened the boxes (IoU 0.860 vs 0.908 without brightening). The rule was chosen on validation: detect on the frame as it is, and brighten only if nothing is found. On validation that loses no note.

| Synthetic test | App before fix | App after fix | Raw detector |
|---|---:|---:|---:|
| Dark images (183): IoU mean, found | 0.881, 98.4% | **0.909, 100%** | 0.909, 100% |
| Other images (1869): IoU mean, found | 0.908, 100% | 0.908, 100% | 0.908, 100% |

Consistency checks:
- The box the app returns is `(x, y, w, h)` in frame pixels.
- The crop passed to the jaal check equals the box clipped to the frame in 2052/2052 cases.
- The Windows overlay draws `(x, y)`–`(x + w, y + h)`.
- The note-pose estimate converts to `(x1, y1, x2, y2)`.
- The colour fallback and face boxes use the same `(x, y, w, h)` format.

## Check 2026-09-29

The tables above still match `results/bbox/bbox_eval.json` and `bbox_app_eval.json`. Composites with several pasted notes are in `BBOX_EXTENDED.md`. A live camera box score was not taken.
