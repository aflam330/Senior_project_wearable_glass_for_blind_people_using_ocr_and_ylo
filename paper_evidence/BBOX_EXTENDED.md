# Multi-note boxes

Live camera frames were not captured. The numbers below are composites: crops from synthetic test images with exact boxes, pasted without overlap onto a blank canvas. Weights: `models/best.pt`. Confidence 0.25. Twelve images at each count. Source: `results/bbox/bbox_multinote.json`.

| notes per image | notes | detected at IoU ≥ 0.5 | mean IoU |
| ---: | ---: | ---: | ---: |
| 2 | 24 | 1.0 | 0.9457147895426855 |
| 3 | 36 | 1.0 | 0.9355042416533456 |
| 5 | 60 | 1.0 | 0.9365677215828507 |

Every pasted note was matched at IoU at least 0.5. This is not a phone photo of several notes on a table. That setting remains NOT_MEASURED.
