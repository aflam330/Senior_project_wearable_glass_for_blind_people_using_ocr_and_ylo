# Paper record, 2026-09-29

This file cites artifacts. It does not replace the paper source. Rounded headlines in older notes are the exact fractions below.

## Authenticity

PRMVT prefix fine-tune, test n=208: 1-view 0.9711538461538461, 6-view 0.9759615384615384.

VCIE selected on validation 1-view: 1-view and 6-view 0.9278846153846154. Above the CNN+ViT 6-view baseline 0.9182692307692307.

MTPT with the 6-then-3 epoch schedule, selected on validation 1-view: 1-view 0.9807692307692307, 6-view 0.9663461538461539.

The frozen side tower matches `her_base`: 1-view 0.9903846153846154, 6-view 0.9182692307692307.

## Occlusion

The three-seed ensemble, test occlusion 0.55: 0.8894230769230769 (`occlusion_decision.json`). With confidence at least 0.99, chosen on validation, wrong verdicts are 0.004807692307692308 of occluded test notes, and 0.46634615384615385 of those notes receive a verdict.

Median fill on the earlier occlusion fine-tune remains 0.875 on all 208 notes. Learned inpainting scored 0.7259615384615384. Those are different checkpoints.

## Boxes

Synthetic test, 2052 notes: mean IoU 0.9083, IoU ≥ 0.5 on all of them, IoU ≥ 0.75 on 0.9839, zero false alarms on 230 empty images (`bbox_eval.json`). Dark synthetic images, 183: app path mean IoU 0.909 and IoU ≥ 0.5 on all of them (`bbox_app_eval.json`). Multi-note composites: `BBOX_EXTENDED.md`. Live camera box accuracy: NOT_MEASURED.

## Theory

Half-data network, McAllester 0.11271182900151713 on 487 held-out training notes. Not a bound on published PRMVT. Full-network zero-mean penalty 57.5892990573559.

## Not measured

Pi 5 latency, phone TFLite latency, user-study outcomes, NSTU/USD/EUR/INR authenticity, ModelNet/MOSI/ADNI/COCO view-count drops, live multi-note photographs.
