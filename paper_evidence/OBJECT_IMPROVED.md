# Object detection: full measurement and one fix (2026-10-13)

Sources: `savior_glass/scripts/eval_objects_coco.py` → `results/object_coco_val2017_full.json`; `savior_glass/scripts/eval_object_shrink.py` → `results/object_shrink_vs_full.json`. Same AP@0.5 code as the earlier 250-image run (predictions with confidence ≥ 0.15, VOC-style area).

No model was trained. The target of 0.75–0.80 was not reached and cannot be reached with these models.

## 1. The earlier number was optimistic

The thesis reports mAP@0.5 = 0.604 for YOLOv8s on a random 250-image subset of COCO val2017. On all 5,000 validation photographs the same model and code give 0.551.

| Model | Picture size 640 | Picture size 320 | 30 "blind-user" classes, 640 | GPU time per picture (640) |
|---|---:|---:|---:|---:|
| YOLOv8n (the one timed on the Pi 5) | 0.453 | 0.342 | 0.463 | 24 ms |
| YOLOv8s (first choice in the glass) | 0.551 | 0.441 | 0.553 | 26 ms |
| YOLOv8m | 0.607 | 0.513 | 0.604 | 50 ms |

- The 30 classes chosen beforehand as most relevant to a blind user (person, vehicles, furniture, bottle, cup, phone, book and so on) score the same as all 80. Choosing classes does not raise the number.
- Weakest of those 30 with YOLOv8s at 640: book 0.17, backpack 0.22, handbag 0.23, bench 0.34. Strongest: cat 0.86, toilet 0.79, dog 0.76, bus 0.75, person 0.73.
- These figures use a confidence cut of 0.15, so they are lower than the published COCO figures for the same weights, which integrate over all confidences.

## 2. A fix: stop shrinking the camera picture

`modes/object_mode.py` resized every frame to 320 × 240 "for fast inference" and then called the detector with its default input size of 640, which enlarges the small picture again. The detector did the same work on half the detail.

Measured with the glass's model on all 5,000 photographs, each first brought to the camera's size (640 × 480):

| Path | mAP@0.5, 80 classes | Detections at the announce confidence (0.50) | CPU time, 4 threads |
|---|---:|---:|---:|
| Shrunk to 320 × 240 (before) | 0.493 | 17,175 | 163 ms |
| Camera frame as it is (after) | 0.521 | 18,902 | 173 ms |

The frame now goes to the detector unshrunk: +2.9 points and 10 % more announceable detections. The CPU times were taken while another job was running on the same CPU, so the 10 ms difference is not a clean measurement; both paths run the detector at the same input size.

This is a correction of a mistake in the code, not a tuned setting. It was checked on COCO val2017 because that is the only labelled object data on disk.

## 3. Not done

- **New classes** (door, stairs, rickshaw, CNG, medicine, food): no labelled photographs of them are on disk. COCO has none of these classes.
- **Fine-tuning** on a subset of COCO classes: not run. One epoch on COCO train2017 takes over an hour on the 4 GB laptop GPU, three seeds and several epochs were outside this pass, and section 1 shows that restricting classes does not help.
- **Distance** ("chair two metres ahead"): no depth model was added and nothing was measured.
- **Pi 5 timing** of the unshrunk path and of YOLOv8s: not measured. The Pi figure in the thesis (207 ms) is for YOLOv8n.

## 4. What the thesis should say

Replace "mAP@0.5 = 0.604 on a 250-image subset" by the full-set figures: YOLOv8s 0.551 at 640, and 0.521 through the glass's own path after the fix (0.493 before).

## Addendum, 2026-10-13 evening: Objects365 and new classes

The request was to download Objects365, keep the classes a blind user needs, fine-tune YOLOv8s and reach mAP 0.70 or more. It was not done.

- **Size.** Objects365 has about 2 million images and several hundred gigabytes. The laptop has 66 GB free on its data drive and a 4 GB graphics card on which one epoch of COCO (118,000 images) takes over an hour.
- **Fit to the Bangladeshi street was not checked.** Its class list was not gone through for this note, and its photographs come from other countries. Whether it has usable examples of cycle rickshaws, CNG auto-rickshaws or open drains is unknown.
- **Bangla names** already exist for all 80 COCO classes in the object mode.
- **Distance.** No depth model was added. A monocular depth network on the Pi's CPU beside the detector has not been timed, and a spoken distance that is wrong is worse for a blind pedestrian than none.

What would be worth doing with a day of photography: 300 to 500 labelled photographs each of rickshaws, CNGs, stairs, doors and open drains from Dhaka streets, split by street, and a fine-tune of YOLOv8s that keeps the COCO classes. That is a data task first.

The measured state is unchanged from the sections above: YOLOv8s 0.551 on all of COCO val2017; 0.521 through the glass path after the shrink was removed.
