# Cross-dataset tests on other Bangladeshi Taka datasets (2026-09-29)

All sets are in `data set for comparison`. No image from them was used to train, tune or select the
deployed detector. Scripts: `realtime_bangla_taka_detection/scripts/eval/eval_external_taka.py` and
`eval_closeup_fallback.py`. Raw: `realtime_bangla_taka_detection/results/external/`.

## Which sets count as external

| Folder | Used? | Why |
|---|---|---|
| A Diverse Image Dataset for Bangladeshi Currency Recognition | **no** | byte-identical (MD5) to the BanglaTaka photos the detector's training composites were built from |
| NSTU-BDTAKA Detection/test | yes | 186 images, boxes labelled only "Taka" (no denomination): tests box finding |
| NSTU-BDTAKA Recognition/test | yes, with care | 9 denominations; 256×256 hand-held close-ups. **88 of its 1,127 original photos also occur in NSTU train**, and photo IDs are sequential video frames |
| Bangla Money (Kaggle) Training folders | yes | 1,536 photos of 8 of our 9 denominations (no 200); 101 photos of 1 taka, a class the detector does not know |
| Large Scale BDT DB 2026 | no | coins and one demonetized-note class, not the 9 notes |

None of these sets label genuine vs counterfeit, so counterfeit detection can only be measured on JaalTaka.

## Deployed detector (YOLOv8s `best.pt`, conf 0.25), no retraining

| Set | Result |
|---|---|
| NSTU detection test (186 notes) | class-agnostic AP50 **0.796**; at conf 0.25 precision 0.832, recall 0.694, IoU of matches 0.820 |
| **Bangla Money (1,536 notes)** | **91.5 % correct denomination**; no detection 3.9 %; 95.2 % correct when a note is found. Weakest: 1000 (78.4 %, mostly called 500), 10 (82.6 %), 500 (83.8 %) |
| NSTU hand-held close-ups (1,144) | **18.5 % correct**; no note found in 66.9 %; 55.9 % correct when found |
| Bangla Money 1-taka photos (101, unknown class) | 57 no detection, **37 announced as 5 taka**, 4 as 50, 3 as 2 |

The detector generalises to independent full-note photos (Bangla Money) but not to hand-held close-ups
where fingers cover part of a folded note (NSTU). An unknown denomination (1 taka) is announced as a known
one 44 % of the time; the detector has no "unknown note" output.

## Close-up fallback

Pipelines: P1 = YOLO only (current app); P2 = YOLO, and when YOLO finds nothing, a close-up MobileNet
classifier (answers only at softmax ≥ 0.65); P3 = the MobileNet alone. Rule written first: pick the highest
(correct − wrong) on development data, then score test once.

**With the app's existing MobileNet** (trained on BanglaTaka close-ups). Development: 1,500 NSTU-train photos. Development chose **P1**.

| | NSTU test (1,039, 88 leaked originals removed): correct / wrong / abstain | Bangla Money: correct / wrong / abstain |
|---|---|---|
| **P1 (chosen)** | 195 / 150 / 694 | 1405 / 71 / 60 |
| P2 | 330 / 608 / 101 | 1415 / 108 / 13 |
| P3 | 237 / 644 / 158 | 1326 / 129 / 81 |

The old MobileNet answers more close-ups, but most of those answers are wrong, so the current app is right to stay silent.

**With a new MobileNet trained on NSTU train** (`scripts/train/train_closeup_classifier.py`). Validation is 10 % of NSTU-train photos, grouped by original ID; best validation accuracy was 0.999, at epoch 2. Development (that validation set) chose **P3**.

| | NSTU test: correct / wrong / abstain | Bangla Money: correct / wrong / abstain |
|---|---|---|
| P1 | 195 / 150 / 694 | 1405 / 71 / 60 |
| P2 | 883 / 150 / 6 (85.0 %) | **1430 / 85 / 21 (93.1 %)** |
| **P3 (chosen on development)** | **1030 / 1 / 8 (99.1 %)** | **741 / 281 / 514 (48.2 %)** |

**Conclusion: in-domain NSTU scores do not show real-world generalisation.**
- The NSTU-trained classifier is near-perfect on NSTU's own validation and test splits, but gets **48 % on the independent Bangla Money photos**. It learned NSTU's recording conditions (hands, backgrounds, video sessions), not the notes.
- The pre-written rule picked P3 only because its development data came from the same non-independent source.
- **The app was not changed.** P2 (YOLO first, NSTU-trained classifier as fallback) is better than the current app on both sets: NSTU 85.0 % vs 18.8 %, Bangla Money 93.1 % vs 91.5 %, but with 14 more wrong answers on Bangla Money.
- Choosing P2 now would mean choosing on test sets. It stays a candidate until it is confirmed on new, independently collected hand-held photos.
