# Serial-number and watermark counterfeit detection for Bangladeshi Taka (2026-09-30)

## Summary

- **The back-lit watermark window is the most transferable counterfeit cue in JaalTaka.** It generalises to counterfeit prints never seen in training, where full-note classifiers drop sharply.
- **Adding it to the prefix network cuts missed new-print counterfeits from 26 / 101 to 7 / 101** (6 views; 87.8 % → 95.5 %; exact McNemar p = 0.0002), with 3 / 121 genuine notes flagged (2.5 %).
- **A serial blacklist catches only prints it has already seen:** 0 of 19 unseen-serial counterfeits on the standard split. Once the watermark is included, the serial adds nothing.

## Domain facts used

- Genuine Taka notes carry a watermark (portrait plus a denomination electrotype) in a blank oval window, visible when the note is held against light. JaalTaka **view 6 is back-lit**, so the window can be measured there.
- JaalTaka counterfeits do have a watermark area, but it is an imitation: blank, faint, or a printed portrait (`results/watermark/crops/`).
- Genuine notes have unique serials. JaalTaka counterfeits share a few printed serials: 279 of the 322 readable counterfeit 500 BDT notes carry 3274658 (`JAALTAKA_SERIAL_AUDIT.md`).

## Method

| Component | What it does | Script |
|---|---|---|
| Watermark window | View 6 registered to the whole-note template (SIFT + RANSAC); window (0.70–0.93, 0.33–0.85) of the note cropped; placed below the upper-right serial so no digits enter the crop. Registered on 1,261 of 1,390 notes | `scripts/eval/watermark_features.py` |
| W_hand | Logistic regression on 4 hand-crafted window features: Laplacian variance, grey std, edge density, window/note brightness | `scripts/eval/watermark_hybrid.py` |
| W_deep | Logistic regression on frozen ResNet-50 features of the window; C chosen on VAL | same |
| S_list | Serial blacklist: first 6 OCR digits among TRAIN counterfeit serials | same |
| S_dup | Bundle anomaly: the same full serial on ≥ 2 TRAIN notes | same |
| Hybrid | Logistic regression fitted on VAL over [network logit, watermark logit, watermark-missing flag, (serial flag)] | same, `hybrid_serial_split.py` |

## Results on the standard seed-42 split (208 test notes; `results/watermark/hybrid.json`)

| Detector | Accuracy | AUC | Genuine called counterfeit | Counterfeits missed | Unseen-serial counterfeits caught |
|---|---:|---:|---:|---:|---:|
| W_hand (191 registered) | — | 0.853 | — | — | — |
| W_deep (191 registered) | 98.9 % | 0.992 | 1 / 105 | 1 / 86 | 17 / 18 |
| S_list | 89.9 % | 0.882 | 1 / 120 | 20 / 88 | 0 / 19 |
| S_dup | 86.1 % | 0.840 | 3 / 120 | 26 / 88 | 0 / 19 |
| PRMVT, view 1 | 97.1 % | 0.988 | 3 / 120 | 3 / 88 | 17 / 19 |
| Hybrid: view 1 + watermark (+ serial) | 99.0 % | 0.992 | 1 / 120 | 1 / 88 | 18 / 19 |
| Hybrid without the serial | 99.0 % | 0.992 | 1 / 120 | 1 / 88 | 18 / 19 |
| PRMVT, 6 views | 98.1 % | 0.994 | 2 / 120 | 2 / 88 | 17 / 19 |
| Hybrid: 6 views + watermark | 99.0 % | 0.992 | 1 / 120 | 1 / 88 | 18 / 19 |

The fitted weight of the serial flag is 0.015 (1 view) and 0.145 (6 views), against 0.7–0.8 for the watermark. With the watermark present, the serial is redundant.

## Results on counterfeit prints never seen in training (serial-disjoint split, 222 test notes)

The split rule is in `make_serial_split.py`. Results: `results/watermark/serial_split_eval.json` and `hybrid_serial_split.json`.

| Detector | Accuracy | AUC | Genuine called counterfeit | Counterfeits missed |
|---|---:|---:|---:|---:|
| ResNet-50 probe, 1 view | 85.1 % | 0.915 | 1 / 121 | 32 / 101 |
| ResNet-50 probe, 6 views | 83.8 % | 0.948 | 1 / 121 | 35 / 101 |
| W_deep watermark alone (197 registered) | 88.3 % | 0.939 | 1 / 98 | 22 / 99 |
| Probe view 1 + watermark | 91.4 % | 0.931 | 5 / 121 | 14 / 101 |
| Prefix network (trained on this split), 1 view | 90.1 % | 0.937 | 1 / 121 | 21 / 101 |
| **Prefix network, 1 view + watermark** | **93.2 %** | 0.951 | 7 / 121 | 8 / 101 |
| Prefix network, 6 views | 87.8 % | 0.961 | 1 / 121 | 26 / 101 |
| **Prefix network, 6 views + watermark** | **95.5 %** | 0.958 | **3 / 121** | **7 / 101** |

- **Paired tests** (exact McNemar, hybrid against network alone): 6 views p = 0.0002 (19 notes fixed, 2 broken); 1 view p = 0.17.
- **Target "< 5 % false counterfeit on genuine":** met by the 6-view hybrid (2.5 %), not by the 1-view hybrid (5.8 %).

## What limits these results

- **Pen marks.** Several genuine notes carry pen marks in the window (circulated notes). A texture feature can partly respond to handwriting rather than the watermark. The learned model was not tested against this confound.
- **Registration.** 9 % of view-6 photos did not register. Those notes get a neutral watermark score.
- **Deployment.** The watermark needs back-lighting. The glass's whole-note photos are front-lit, so the window cannot be scored from them. A guided "hold the note up to the light" capture is the deployable form; it is READY_FOR_DEVICE, not measured.
- **Serials on real whole-note photos.** See the section below.

## Serial blacklist on real whole-note photos

Script: `scripts/eval/serial_whole_note.py` → `results/watermark/serial_whole_note.json`. EasyOCR reads both serial boxes of each YOLO crop, in both orientations. A photo is flagged if a 6-digit prefix matches a JaalTaka TRAIN counterfeit serial.

| Photos | Flagged as a known counterfeit print | A serial read at all |
|---|---:|---:|
| Counterfeit-set counterfeit originals (4 physical notes) | 12 / 25 | 17 / 25 |
| Counterfeit-set genuine | 0 / 150 | 80 / 150 |
| Bangla Money genuine | 0 / 150 | 39 / 150 |
| BanglaTaka genuine | 0 / 150 | 93 / 150 |

- **All 12 flagged photos are one physical note:** the 1000 BDT burst reading serial 2628724, which also appears among JaalTaka training counterfeits.
- The other three counterfeit groups (the 500 BDT series, and two 1000 BDT notes) were not flagged.
- **No genuine photo was flagged (0 / 450; 95 % upper bound 0.8 %).**

**Reading.** On real photos a serial blacklist is safe for genuine notes and catches a counterfeit only when its print is already known: here 1 of 4 prints. That is what Proposition 12 predicts. It is a useful extra signal for known bundles, not a general detector.

## Three seeds on unseen prints (added 2026-09-30)

Prefix network retrained on the serial-disjoint split with seeds 43 and 44. Same watermark classifier; combination fitted on VAL per seed. Source: `results/watermark/hybrid_serial_split_seeds.json`.

| | 1 view | 6 views |
|---|---:|---:|
| Prefix network, mean ± sd over seeds 42–44 | 89.9 ± 0.7 % | 89.3 ± 1.3 % |
| **Network + watermark** | **93.8 ± 1.0 %** | **94.7 ± 0.9 %** |
| Genuine called counterfeit, hybrid (per seed, of 121) | 7, 5, 2 | 3, 5, 4 (mean 3.3 %) |
| Counterfeits missed, hybrid (per seed, of 101) | 8, 10, 9 | 7, 9, 7 |
| Exact McNemar, hybrid vs network (per seed) | p = 0.17, 0.049, 0.0063 | p = 0.0002, 0.096, 0.0074 |

The watermark improves the network in all 6 seed-and-view comparisons, significantly (p < 0.05) in 4. The six-view hybrid keeps genuine false alarms under 5 % on every seed.

## Device-sized watermark model (added 2026-09-30)

MobileNetV3-Small fine-tuned on the window crops of the serial-disjoint TRAIN notes; epoch chosen on VAL AUC (`scripts/train/train_watermark_mobilenet.py` → `results/watermark/mobilenet.json`).

| Watermark model, unseen prints (197 registered test notes) | Accuracy | AUC | Genuine called counterfeit | Counterfeits missed |
|---|---:|---:|---:|---:|
| ResNet-50 features + logistic regression | 88.3 % | 0.939 | 1 / 98 | 22 / 99 |
| **MobileNetV3-Small, fine-tuned** | **91.9 %** | **0.962** | 1 / 98 | 15 / 99 |

**Export.** FP32 ONNX, 6.1 MB, gives the same decisions as PyTorch on 100 % of test crops. Three INT8 attempts all changed too many decisions and are not used: dynamic 44.7 %, static QDQ 53.3 %, Conv-only 62.9 % agreement (`Unused/README.md`).

**In the app.** `savior_glass/modes/watermark_check.py`, behind `WATERMARK_CHECK_ENABLED = False`. It says "জলছাপ স্পষ্ট" (watermark clear) or "জলছাপ স্পষ্ট নয়, আসল কিনা হাতে যাচাই করুন" (not clear, check by hand), never "counterfeit". Running the app module on the unseen-print test photos: window found on 196 / 222; accuracy 93.4 %; 1 / 98 genuine "not clear"; 12 / 98 counterfeit "clear" (`results/watermark/app_module_check.json`). Not validated on the glass camera.
