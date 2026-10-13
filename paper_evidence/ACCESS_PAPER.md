# A Multi-Modal Assistive Glass for Visually Impaired Users in Bangladesh: Design, Measurement and What Did Not Carry Over

*Draft for an application journal (IEEE Access style), 2026-10-13. Every number is from a result file of the project; the file is named in the section or in `CLAIM_REGISTRY.json`. Sections marked NOT MEASURED have no number and must not be given one. Authors: MD Abdun Fattah Lam, Mukut Protim Memo, Tanjila Hossain Tapti; supervisor Mohammad Shidujaman.*

## Abstract

We describe SARA, a camera glass on a Raspberry Pi 5 that speaks Bangla and works without the internet. It reads Bangla and English text, names objects, names the nine Bangladeshi Taka notes, and gives a cautious answer on whether a 500 or 1,000 Taka note is genuine. We report each function on held-out data and the speed on the device, and we report with the same weight what did not carry over from the laboratory: a Taka detector trained on automatically labelled pictures reaches mAP@0.5 of 0.995 on held-out pictures and 91.5 % on 1,536 photographs by other people, but 18.5 % on close-ups with fingers; a text reader with 2.5 % character error on a rendered benchmark has 39 % (Bangla) and 62 % (Latin script) on real street words; an ultraviolet fibre check that accepted 92.8 % of genuine notes also accepted simulated printed copies until a colour condition was added. The counterfeit answer never says "counterfeit": none of 88 test counterfeits was passed as genuine. A study with visually impaired users is prepared and has not been run.

## 1. Introduction

A blind person in Bangladesh who wants to read a label or know which note is in the hand usually has to ask someone. Commercial aids are costly, built for English and often need a phone or a connection. The contribution of this paper is not a new network. It is (i) a complete device with every function measured separately and on the target hardware, (ii) an account of four places where a good laboratory number did not survive contact with other data, and (iii) the engineering that made the device usable: a safe spoken verdict, a speech queue that keeps Bangla sentences whole, and a text detector that runs twice as fast with identical output.

## 2. System

- **Hardware.** Raspberry Pi 5 (8 GB), Camera Module 2 at 640 × 480, five push-buttons (mode, action, read, volume up, volume down), a vibration motor, an earphone, a USB power bank, a 3D-printed frame.
- **Modes.** Text, objects, currency, emotion, and an optional online description. One button cycles them; every change is spoken.
- **Speech.** espeak-ng for Bangla and Piper for English, chosen per sentence. Letters decide the language; punctuation and digits stay with the sentence they stand in. One message is spoken whole; the lines of a page are never dropped from the queue.
- **Software.** Python; each mode implements the same four functions; models are loaded on first use; a field log records every event.

## 3. Methods and results by function

### 3.1 Taka recognition

5,073 background-free note photographs were pasted onto real photographs with random tilt, rotation, size, colour and blur; the program wrote the box labels, giving 22,333 pictures with no manual annotation. YOLOv8s, 80 epochs.

| Test | Result |
|---|---|
| 2,282 held-out pictures of the same kind | precision 0.995, recall 0.996, mAP@0.5 0.995, mAP@0.5:0.95 0.849 |
| 1,536 photographs by other people (Bangla Money) | 91.5 % correct value |
| 1,144 hand-held close-ups (NSTU-BDTAKA) | 18.5 % correct value |
| Several notes in one picture (count and total) | see `NEW_FEATURES.md` |

### 3.2 Counterfeit answer

A multi-view network trained on random prefixes of the six views of the JaalTaka set stays accurate when the user captures fewer views: 98.2 ± 0.7 % with one view against 71.2 ± 11.4 % for the same network trained on six (three seeds, 208 test notes). An audit of serial numbers showed that the usual split of this set puts copies of one print in training and test; on a split with unseen prints accuracy is about 90 %, and 95.0 % with a check of the back-lit watermark. The glass says "likely genuine" only above the highest score of any validation counterfeit and otherwise "check by hand": 0 of 88 test counterfeits passed. On whole-note photographs by other people it confirms 0.4–2 % of genuine notes; it is safe and seldom decisive.

### 3.3 Watermark with a small physics-based classifier (DMWR)

The back-lit photograph is divided by a blurred copy of itself, which removes the lamp; the result is compared with the pixel-wise median of genuine training notes of that value; a small two-stream network reads the image and the residual. 47,818 parameters: 91.5 ± 0.3 % on unseen prints, against 92.9 % for a fine-tuned MobileNetV2 with 2.23 million (McNemar p = 0.375, 0.688, 0.375).

### 3.4 Ultraviolet fibre check (prototype)

Genuine 200, 500 and 1,000 Taka notes show glowing fibres on dull paper under a UV lamp. The check separates the lamp's blue light from fluorescence in the same picture and counts thin fluorescent structures. On 111 photographs of genuine notes it accepted 92.8 %. A simulation with fibre-free printed sheets then showed that it accepted 93 % of those as well once JPEG compression was applied. With a colour condition, 85.6 % of genuine photographs are accepted and 88.6 % of held-out simulated sheets are rejected. No counterfeit note has been tested; the check is off by default.

### 3.5 Text reading

EasyOCR with contrast enhancement, tuned detector settings, a rule that keeps only the main block of text, and clean-up rules.

| Data | Character error |
|---|---|
| Rendered benchmark, test set, before / after this work | 9.3 % / 2.4 % |
| Same, with the settings now in the glass | 2.5 % (Bangla 3.3 %, English 1.7 %) |
| 1,000 real word photographs (ICDAR 2019 MLT), first reading | Bangla 41.7 %, Latin script 64.9 % |
| Same, confidence filter chosen on a real validation sample | Bangla 39.2 %, Latin script 62.5 % |
| Same, recogniser fine-tuned on real words | see `OCR_REAL_IMPROVED.md` |

### 3.6 Objects

COCO-pretrained YOLOv8. On all 5,000 COCO val2017 photographs: YOLOv8n 0.453, YOLOv8s 0.551, YOLOv8m 0.607 (mAP@0.5, predictions with confidence ≥ 0.15). An earlier figure of 0.604 for YOLOv8s came from 250 photographs. The object mode shrank each frame to 320 × 240 before a detector that enlarges it again; without the shrink the glass path scores 0.521 instead of 0.493.

### 3.7 Emotion

EfficientNet-V2-S with a mirrored second view: 86.5 % on the official RAF-DB test set (macro-F1 0.795).

## 4. Speed

Raspberry Pi 5, 100 calls per part (1 October 2026): currency answer 327 ms, watermark check 50 ms, object scan 207 ms (YOLOv8n), emotion 48 ms, one text capture 18.7 s. Thirty minutes of mixed use: 1.48 GB peak memory, 77.1 °C peak, no throttling.

After that measurement, on a laptop CPU limited to four threads: a greedy decoder reads a 12-line page in 6.5 s instead of 8.9 s at the same validation error; the text detector through ONNX Runtime takes 2.6 s instead of 5.6 s with identical output; an INT8 detector is faster again but costs 0.57–0.65 points of validation error and is not the default. **The Pi has not been timed with these changes.**

## 5. What did not carry over

1. **Close-ups of notes.** 91.5 % on whole notes, 18.5 % when fingers cover a folded note.
2. **The counterfeit benchmark.** About 98 % on the usual split, about 90 % on unseen prints. With about 24 test prints, differences of a few points cannot be established.
3. **Rendered text to real text.** 2.5 % to 39–62 %.
4. **The first UV check.** It recognised genuine notes and, in simulation, printed copies alike.
5. **The glass camera.** In a first hands-on test, currency recognition was unreliable and Bangla text was not read through the 640 × 480 fixed-focus camera. NOT MEASURED beyond that observation.

## 6. User study

NOT RUN. The protocol (within-subject, denomination accuracy and time per note as primary outcomes, SUS and NASA-TLX), the consent text, the logging and the analysis scripts exist (`USER_STUDY_PROTOCOL.md`, `STUDY_PREREGISTRATION.md`). Trial sessions with one visually impaired participant have started; no result is reported.

## 7. Limitations

One counterfeit collection; no counterfeit under UV; no second currency; the Pi not timed after the speed work; object classes are those of COCO; battery life not measured.

## 8. Conclusion

The parts of an affordable, private assistive glass for Bangla-speaking users can be built from open models and low-cost hardware. Where our numbers were checked against other people's data, they fell, sometimes by a factor of ten. We think the list in Section 5 is the most useful part of this paper for anyone building a similar device.
