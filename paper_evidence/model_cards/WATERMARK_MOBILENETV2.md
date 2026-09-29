---
license: mit
tags: [counterfeit-detection, banknote, bangladesh, watermark, mobilenetv2, onnx, int8]
datasets: [JaalTaka]
---

# Watermark-window classifier (MobileNetV2), Bangladeshi Taka 500 / 1,000

**What it does.** Scores the watermark window of a **back-lit** photo of a 500 or 1,000 Taka note as genuine or counterfeit. The window is found by registering the photo to a whole-note template (`savior_glass/modes/watermark_check.py`).

**Training.**
- Data: window crops of the JaalTaka serial-disjoint TRAIN notes, from view 6 (back-lit).
- Base: ImageNet MobileNetV2, fine-tuned 10 epochs.
- Epoch chosen on VAL AUC.
- Chosen over a MobileNetV3-Small by VAL AUC: 0.9957 vs 0.9943.

**Evaluation** (serial-disjoint TEST: counterfeit prints never seen in training; 197 notes whose window registered; `results/watermark/mobilenetv2.json`):
- accuracy 92.9 %, AUC 0.976;
- 2 / 98 genuine called counterfeit;
- 12 / 99 counterfeits missed.

**Files.**
- `watermark_mobilenetv2.onnx`: FP32, 8.9 MB.
- `watermark_mobilenetv2_int8.onnx`: static INT8, 2.6 MB, calibrated on 200 TRAIN crops; first convolution and classifier in FP32; 100 % the same test decisions as FP32.

**Out of scope.**
- Front-lit photos: the watermark is invisible.
- Denominations other than 500 / 1,000.
- Not validated on the glass camera.
- Never use it to tell a user a note is counterfeit; the reference app says only "watermark clear" or "not clear, check by hand".

**Known issue.** Pen marks on circulated genuine notes can fall inside the window.
