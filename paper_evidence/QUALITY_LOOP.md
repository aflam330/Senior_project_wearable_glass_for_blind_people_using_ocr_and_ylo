# Quality loop, 11–13 October 2026

One row per thing that was tried. "Kept" follows the project's rules: a change is kept if it was chosen on training or validation data, or fixed before a test was read, or if it corrects a mistake in the code. Earlier passes: `QUALITY_LOOP_FINAL.md`, `QUALITY_LOOP_ACCURACY.md`.

| # | Component | Before | Change | After | Kept |
|---|---|---|---|---|---|
| 1 | Page reading, decoder | Beam search: 8.9 s per 12-line page, validation CER 2.18 % | Greedy decoder | 6.5 s, 2.14 % | yes |
| 2 | Page reading, picture size | 1,200 px | Camera's own size | 5.3 s, 2.92 % (+0.74) | no (over the 0.5-point rule) |
| 3 | Page reading, speech | Nothing spoken until the page is done | Lines spoken as they are read | First line after about 5.5 s | yes |
| 4 | Line-by-line path | Validation CER rose to 4.95 % (background text read) | Main-text-block rule on pages too | 2.14 % | yes (fix) |
| 5 | Bangla speech | Cut at every comma, digit and full stop; long texts lost their start; failing voice retried each sentence | Letters decide the language; whole messages queued; failing voice tried once | 7 checks pass (`test_tts_queue.py`) | yes (fix); not yet heard on the Pi |
| 6 | UV fibre check, first test | — | Fibre finder + threshold from the first 60 % | 29 / 43 genuine notes accepted | superseded by 7 |
| 7 | UV fibre check, protocol | Threshold from front-side photographs, tested on back-side ones | Blocked 5-fold over capture order, same rule | 103 / 111 (92.8 %) | yes |
| 8 | UV claim | "Colour balance cancels" | Derived: only lamp intensity cancels | Claim corrected in code, thesis and summary | yes (correction) |
| 9 | Object evaluation | mAP@0.5 0.604 on 250 photographs | All 5,000 COCO val2017 photographs | 0.551 | yes (replaces the optimistic figure) |
| 10 | Object mode | Frame shrunk to 320 × 240 before a detector that enlarges it again | Frame passed as it is | 0.493 → 0.521 | yes (fix) |
| 11 | Object classes | 80 COCO classes | 30 classes fixed beforehand as relevant to a blind user | 0.553 against 0.551 | reported; no gain |
| 12 | Text detector runtime | PyTorch: 5.55 s, page 9.99 s | ONNX Runtime, full precision | 2.63 s, page 5.59 s, same validation CER | yes |
| 13 | Text detector precision | — | INT8 static quantisation | 1.29 s, page 4.28 s, validation CER 2.79 % (+0.65) | no as default (over the rule); opt-in |
| 14 | Text reading on real photographs | Never measured | 1,000 real word images, read once | CER 41.7 % Bangla, 64.9 % Latin script | reported |
| 15 | Text-finding settings for real photographs | Minimum confidence 0.2: real test CER 41.7 % Bangla, 64.9 % Latin | Four candidates on a real validation sample; winner within 0.5 points on rendered validation | Minimum confidence 0.1: 39.2 % and 62.5 %; nothing read 17 % and 38 % (was 25 % and 48 %); rendered validation 2.14 % → 2.58 % | yes |
| 16 | Online mode, cloud | — | 12-image run | Refused: no credit | blocked |
| 17 | Online mode, local model | — | SmolVLM-500M | Could not load: Windows Application Control blocks SciPy | blocked |

## Where the loop stopped, and why

| Component | Why no further step here |
|---|---|
| UV check | No counterfeit notes and no printed negatives. A more sensitive detector cannot be told apart from one that counts noise |
| Text reading speed on the Pi | Needs the Pi |
| Text reading on real signs | Settings are exhausted after row 15; the next step is training data from real photographs |
| Objects beyond COCO | No labelled photographs of the new classes |
| Counterfeit prints | No new counterfeit notes |
| User study | No participants yet |
