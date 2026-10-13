# Status of the ten requested improvements (2026-10-13)

One row per requested task. "Done" means a number exists in a result file. Nothing here is estimated or projected.

| # | Task and target | Status | What exists | What is needed to go further |
|---|---|---|---|---|
| 1 | UV fibre check 67 % → 85 % | **Done, with limits** | Blocked 5-fold: 103 / 111 genuine notes accepted (92.8 %), 71 / 71 plain-surface patches rejected (`UV_CHECK_IMPROVED.md`) | Counterfeit notes, printed negatives, glass-camera photographs |
| 2 | OCR on the Pi 18.7 s → 3–5 s | **Not measured** | Laptop, 4 threads: greedy decoder 6.5 s vs 8.9 s per 12-line page at the same validation CER; line-by-line speech. Earlier Pi run: canvas 640 was 6.9 s vs 20.2 s on one frame, +0.55 points raw CER (opt-in) | A run of `scripts/benchmark_pi5.py` on the Pi. No Pi number can be produced on the laptop |
| 3 | Objects mAP 0.604 → 0.80 | **Not started** | COCO-pretrained YOLOv8s, mAP@0.5 0.604 on 250 COCO images | Labelled photographs of the new classes (stairs, door, rickshaw, CNG, medicine). No such dataset is on disk |
| 4 | Online mode working | **Blocked** | Code path for two providers; `scripts/eval_online_mode.py` | API credit on the user's account, or a decision to download and run a local vision-language model (several GB; too slow for the Pi) |
| 5 | OCR on real photographs | **Not started** | Benchmark of rendered text, some placed in real photographs: CER 2.4 % | About 200 real photographs with their text typed out |
| 6 | Counterfeit prints 24 → 100+ | **Not possible from existing data** | Print-disjoint split with about 24 test prints; a print-locked collection tool | New counterfeit notes from a bank or the police. Augmenting the existing prints adds pictures, not prints |
| 7 | User study with 105 participants | **Not started (trials began)** | Protocol, logging, voice questionnaire | Ethics approval, participants, sessions |
| 8 | Proofs | **Not written** | The safe-threshold bound 1/(n+1) is a standard exchangeability result and is in the thesis | The other five items are empirical findings on one dataset, not theorems. A proof would need assumptions that the data cannot check |
| 9 | 20 new figures | **1 new figure** | `fig_uv_fibres.jpg` | Figures need new results |
| 10 | Four papers | **Not written** | Thesis, slides, feature summary | The user-study paper needs a user study; the others need the items above |

## Also fixed in this round (not on the list)

- Bangla speech no longer breaks at commas, digits and full stops; long texts keep their beginning; the Piper voice that fails on Bangla is tried once per run (`savior_glass/scripts/test_tts_queue.py`).

## Venue probabilities

Not given. There is no honest way to put a number on acceptance.
