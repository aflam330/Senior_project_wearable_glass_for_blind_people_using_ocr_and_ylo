# Status of the ten requested improvements (end of 13 October 2026)

One row per requested task. A number appears here only if it is in a result file. Nothing is estimated.

| # | Task and target | Outcome | Measured | Target met? |
|---|---|---|---|---|
| 1 | UV fibre check 67 % → 85 % | Blocked 5-fold evaluation; threshold 6 fibres; spoken guidance | 103 / 111 genuine notes accepted (92.8 %); 71 / 71 plain-surface patches rejected (`UV_CHECK_IMPROVED.md`) | **Yes for genuine notes.** No counterfeit tested |
| 2 | OCR on the Pi 18.7 s → 3–5 s | Text detector through ONNX Runtime (default); INT8 opt-in; greedy decoder; line-by-line speech | Laptop, 4 threads: detector 5.55 → 2.63 s, page 9.99 → 5.59 s, same validation CER; INT8 page 4.28 s at +0.65 points (`OCR_SPEED_IMPROVED.md`) | **Unknown.** No Pi measurement |
| 3 | Objects mAP 0.604 → 0.80 | Full evaluation; one fix (no needless shrink) | YOLOv8s 0.551 on 5,000 photographs (the 0.604 was a 250-image sample); glass path 0.493 → 0.521 (`OBJECT_IMPROVED.md`) | **No.** The honest figure went down, then up by 3 points |
| 4 | Online mode working | Two routes tried | Cloud: refused, no credit. Local model: blocked by Windows Application Control (`ONLINE_MODE.md`) | **No.** Blocked outside the code |
| 5 | OCR on real photographs | First real test; settings study with its own real validation sample | 1,000 real word images: CER 41.7 % → 39.2 % Bangla, 64.9 % → 62.5 % Latin script; nothing read 25 % → 17 % and 48 % → 38 % (`OCR_REAL.md`) | **Tested: yes.** Good: no. Rendered benchmark is 2.4 % |
| 6 | Counterfeit prints 24 → 100+ | Nothing | — | **No.** Needs real counterfeit notes; augmenting 24 prints adds pictures, not prints |
| 7 | User study with 105 participants | Nothing new; package was already complete | — | **No.** Needs ethics approval and participants |
| 8 | Proofs | Six items sorted into proved, proved under a stated model, and not a theorem | `THEORY_STRONG.md`; one earlier claim corrected (colour balance does not cancel in the UV method) | Partly: 4 statements proved; cross-currency transfer is not a theorem |
| 9 | 20 new figures | 8 new figures, each from a result file | `figures/fig41`–`fig47`, `fig_uv_fibres.jpg` | **No (8 of 20).** Figures need results |
| 10 | Four papers | Index of the existing drafts and what each lacks | `PAPERS/README.md` | **No.** Three drafts existed; the user-study paper cannot be written without a study |

## Also done in this pass

- Bangla speech no longer breaks at commas, digits and full stops; long texts keep their beginning; the Piper voice that fails on Bangla is tried once per run.
- The thesis and `SARA_Features.pdf` carry all of the above, including the corrected object figure and the real-photograph results.

## What only the team can supply

1. A run of `scripts/benchmark_pi5.py` on the Pi (task 2), once as it is and once with `OCR_DETECTOR=onnx_int8`.
2. UV photographs through the glass camera, printed sheets without fibres, and counterfeit notes at a bank or with the police (tasks 1 and 6).
3. Real photographs of Bangla signs and labels from Bangladesh with their text typed out, for training and testing (task 5).
4. Labelled photographs of the new object classes (task 3).
5. API credit, or the Windows policy lifted for the Python packages (task 4).
6. Ethics approval and participants (task 7).

## Venue probabilities

Not given. There is no honest way to put a number on acceptance.
