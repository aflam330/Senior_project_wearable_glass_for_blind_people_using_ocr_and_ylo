# Route to an A* venue: what is built, what only the authors can do (2026-10-01)

More glass modes or another fusion network on the same ~24 counterfeit prints will not move this past a workshop. Three artifacts can, in this order.

## 1. Print-disjoint dataset (NeurIPS Datasets and Benchmarks)

**Built.** `scripts/dataset/print_capture.py`, with a self-test of every rule.
- Per photo it stores print id, note id, currency, denomination, label, camera, lighting, blur and exposure.
- It fixes the split per print before training and refuses a print in two splits.
- `check` and `export` produce a release with split lists and a hashed `splits.json`.
- Plan and targets: `PRINT_DISJOINT_DATASET_PLAN.md`.

**The authors must:**
- find a bank or law-enforcement partner;
- collect about 100 unseen (test) counterfeit prints, 50 at minimum, with phone and glass photos under room light and backlight;
- decide on a second currency (INR or USD). The earlier rule was Taka only, and nothing was downloaded.

## 2. Watermark localizer, not a hand-drawn box (a vision main track, only if it wins)

**Built and measured on JaalTaka** (`WATERMARK_LOCALIZER.md`).
- **Method.** A learned corner regressor replaces template + SIFT + fixed box at run time.
- **Accuracy.** Unseen prints, three seeds:
  - it ties the SIFT path on the notes both can read (92.6 vs 92.9 %, McNemar p = 1.0 on every seed);
  - it reads the 25 photos SIFT cannot, reaching 222 / 222 at 92.6 ± 0.5 %.
- **Speed.** About 6× faster on a laptop CPU.
- **On the glass:** the app path is on the glass (93.2 % with INT8).
- **For another currency.** A hand-click window annotator and a manifest-to-labels converter are ready. The same training script runs on a new dataset; this was checked end to end on a throwaway set.

**Not shown yet.**
- **Labels.** They come from the old box.
- **Scope.** One currency.
- **Against the fair baseline.** The watermark alone is below the full-resolution ResNet-50 (94.4 %).

**The claim that would make it main-track.** The localizer, or a hybrid built on it, beats a fine-tuned full-resolution ResNet-50 at the print level on both currencies, over three seeds. That test can only be run on the new dataset. If it ties, the paper is route 1.

## 3. Blind-user study plus Pi timings (ASSETS)

**Built.** Capture guidance on the glass (`savior_glass/modes/capture_guide.py`, on whenever the watermark check is on).
- **Flow.** For 500 or 1,000 Taka it asks for the note against the light, rejects dark and blurry frames with a spoken reason, then checks. It gives up after 10 s with "check by hand" and never says "counterfeit".
- **Thresholds.** Set on validation. On test photos they accept 213 / 222 clean and 0 / 222 darkened or defocused.
- **Logging.** Every check is logged: participant, condition, seconds, rejected frames, outcome.
- **Study baseline.** `STUDY_CONDITION=unguided` switches rejection off.
- **Analysis.** `paper_evidence/user_study/analyze_study_events.py` is pre-registered in `STUDY_PREREGISTRATION.md`.
- **Tests.** Checked by `savior_glass/scripts/test_capture_guide.py`.

**The authors must:**
- recalibrate the two thresholds on glass-camera photos;
- register the study;
- run it with blind or low-vision participants;
- run `python scripts/benchmark_pi5.py --iters 100 --sustained 30` on the Pi 5. It now times both watermark paths.

## Venue by artifact

| Finished | Venue it can support | Status |
|---|---|---|
| Public print-disjoint set + audit + written split + second currency | NeurIPS Datasets and Benchmarks | tools ready; data not collected |
| Localizer that beats a fair ResNet-50 on both currencies, at print level | vision main track, arguable | method ready; ties on current data; needs the new set |
| Blind-user study on the real glass + Pi timings | ASSETS | software, logging, analysis ready; study and Pi not run |
| Current state (JaalTaka audit, serial split, hybrid, safe policy) | workshop | done |
