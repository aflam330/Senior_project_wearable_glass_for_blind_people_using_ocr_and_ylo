# Theory catalogue (2026-09-30)

`THEOREMS.md` and `THEORY_FINAL.md` hold the proved statements. This page does not rename lemmas until they add up to fifteen. A theorem is listed only if the proof is in those files or below.

| # | Statement | Where |
|---|---|---|
| 1 | More views cannot hurt the Bayes error (monotonicity) | Theorem 1 |
| 2 | Fixed-view training need not control the one-view risk; prefix training can | Theorem 3, with a counterexample |
| 3 | The prefix-trained predictor is not identifiable from a single view count | Proposition 11 |
| 4 | Prefix training converges under standard stochastic-gradient conditions | Theorem 15, applied |
| 5 | Oracle gap between the best prefix and the evaluated policy | Theorem 5 |
| 6 | Rejection threshold is a constrained Neyman–Pearson / Chow rule | Theorem 7 |
| 7 | A max-of-validation threshold has a finite-sample false-accept bound on the training distribution | Theorem 8 |
| 8 | That bound gains a shift term off the training distribution | Theorem 9 |
| 9 | Serial blacklist recall equals the share of prints already seen, and is 0 on a new print | Proposition 12 |
| 10 | OR and AND fusion bounds on the two error types | Proposition 14 |
| 11 | With m prints, the catch-rate error is order 1/√m, not 1/√n | Theorem 16 |
| 12 | Quarter-decoding before a fixed output size discards high-frequency energy that a full decode keeps | Proposition 17, below |

Items the prompt asked for that are not separate theorems: "watermark–serial hybrid optimality", "multi-view fusion optimality", and "domain-shift bound" beyond Theorem 9. Those would need assumptions the measurements do not support. The hybrid does not beat the full-resolution fine-tune, so an optimality claim for it would be false.

## Proposition 17 (downsample-then-resize)

**Setting.** A band-limited image is sampled on a fine grid. One pipeline keeps those samples and resizes to a fixed grid G. The other first keeps every fourth sample in each axis (the JPEG reduced decode) and then resizes to the same grid G.

**Claim.** The second pipeline's output is a function of a strict subset of the samples the first pipeline uses. There exist images, identical on the kept quarter-samples and different on the discarded samples, whose resize to G differs. Therefore no function of the quarter-decoded image can reproduce every full-decode resize.

**Proof.** The quarter decode discards three of every four samples. Resize-to-G is a fixed linear filter of its input. Choose two fine-grid images that agree on the retained phase and differ by a high-frequency pattern supported on the discarded phase, with that pattern not in the kernel of the resize filter. Their quarter decodes match, so every later function matches, while their full-decode resizes differ. ∎

**What was measured.** On one genuine view, after both paths are resized to short side 256, Laplacian variance is 1814 (full decode) and 1047 (quarter decode). That is an illustration of Proposition 17, not a test-set accuracy. The accuracy consequence that was measured is the fine-tune gap in `DATA_PIPELINE_BUG.md`: 92.0 / 89.9 % versus 94.9 / 94.4 %.
