# Venue readiness (2026-09-29)

**These are judgement calls, not measurements.** No formula predicts acceptance. The ranges below come from how reviewers at each kind of venue usually weigh the evidence this project has and lacks. Use them to choose where to submit, not as probabilities to quote.

## What the work has now

| Strength | Evidence |
|---|---|
| A controlled, three-seed demonstration of view-count collapse in joint fusion networks, with a simple fix | `SAME_ARCH_RESULTS.md` (JaalTaka), `VCDS_MVPN.md` (MVP-N attention head) |
| Theory that matches the observations: a per-count bound, non-identifiability under fixed-view training, and a safety guarantee for the deployed threshold | `THEOREMS.md` Theorems 3, 8, 9, 10; Proposition 11 |
| An honest deployment study: a measured failure of the close-up checker on whole notes, and a safe policy with measured safety | `JAAL_VERDICT_FIXED.md` |
| Same-split baselines, paired tests, value-checked claims, reproducible metrics | `SOTA_COMPARISON.md`, `FINAL_SCAN.md` |
| A working offline Bangla assistive glass | `savior_glass/` |

## What reviewers will push on

| Weakness | Why it matters | Fix within reach? |
|---|---|---|
| Frozen ImageNet probes tie the best network on JaalTaka (60 paired tests, none significant) | An A* reviewer asks "why not just use a per-view classifier?" | Partly: reframe around joint fusion. The answer "per-view scoring is immune" must be in the paper |
| Prefix / view dropout is close to known modality-dropout training | Method novelty looks small | Partly: the novelty is the analysis (non-identifiability, safety bound, deployment), not the trick |
| The fix is architecture-dependent (the MVP-N concat head is a counterexample) | Weakens a general claim | Yes: state it as a finding, not a hidden limitation |
| One authentication dataset, 208 test notes, reused by about 110 runs | Small, and selection pressure | Needs a new test set (data collection) |
| Whole-note counterfeit evidence is about 4 physical notes | The deployment claim is thin | Needs your counterfeit photos |
| No user study, no Raspberry Pi 5 numbers | Assistive-technology venues require both | You plan to do these |

## Rough acceptance ranges by venue type

| Venue type | Examples | Now | With the user study + Pi 5 numbers + about 50 grouped whole-note counterfeit photos |
|---|---|---:|---:|
| A* ML / vision, main track | CVPR, ICCV, NeurIPS, ICML | about 3–8 % | about 8–15 % |
| A* accessibility / HCI | ASSETS, CHI | near 0 % (no user study) | about 20–35 % |
| Vision applications track | WACV (applications) | about 10–20 % | about 20–30 % |
| Q1 journal, applied ML / pattern recognition | Pattern Recognition Letters, Expert Systems with Applications, Engineering Applications of AI | about 25–40 % | about 40–55 % |
| Open-access engineering journal | IEEE Access, Sensors | about 45–60 % | about 55–70 % |
| Regional IEEE conference | ICCIT, TENSYMP, ICECE | about 65–85 % | about 75–90 % |

## What would move an A* submission most

1. **A user study** with blind participants on the real glass, including the "check by hand" policy. This is the single biggest step for an accessibility venue.
2. **A fresh authentication test set:** new notes, a new phone, a new day, grouped by physical note, and whole-note photos from the glass camera for the counterfeit check. This answers both the reuse and the deployment criticism.
3. **One more non-currency multi-view benchmark with an attention-fusion model trained end-to-end,** not only on frozen features. This would make "VCDS in attention fusion" a general claim.
4. **Framing:** submit the ML paper as "view-count shift in joint multi-view fusion: when it happens, why (non-identifiability), and how to deploy safely". Keep the assistive glass as the application. Do not compete on JaalTaka accuracy.

---

## Updated verdict, 2026-09-30 (after the serial-disjoint and watermark work)

**New since the last estimate:**
- The standard JaalTaka accuracy (about 98 %) is inflated by shared counterfeit prints. On unseen prints it is 88–90 %.
- A back-lit watermark detector lifts unseen-print accuracy to 95.5 % (p = 0.0002).
- A serial blacklist fails on new prints, as a proposition predicts.
- Prefix training's benefit is shown to depend on the fusion design.

These make the work more credible and more original. They do not change its scale: one authentication dataset, about 20 unseen counterfeit prints, no user study yet.

| Venue | Now | With the user study, Pi 5 numbers and glass-camera counterfeit photos |
|---|---:|---:|
| Regional (TENCON, ICCIT, ICAEE) | 75–90 % | 80–90 % |
| IEEE Access | 55–70 % | 60–75 % |
| Q1 journal (Pattern Recognition Letters, ESWA, EAAI) | 40–55 % | 50–65 % |
| A* workshop (CVPR / NeurIPS workshops) | 40–60 % | 45–65 % |
| NeurIPS Datasets & Benchmarks (as a benchmark audit: serial-disjoint protocol, whole-note set) | 10–20 % | 15–25 % |
| CVPR / ICCV / ECCV main | 5–10 % | 8–15 % |
| NeurIPS / ICML / ICLR main | 3–8 % | 5–10 % |
| ASSETS / CHI | ~0 % (no user study) | 20–35 % |
| IEEE THMS / T-RO / RA-L | ~5 % | THMS 15–25 % |

**Why 50–60 % at an A* main track is not reachable from lab work alone:**
1. The whole evidence base is one small counterfeit dataset (about 20 unseen prints in the hardest test).
2. The core training fix is simple and close to known view- and modality-dropout.
3. Strong frozen baselines tie the network on the standard split.

A main-track reviewer weighs scale and generality. No additional analysis on the same data changes that.

**Strongest contribution:** the combination of
- an audit showing that a public counterfeit benchmark is not print-disjoint (with the corrected protocol);
- a watermark-aware detector that recovers accuracy on unseen prints;
- a deployable safety policy with a finite-sample guarantee on an assistive device.

**Recommended submission order:**
1. Regional IEEE conference now (system and safety).
2. A* workshop on trustworthy or assistive AI (benchmark audit and watermark).
3. Q1 journal with the full study once the user study is done.
4. ASSETS with the user study.
5. NeurIPS D&B if you add glass-camera counterfeit photos and release the serial-disjoint split.
