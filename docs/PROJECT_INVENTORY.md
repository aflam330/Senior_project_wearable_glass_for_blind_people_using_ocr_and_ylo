# Project inventory

Organized 2026-09-29 from the folders on disk. Numbers below are copied from the cited files. Where the earlier list disagreed with those files, this page follows the files.

## 1. Folders

| Folder | Role |
| --- | --- |
| `savior_glass/` | Glass app: currency, OCR, objects, emotion, buttons, speech |
| `realtime_bangla_taka_detection/` | Detector, authenticity research, scripts, saved results |
| `data set/` | JaalTaka, detector composites, RAF-DB, COCO, OCR, counterfeit photos |
| `data set for comparison/` | BanglaTaka / Diverse, Bangla Money, NSTU-BDTAKA, Large Scale BDT, ModelNet40 |
| `paper_evidence/` | Measured write-ups and `CLAIM_REGISTRY.json` |
| `docs/` | Two-paper notes. This file. |
| `Unused/smart-glass/` | Old copy. Do not cite. |

## 2. Datasets

| Set | Where | What it is for | Use in a paper |
| --- | --- | --- | --- |
| JaalTaka | `data set/JaalTaka` | 1,390 notes × 6 views, genuine vs counterfeit | Authenticity test. Note-disjoint split 974 / 208 / 208, seed 42 |
| BanglaTaka raw | `data set/Bangladeshi_Paper_Currency_Raw` | Denomination photos used to build detector composites | Training source, not an external test |
| Diverse image set | `data set for comparison/A Diverse Image Dataset...` | Same photos as BanglaTaka raw (MD5 match) | Do not report as a second dataset. `CROSS_DATASET_TAKA.md` |
| Bangla Money | `data set for comparison/Bangla Money dataset Kaggle` | Denomination photos | External detector test |
| NSTU-BDTAKA | `data set for comparison/NSTU-BDTAKA dataset Mendeley` | Boxes and denomination close-ups | External detector test, with the leakage note in `CROSS_DATASET_TAKA.md` |
| Large Scale BDT DB 2026 | `data set for comparison/Large Scale BDT DB 2026 Kaggle` | 100,000 images, coins plus one demonetized-note class | Not the 9-note detector task |
| Counterfeit image set | `data set/Bangladeshi Counterfeit Currency Image Dataset` | Whole-note genuine and counterfeit photos, no note IDs | Deployment check only. Too small and ungrouped to train a new checker |
| ModelNet40 | `data set for comparison/ModelNet40 normal_resampled` | 12,318 point clouds, 40 classes | Rendered view-count probe, not currency |
| MVP-N | not on disk | — | Not measured |
| Detector composites | `data set/currency_yolo_data` | 22,333 synthetic composites (train 17,839 / valid 2,212 / test 2,282) | Detector train/val/test |
| Fresh composite test | `data set/currency_yolo_newtest` | 566 composites from test-split notes on COCO train2017 backgrounds | Second synthetic test, mAP50 0.992. `NEW_TEST_SET.md` |
| Tight-box crops | `data set/yolo_bbox_annotated` | 5,082 BanglaTaka crops with YOLO boxes | Bounding-box experiments (`results/bbox`) |
| COCO 2017 | `data set/coco2017` | val2017 backgrounds; object-mode subset | Composite backgrounds, object-naming test |
| RAF-DB | `data set/RAF-DB` | 15,339 faces, 7 emotions, official split | Emotion model, 86.5% |
| CK+48, FER2013 (`facial_expressions-master`) | `data set/` | Earlier emotion sets | Earlier experiments only. Not cited |
| Ekush + MatriVasha | `data set/OCR bangla data set` | Bangla handwritten letters | Ekush CNN (writer-disjoint split) |

## 3. What the glass does now

Currency mode loads `realtime_bangla_taka_detection/models/best.pt`, announces the denomination, and does **not** say genuine or counterfeit. `JAAL_VERDICT_ENABLED` is false. The spoken line ends with "জাল যাচাই করা হয়নি". Evidence: `paper_evidence/JAAL_VERDICT_FIXED.md`.

That switch is the measured response to the deployed verdict: on whole-note photos, PRMVT (trained on JaalTaka close-ups) was wrong on 47.7% of genuine Bangla Money images, 59.2% of genuine BanglaTaka images, 20.4% of genuine counterfeit-set images, and 40% of the counterfeit images in that set.

## 4. Numbers to cite

| Result | Value | File |
| --- | --- | --- |
| PRMVT, JaalTaka test, 1 view, seed 42, after the 2026-09-28 eval fix | 0.9711538461538461 | `results/qduig/prefix_ft/seed42/test_views_20260928/1view/test_metrics.json` |
| PRMVT, 6 views, same fix | 0.9807692307692307 | `.../6view/test_metrics.json` |
| CNN+ViT baseline, 1 view / 6 views | 0.7355769230769231 / 0.9182692307692307 | `results/qduig/eval/seed42/baseline/` |
| Detector test, 2,282 images | P 0.9949052081331781, R 0.9963485492831522, mAP50 0.9949300492610836, mAP50-95 0.8488711569132964, inference 13.863163540782018 ms on the RTX 3050 | `results/training_v2/test_eval/test_metrics.json` |
| Bangla Money denomination, no retraining | 91.5% of 1,536 photos | `paper_evidence/CROSS_DATASET_TAKA.md` |
| NSTU detection test | AP50 0.796 on 186 images | same |
| NSTU hand-held close-ups | 18.5% of 1,144 | same |
| Occlusion ensemble, 55% box, 6 views | 0.8894230769230769 | `results/robustness/occlusion_decision.json` |
| Rejection, threshold 0.99 from validation, occluded test | wrong-verdict share 0.004807692307692308; answered 0.46634615384615385 | same |
| ModelNet 10-class renders, 100 test objects | 0.64 at 1 view and 0.77 at 6 views, both trainings | `results/vcds_modelnet/subset10_seed42_e40.json` |
| Claims | 136, validation exit 0 | `paper_evidence/CLAIM_REGISTRY.json` |

Do not cite: wild 96.9% (those photos are the training source), the pre-fix PRMVT 6-view figure 0.97596, a full-network PAC-Bayes penalty as a certificate (it is 57.5892990573559), or "1-view 58.65% → 97.12%" as one model. 0.5865 is the old Q-DUIG checkpoint. 0.97115 is PRMVT.

## 5. Research pieces already written

`paper_evidence/VCDS_THEOREM.md`, `VCDS_UNIVERSAL.md`, `ORACLE_GAP_LAW.md`, `SEQUENTIAL_PROBLEM.md`, `SAFETY_FRAMEWORK.md`, `JAALTAKA_SEQ_BENCHMARK.md`, `OPEN_SOURCE.md`, `SOTA_BEAT.md`, `PI5_RESULTS.md`, `MOBILE_RESULTS.md`, `USER_STUDY_PROTOCOL.md`.

## 6. Still open

| Item | Who | Why it is still open |
| --- | --- | --- |
| Whole-note counterfeit photos, grouped by physical note | You | The only counterfeit whole-note set has 87 images and no note IDs. A new checker cannot be trained and tested on it without leakage. Until those photos exist, the glass must keep the jaal sentence off. |
| Run `savior_glass/scripts/benchmark_pi5.py` on the Pi 5 | You | The script is ready. No Pi numbers are in the repo. |
| User study with the planned participants | You | Protocol and analysis script exist. No participant data. |
| MVP-N | Optional | Not required for the JaalTaka claims. |
| Author names, affiliation, supervisor | You | Not in the repository. |
| Claude online mode | You | Needs an API key. It is a mode in the app, not a measured result. |

## 7. The pasted inventory checked against the files (2026-09-29)

Lines from the pasted list that do not match the files. The files win.

| Pasted line | What the files say | Source |
| --- | --- | --- |
| PRMVT seeds 43, 44 "pending" | Trained. 3-seed mean 96.5 ± 1.5 % at 1 view, 98.1 ± 1.0 % at 6 views | `results/qduig/prefix_ft/seed4{2,3,4}/test_views_20260928/`, `SAME_ARCH_RESULTS.md` |
| PRMVT 6-view 97.60 % | 97.60 % is the pre-fix value. After the NaN fix it is 98.08 % (204/208). `SOTA_BEAT.md` was corrected on 2026-09-29 | `CORRECTIONS.md`, `final_pass_20260929.json` |
| "1-view 58.65 % → 97.12 %" | Two different models (old Q-DUIG checkpoint vs PRMVT). The same-network comparison is `SAME_ARCH_RESULTS.md` | same |
| CNN+ViT 76.1 % → 92.3 % | 3-seed means (1 view / 6 views). Seed 42 alone: 73.6 % / 91.8 % | `WEAK_RESULTS_FIX.md` |
| roboeye_live.py crashes | Fixed. 150-frame webcam run, 11 FPS, exit 0 | `ROBOEYE_LIVE_FIXED.md` |
| PrototypeAuthenticator 512 vs 576 | Fixed: mismatched prototypes are rejected. On 2026-09-29 the prototypes were rebuilt from **train-split notes only** (80 + 80; before, test notes could enter). Genuine and fake similarity are nearly equal (0.915 vs 0.907 on a smoke-test note), so the method stays off | `roboeye/clip_zero_shot.py` |
| Close-up fallback "unused" | Measured and kept out on purpose. The NSTU-trained classifier scores 99.1 % on NSTU but 48 % on Bangla Money | `CROSS_DATASET_TAKA.md` |
| Deployed jaal verdict broken | Switched off on purpose (`JAAL_VERDICT_ENABLED = False`). Needs new grouped whole-note counterfeit photos | `JAAL_VERDICT_FIXED.md` |
| 129 claims | 136 claims, validator exit 0 on 2026-09-29 | `CLAIM_REGISTRY.json` |
| 15 datasets | 16 folders on disk (table in section 2). MVP-N was never downloaded and no claim needs it | `SCAN_COMPLETE.md` |
| "her_base 99.04 %" | A best-of-many result on a reused test split. Do not present it as a result | `NEW_TEST_SET.md` |
| Venue odds (A*, IEEE Access, regional) | Not from any file in this repository. Not a project result | — |

Also fixed on 2026-09-29: `savior_glass/scripts/test_speech.py` and `eval_assistive_stack.py` crashed when printing Bangla to a Windows console (cp1252). `scripts/train/run_same_arch.py` treated a killed run's per-epoch `checkpoint.pt` as finished; it now waits for `train_summary.json`.

## 8. Same-network comparison, finished 2026-09-29

Same PRMVT network, 3 seeds, 208 test notes (`paper_evidence/SAME_ARCH_RESULTS.md`).

| Training | 1 view | 6 views |
| --- | --- | --- |
| Fixed 6 views, shared BN | 71.2 ± 11.4 % | 98.1 ± 0.5 % |
| Prefix 1–6 views, shared BN | 98.2 ± 0.7 % | 98.7 ± 0.3 % |
| Fixed 6 views, per-count BN | 87.2 ± 9.7 % | 99.0 ± 0.0 % |
| Prefix 1–6 views, per-count BN (PRMVT) | 96.5 ± 1.5 % | 98.1 ± 1.0 % |

With shared BN, prefix training wins at 1 view on every seed (McNemar p from 9e-10 to 2e-22). With per-count BN, it wins on seeds 42 and 44 (p 8e-12 and 0.013) but not seed 43 (p 0.63). At 6 views no arm differs significantly. The claim to make is "prefix training removes the 1-view collapse and makes 1-view accuracy stable across seeds", not a fixed-size jump such as 58.65 → 97.12.

## 9. Second pass, 2026-09-29 (evening)

| Item | Result | File |
| --- | --- | --- |
| Jaal verdict | **Safe policy on** for 500 / 1,000 Taka: "সম্ভবত আসল" (likely genuine) only if p > 0.99959, else "হাতে যাচাই করুন" (check by hand); never "জাল". JaalTaka test: 0 / 88 counterfeits passed (CI 0–4.2 %). Whole-note photos: 0 / 25 passed; 1,889 app runs spoke "counterfeit" 0 times. It rarely confirms a whole note (0.4–2 % of independent genuine photos) | `paper_evidence/JAAL_VERDICT_FIXED.md` |
| Frozen backbones on the same split | ResNet-50 probe 98.1 % / 99.0 % (1 / 6 views); 60 paired tests against PRMVT and the prefix network: none significant | `paper_evidence/SOTA_COMPARISON.md` |
| Leaderboard | 12 three-seed methods plus 5 probes; 44 single-seed rows listed, not ranked | `paper_evidence/BENCHMARK_FINAL.md` |
| Recommended model | Prefix network with shared BN (chosen on validation: 0.9869 vs 0.9843) | same |
| Safety under bad light | Confidence rejection alone still gives 59–86 wrong verdicts out of 208 at 1 view; an image-quality gate fixed on clean validation brings that to 0 | `results/safety/`, `paper_evidence/SAFETY_FRAMEWORK.md` |
| Claims | 177 claims; 148 now checked against the number in their artifact; all pass | `paper_evidence/CLAIM_VALUE_AUDIT.md` |
| Stale numbers fixed | PRMVT 6-view in `SOTA_BEAT.md`; 5 LaTeX tables; HER ECE claim; theorem wording | `paper_evidence/CORRECTIONS.md` rows 10–19 |
| Paper | Full draft | `paper_evidence/PAPER_FINAL.md` |

Do not cite, in addition to section 4: HER ECE 0.0328 or 0.0245 (pre-fix scores; post-fix raw PRMVT ECE is 0.0146), and "PRMVT is state of the art on JaalTaka".

## 10. Third pass, 2026-09-29 (night)

| Item | Result | File |
| --- | --- | --- |
| Figures 1–12 | Redrawn from post-fix results; the old Q-DUIG-era set moved to `Unused/superseded_figures/` | `paper_evidence/figures/` |
| Extra seeds | MTPT, VCIE, APC, CRIS, MAVT, SAVS now have seeds 43–44; 18 three-seed methods; none beats the shared-BN prefix network | `paper_evidence/BENCHMARK_FINAL.md` |
| Approach A (synthetic whole notes) | Negative: real-photo AUC 0.689; would pass 14 / 20 real counterfeits; not deployed | `paper_evidence/JAAL_VERDICT_FIXED.md` |
| MVP-N (downloaded, 930 MB) | Attention head: prefix fixes view-count collapse (40.7 → 48.0 % at 1 view). Concat head: no gain | `paper_evidence/VCDS_MVPN.md` |
| JaalTaka serial audit | 279 / 322 readable counterfeit 500s share one serial; a lookup scores 90.2 %. Masking the serial costs PRMVT only 97.1 → 92.3 %, so it is mostly not reading it | `paper_evidence/JAALTAKA_SERIAL_AUDIT.md` |
| New theorems | Theorems 8–10 and Proposition 11, with proofs | `paper_evidence/THEOREMS.md` |
| FINAL_RESULTS.md | Rewritten; old log archived | `paper_evidence/FINAL_RESULTS.md` |
| Venue estimate | Judgement-based ranges | `paper_evidence/VENUE_READINESS.md` |
