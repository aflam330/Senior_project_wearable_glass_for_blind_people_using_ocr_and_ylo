# The 25 listed weaknesses: what was done, with evidence (14 October 2026)

One section per weakness, in the order of the request. Every number is from the file named. "Not possible here" means the work needs something this laptop cannot supply (the Pi, real counterfeits, people), and says what.

| # | Weakness | Status | Evidence |
|---|---|---|---|
| 1 | Pi hot, 88.7 °C, throttled | **Analysed; governor added; not measured on the Pi.** The Pi idled at 71 °C with the fan at 7,800 rpm: cooling or room heat first. Model of the log: average load above ~50 % keeps it above 80 °C. Governor slows/pauses automatic object scans and preview when hot; button presses never wait | `THERMAL.md`, `savior_glass/thermal.py`, `scripts/test_thermal.py` |
| 2 | Text reading 18–39 s | **Partly.** One detector pass per capture (sign 34 s → ~18 s expected), instant "no text", 8-bit detector on the Pi (now within the 0.5-point rule), spoken "reading" and ticks. Smaller picture (880 px) rejected on test. Pi timing pending | `PI5_SECOND_RUN_2026-10-13.md`, `OCR_REAL_IMPROVED.md`, `scripts/pi_check_all.sh` |
| 3 | Real OCR 28.7 % / 50.6 % | **Improved earlier today** (from 39.2 % / 62.5 %), recogniser trained on real words and blended. No further real Bangla training data on disk | `OCR_REAL_IMPROVED.md` |
| 4 | Hand-held close-ups 18.5 % | **Fixed in part.** Detector fine-tuned with 6,000 synthetic hand-held close-ups; epoch and threshold chosen on NSTU validation. Test: 11.8 → 47.5 % right (laptop), 8.7 → 33.5 % (Pi); Bangla Money 89.6 → 94.1 %. Costs: more one-taka notes announced, multi-note total 97.0 % | `HANDHELD_FINETUNE.md` |
| 5 | Whole-note "likely genuine" 0.4–2 % | **Not changed.** The safe threshold is what keeps 0 of 88 counterfeits passing; lowering it on whole notes needs whole-note counterfeits to set it safely, which do not exist on disk | `JAAL_VERDICT_FIXED.md` |
| 6 | UV check, no real counterfeit | **Not possible here.** More detectors tuned on simulated sheets would measure the simulator. Needs real counterfeits or printed sheets under the lamp | `UV_SYNTHETIC_TEST.md` |
| 7 | Voice on the Pi unconfirmed | **Fixed in code** (render first, then play; English sample-rate bug; Piper Bangla voice; audio cache; espeak fallback). Pi listening pending | `VOICE_2026-10-14.md` |
| 8 | Voice commands with people | **Improved, people still untested.** Wake word "glass" + edge rule, chosen on 8 synthetic voices; 12 unseen voices: clean 100 %, whisper 98.9 %, street noise 0 dB 82.5 %, people talking 10 dB 54.3 % (was 3.2 %); false commands 13 of 5,040 (was 212) | `VOICE_COMMANDS_FAST.md` |
| 9 | Objects mAP 0.551 | **Not changed.** YOLOv8m reaches 0.607 but is too slow for the Pi; custom classes need labelled photos | `OBJECT_IMPROVED.md` |
| 10 | ~24 counterfeit prints | **Not possible here.** Augmenting 24 prints adds pictures, not prints | `SECOND_CURRENCY_SYNTHETIC.md` |
| 11 | Frozen ResNet ties the model | **Measured before**: on unseen prints the network + watermark fusion (95.0 %) is the result to lead with | `NOVEL_FUSION.md`, `BEAT_RESNET50_FINAL.md` |
| 12 | Online mode blocked | **Not possible here**: no API credit; Windows policy blocks SciPy for local vision models | `ONLINE_MODE.md` |
| 13 | Battery unmeasured | **Logger added**, sums the Pi 5's supply rails every 2 s and estimates hours; not run yet | `savior_glass/scripts/power_log.py` |
| 14 | No user study | **Ready since September** (protocol, voice questionnaire, field log, analysis); needs participants | `USER_STUDY_READY_FINAL.md` |
| 15 | No Bangla voice commands | **Measured, not deployed**: offline Whisper base, 45 % (clean) / 62 % (noise) on four unseen voices, no wrong or false command, ~1.7 s laptop | `BANGLA_VOICE_COMMANDS.md` |
| 16 | Objects COCO-only | **Not possible here**: Objects365 is far too large for this laptop; no labelled blind-user classes on disk | — |
| 17 | 1-taka announced 22.8 % | **Unchanged**: needs out-of-scope notes for validation that are not in the test set; none on disk | `OPEN_SET_REJECTION.md` |
| 18 | Watermark false-counterfeit 6.7 % | **Source not found**: no result file has a 6.7 % watermark false-counterfeit rate. The glass never says "counterfeit"; a watermark doubt becomes "check by hand" | `SAFETY_FRAMEWORK.md` |
| 19 | DMWR ties MobileNetV2 | **Unchanged**: the tie at 46× fewer parameters is the result | `WATERMARK_DMWR.md` |
| 20 | No second currency | **Declined with reasons** (13 Oct): edited genuine images measure the edits | `SECOND_CURRENCY_SYNTHETIC.md` |
| 21 | No cross-camera test | **Partly**: external photo sets (Bangla Money 91.5 %, NSTU) are other people's cameras; the glass camera itself is untested | `CROSS_DATASET_TAKA.md` |
| 22 | No session-disjoint test | **Not possible**: JaalTaka has no session or camera IDs; print-disjoint split used instead | `SECOND_COUNTERFEIT.md` |
| 23 | Watermark localizer ties | **Unchanged** | `WATERMARK_LOCALIZER.md` |
| 24 | Occlusion 88.9 % | **Measured before**: occlusion fine-tune 88.9 % (3-seed ensemble); with rejection 0.5 % wrong verdicts | `OCCLUSION_LIMITATION.md` |
| 25 | Low light 58.7 % | **Measured before**: 58.7 % was the untouched network; gamma repair 83.2 %, robust fine-tune 93.75 % (seed 42) at −1 point clean | `LOWLIGHT_FIX_RESULTS.md` |

## Requested algorithms

- **Triple-signal fusion (watermark + UV + network):** not measurable. No note has all three: the UV photographs are of our own genuine notes, the watermark and network scores are on JaalTaka notes. The two-signal fusion that can be measured (network + watermark, validation-fitted) gives 95.0 % on unseen prints (`NOVEL_FUSION.md`).
- **Adaptive UV detection (wavelets, Gabor, U-Net, FFT, voting):** not built, for the reason under #6.
- **Thermal-aware operation:** built (`THERMAL.md`).
- **Natural Bangla voice:** built (`VOICE_2026-10-14.md`).
- **Second currency:** declined (#20).


## Added 15 October 2026

The hand-held fine-tune crashed twice before it completed (out of memory beside a Whisper study; then a session stop); cause and fix in `CRASH_FIXED.md`. Bugs found on the way: `BUGS_FIXED.md`. Every check that runs on the laptop: `TEST_RESULTS.md`.
